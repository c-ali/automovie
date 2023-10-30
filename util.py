import re
import cv2
import openai
from pytube import Search
import os
from pathlib import Path
from prompts import *

def remove_prefixes_and_split(text):
    # Split the text by newline
    lines = text.split('\n')

    # Use a regular expression to remove prefixes like "1.", "5.", "13.", etc.
    cleaned_lines = [re.sub(r'^\d+\.\s*', '', line) for line in lines]

    # Remove empty lines
    cleaned_lines = [l for l in cleaned_lines if len(l) > 0]

    return cleaned_lines

def split_after_n_words(text, n=7):
    words = text.split(" ")
    split_lines = []
    line = ""
    for i, word in enumerate(words):
        line += word
        line += " "
        if (i + 1) % n == 0 and (i + 1) != len(words):
            line = line[:-1]
            split_lines.append(line)
            line = ""
    split_lines.append(line)
    return split_lines

def split_after_n_chars(text, n=40):
    words = text.split(" ")
    split_lines = []
    line = ""

    for word in words:
        # If adding the next word exceeds the character limit, append the current line
        if len(line) + len(word) > n:
            split_lines.append(line.strip())
            line = ""
        line += word + " "

    # Add the last line if it's not empty
    if line:
        split_lines.append(line.strip())

    return split_lines
def add_caption_to_frame(img, caption, add_linebreaks=True, high_res=False):
    # Write some Text^
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (15, 750 if high_res else 500)
    dy = 20
    fontScale = 0.55
    fontColor = (255,255,255)
    thickness = 2
    lineType = 3

    if not add_linebreaks:
        cv2.putText(img, caption,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
    else:
        fontScale = 0.7
        chars_to_split = 60 if high_res else 40
        broken_caption = split_after_n_chars(caption, chars_to_split)
        for i, text_line in enumerate(broken_caption):
            y = bottomLeftCornerOfText[1] + (i -len(broken_caption)) * dy
            cv2.putText(img, text_line,
                        (bottomLeftCornerOfText[0], y),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
def apply_caption(lb, caption, add_linebreaks=True, high_res=False):
    for i in range(len(lb.tree_final_imgs)):
        arr = lb.tree_final_imgs[i].copy()
        add_caption_to_frame(arr, caption, add_linebreaks=add_linebreaks, high_res=high_res)
        lb.tree_final_imgs[i] = arr



# LLM story, prompt and music recomendation functions
# When a LLM is not None, these functions use a llama-cpp-python locally. Else, ChatGPT AI is used
def create_prompts(raw_story, temperature=0.6, llm=None):
    if llm is not None:
        llama_postprompt = get_llama_postprompt("visual description of the story")
        ret = llm(get_prompt_prompt(raw_story)+llama_postprompt, max_tokens=2500, stop=["Q:"], echo=True)["choices"][0]["text"]
        ret = get_substring_after(ret, llama_postprompt)
    else:
        ret = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        temperature=temperature,
        prompt=get_prompt_prompt(raw_story),
        max_tokens=2500,
    ).choices[0].text
    return ret

def create_story(theme, num_prompts, temperature=0.6, llm=None):
    if llm is not None:
        llama_postprompt = get_llama_postprompt("story")
        ret = llm(get_story_prompt(theme, num_prompts)+llama_postprompt, max_tokens=2500, stop=["Q:"], echo=True)["choices"][0]["text"]
        ret = get_substring_after(ret, llama_postprompt)
    else:
        ret = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        temperature=temperature,
        prompt=get_story_prompt(theme, num_prompts),
        max_tokens=2500,
    ).choices[0].text
    return ret

def create_music_recommendation(raw_story, llm=None, gen_music = False):
    postprompt = "" if llm is None else get_llama_postprompt("music recommendation")
    prompt = get_music_desc(raw_story) if gen_music else get_music_prompt(raw_story)
    full_prompt = prompt + postprompt
    if llm is not None:
        ret = llm(full_prompt, max_tokens=50, stop=["Q:"], echo=True)["choices"][0]["text"]
        ret = get_substring_after(ret, postprompt)
    else:
        ret = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        temperature=0.6,
        prompt=full_prompt,
        max_tokens=100,
    ).choices[0].text
    return ret

# Youtube AutoMusic

def youtube2mp3(query):
    if os.path.exists("soundtrack.mp3"):
        os.remove("soundtrack.mp3")

    i = 0
    search = Search(query).results

    if len(search) == 0:
        raise RuntimeError("Song could not be found")
    # Try to download videos in order of search
    while True:
        yt = search[i]
        try:
            video = yt.streams.filter(abr='160kbps').last()
            out_file = video.download(output_path="./")
            break
        except:
            i += 1
            if i == len(query):
                break

    new_file = Path(f'soundtrack.mp3')
    os.rename(out_file, new_file)

    ##@ Check success of download
    if new_file.exists():
        print(f'{yt.title} has been successfully downloaded.')
    else:
        print(f'ERROR: {yt.title}could not be downloaded!')

def get_substring_after(input_str, delimiter):
    """
    Return the part of the input_str that comes after the delimiter, including newlines.

    Parameters:
    input_str (str): The string from which to extract the substring.
    delimiter (str): The substring after which to start extracting.

    Returns:
    str: The extracted substring.
    """
    # Escape the delimiter to ensure any special characters are treated literally
    escaped_delimiter = re.escape(delimiter)

    # Construct the regex pattern, ensuring to capture everything (including newlines) after the delimiter
    # The (?s) flag is used to make the '.' special character match any character whatsoever, including a newline
    pattern = f"(?s){escaped_delimiter}(.*)"

    # Search for the pattern in the input string
    match = re.search(pattern, input_str)

    # If a match is found, return the captured group; otherwise, return an empty string
    return match.group(1) if match else ""