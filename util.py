import re
import cv2
import openai
from pytube import Search
import os
from pathlib import Path


def remove_prefixes(text):
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



# ChatGPT functions

def create_prompts(raw_story, temperature=0.6):
    return openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        temperature=temperature,
        prompt=f"Create a list of scene descriptions matching the following story {raw_story}."
               f"The amount of items on the list should be the same and each description should match the corresponding story point."
               f"Good examples of descriptions would be: 1. full body picture of one tall blond girl, model, with space buns hair style, she wear a black short, and black patterns tights ,black lace top with intricated yellow details, black ankle boots in a rooftop in Paris, masterpiece,"
               f"2. A moustached man with deep creases and wrinkles in his face, stunning ochre eyes, wild windswept long hair, taking a selfie near a faraway castle at sunset."
               f"Close up of an adult woman as (Alice in Wonderland:1.1), dressed in tight attire, amidst an intense city landscape with a Blade Runner aesthetic, narrow street crowded with people, rain, steam, neon, dark, (small:1.5)",
        max_tokens=2000,
    ).choices[0].text

def create_story(theme, num_prompts, temperature=0.6):
    return openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        temperature=temperature,
        prompt=fr"Tell an interesting, breathtaking story on the theme of {theme}. The story should be structured as list of {num_prompts} where each item starts with a number (eg. 1. first prompt). Each item number should be one rather short sentence and the items are separated by a newline.",
        max_tokens=2000,
    ).choices[0].text

def create_music_recommendation(raw_story):
    return openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        temperature=0.6,
        prompt=fr"Give me a song recommendation for a sountrack illustrating the following story. The reply should only contain the name of the artis and the name of the song. \n {raw_story}",
        max_tokens=100,
    ).choices[0].text

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