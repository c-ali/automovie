# caption imports
import numpy as np
import cv2
import os
# latent bleeding imports
import torch
import ffmpeg

torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)
import warnings

warnings.filterwarnings('ignore')
from latent_blending import LatentBlending
from stable_diffusion_holder import StableDiffusionHolder
from movie_util import concatenate_movies
from huggingface_hub import hf_hub_download
# chatgpt imports
import re
import openai

# Settings
fps = 24
num_prompts = 2
duration_single_trans = 7
depth_strength = 0.55  # Specifies how deep (in terms of diffusion iterations the first branching happens)


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
def add_caption(img, caption, add_linebreaks=False):
    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (15, 500)
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
        broken_caption = split_after_n_chars(caption)
        for i, text_line in enumerate(broken_caption):
            y = bottomLeftCornerOfText[1] + (i -len(broken_caption)) * dy
            cv2.putText(img, text_line,
                        (bottomLeftCornerOfText[0], y),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

theme = input("Please input the theme of the movie \n")
openai.api_key = open("openai_apikey", "r").read()

# Create Prompts with GPT 3.5
prompts_req = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    temperature=0.6,
    prompt=f"Create a list of {num_prompts} descriptive prompts for an image generation AI on the theme of {theme}."
           f" The prompts should tell a time coherent story where each image is related to the last and the next."
           f" Separate the prompts by a newline",
    max_tokens=2000,
)

# Create a story matching the prompts also using GPT
raw_prompts = prompts_req.choices[0].text
split_prompts = remove_prefixes(raw_prompts)
print(f"Theme: {theme}. Prompts: {raw_prompts}.")

story_req = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    temperature=0.6,
    prompt=f"Tell a story matching the following prompts. Write two sentences for each prompt and output it as a list. {split_prompts}",
    max_tokens=2000,
)
raw_story = prompts_req.choices[0].text
split_story = remove_prefixes(raw_story)

print(f"Story: {raw_story}. Generating movie...")

# %% First let us spawn a stable diffusion holder. Uncomment your version of choice.
fp_ckpt = hf_hub_download(repo_id="stabilityai/stable-diffusion-2-1-base", filename="v2-1_512-ema-pruned.ckpt")
# fp_ckpt = hf_hub_download(repo_id="stabilityai/stable-diffusion-2-1", filename="v2-1_768-ema-pruned.ckpt")
sdh = StableDiffusionHolder(fp_ckpt)
lb = LatentBlending(sdh)

# Specify a list of prompts below
list_prompts = split_prompts

# You can optionally specify the seeds
# list_seeds = [954375479, 332539350, 956051013, 408831845, 250009012, 675588737]
t_compute_max_allowed = 12  # per segment
out_name = "out.mp4"

# list_movie_parts = []
parts = []
# Create transitions

for i in range(len(list_prompts) - 1):
    # For a multi transition we can save some computation time and recycle the latents
    if i == 0:
        lb.set_prompt1(list_prompts[i])
        lb.set_prompt2(list_prompts[i + 1])
        recycle_img1 = False
    else:
        lb.swap_forward()
        lb.set_prompt2(list_prompts[i + 1])
        recycle_img1 = True

    fp_movie_part = f"tmp_part_{str(i).zfill(3)}.mp4"
    part_nr = f"tmp_part_{str(i).zfill(3)}"
    # Run latent blending
    lb.run_transition(
        recycle_img1=recycle_img1,
        depth_strength=depth_strength,
        t_compute_max_allowed=t_compute_max_allowed
    )

    # Save movie
    lb.write_movie_transition(fp_movie_part, duration_single_trans, fps=fps)
    lb.write_imgs_transition(part_nr)
    # list_movie_parts.append(fp_movie_part)
    parts.append(part_nr)

# Add captions to the images
cap_frames_dir = "cap_frames"
os.makedirs(cap_frames_dir, exist_ok=True)
frames = 0
for part, caption in zip(parts, split_story):
    for dirpath, dnames, fnames in os.walk(os.path.join("./", part)):
        for f in fnames:
            if f.endswith(".jpg"):
                frames += 1
                img = cv2.imread(os.path.join(part, f))
                add_caption(img, caption, add_linebreaks=True)
                cv2.imwrite(os.path.join(cap_frames_dir,f"{frames}.jpg"), img)

(
    ffmpeg
    .input(cap_frames_dir+'/*.jpg', pattern_type='glob', framerate=fps)
    .output(out_name)
    .run()
)
'''
# Finally, concatente the result
list_movie_parts = [f"{p}.mp4" for p in parts]
concatenate_movies(out_name, list_movie_parts)
    '''