from util import *
from prompts import *
import numpy as np
# latent bleeding imports
from tqdm import tqdm
import ffmpeg
import torch
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)
import warnings

warnings.filterwarnings('ignore')
from latent_blending import LatentBlending
from stable_diffusion_holder import StableDiffusionHolder
from movie_util import concatenate_movies
from huggingface_hub import hf_hub_download
# chatgpt / llm imports
import openai
from llama_cpp import Llama
import gc

# StableDiffusion / Latentbleeding Settings
fps = 24
duration_single_trans = 10
depth_strength = 0.55  # Specifies how deep (in terms of diffusion iterations the first branching happens)
high_res = False


# LLM Settings
openai.api_key = open("openai_apikey", "r").read()
#local_llama_path ="/home/chris/workspace/sd_ckpts/llama-2-70b-chat.Q5_K_M.gguf"
local_llama_path ="./llama-2-70b-chat.Q3_K_M.gguf"

temperature = 0.8
max_tries = 3
num_prompts = 10
# Local LLM
run_local = True
n_ctx = 2048 # Context window lenth
n_gpu_layers = 35

# Debug options
debug_visuals = False
debug_prompts = False

# Init local LLM model when needed#
llm = None
if run_local:
    llm = Llama(model_path=local_llama_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)

if debug_visuals:
    raw_prompts = debug_prompts
    raw_story = debug_story
else:
    if debug_prompts:
        theme = "A woman sitting on the toilet"
    else:
        theme = input("Please input the theme of the movie \n")
        print("Generating prompts...")
    # Create a story matching the prompts also using GPT
    for i in range(max_tries):
        raw_story = create_story(theme, num_prompts, temperature=temperature, llm=llm)
        print(f"Theme: {theme}. Story: {raw_story}.")
        split_story = remove_prefixes_and_split(raw_story)
        if len(split_story) == num_prompts:
            break
        if len(split_story) > num_prompts:
            split_story = split_story[:num_prompts]
            break

    # Create Prompts with GPT 3.5
    for i in range(max_tries):
        raw_prompts = create_prompts(raw_story, temperature=temperature, llm=llm)
        print(f"Prompts: {raw_prompts}.")
        split_prompts = remove_prefixes_and_split(raw_prompts)
        if len(split_prompts) == num_prompts:
            break
        if len(split_prompts) > num_prompts:
            split_prompts = split_prompts[:num_prompts]
            break

# Create song recommendation with LLM
raw_song_rec = create_music_recommendation(raw_story, llm=llm)
song_rec = remove_prefixes_and_split(raw_song_rec)
if len(song_rec) > 1:
    song_rec = song_rec[0]
print(f"Recommended song: {raw_song_rec}")
if run_local:
    del llm
    gc.collect()
print("Generating movie...")




# Join the first two captions
if len(split_story) > 1:
    split_story = [split_story[0] + " " + split_story[1]] + split_story[2:]

# %% First let us spawn a stable diffusion holder. Uncomment your version of choice.
if high_res:
    fp_ckpt = hf_hub_download(repo_id="stabilityai/stable-diffusion-2-1", filename="v2-1_768-ema-pruned.ckpt")
else:
    fp_ckpt = hf_hub_download(repo_id="stabilityai/stable-diffusion-2-1-base", filename="v2-1_512-ema-pruned.ckpt")

fp_ckpt = "/home/chris/workspace/sd_ckpts/photon_v1-5.st"
#fp_config = "/home/chris/workspace/sd_ckpts/artiusV21.yaml"

sdh = StableDiffusionHolder(fp_ckpt=fp_ckpt)
lb = LatentBlending(sdh)

# Add default negative prompt
lb.set_negative_prompt(neg_prompt)
sdh.guidance_scale = 6
sdh.num_inference_steps = 20
seeds = np.random.randint(0,954375479,num_prompts).tolist()

# Specify a list of prompts below
list_prompts = split_prompts

# You can optionally specify the seeds
# list_seeds = [954375479, 332539350, 956051013, 408831845, 250009012, 675588737]
t_compute_max_allowed = 12  # per segment
out_name = "out.mp4"

# list_movie_parts = []
parts = []

# Create transitions
for i in tqdm(range(len(list_prompts) - 1), desc="Total Progress"):
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
        t_compute_max_allowed=t_compute_max_allowed,
        fixed_seeds=seeds[i:i+2]
    )

    # Apply captions & save movie
    apply_caption(lb, split_story[i], high_res=high_res)
    lb.write_movie_transition(fp_movie_part, duration_single_trans * 2 if i == 0 else duration_single_trans, fps=fps)
    parts.append(part_nr)


# Finally, concatente the result
list_movie_parts = [f"{p}.mp4" for p in parts]
concatenate_movies(out_name, list_movie_parts)

# Add sound (automusic from Youtube)
print("Adding music")
youtube2mp3(raw_song_rec)
input_video = ffmpeg.input('out.mp4')
input_audio = ffmpeg.input('soundtrack.mp3')
if os.path.exists("final_movie.mp4"):
    os.remove("final_movie.mp4")
(
    ffmpeg.output(input_video, input_audio, 'final_movie.mp4',  shortest=None, vcodec='copy')
    .run()
)
