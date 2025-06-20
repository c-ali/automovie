from util import *
from presets import presets
from prompts import *
import numpy as np
# latent bleeding imports
from tqdm import tqdm
import ffmpeg
import torch

torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)
import warnings

# this is needed as latentblending is in a subfolder
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'latentblending'))

warnings.filterwarnings('ignore')
from gen_music import gen_music
from diffusers import AutoPipelineForText2Image
from latentblending.blending_engine import BlendingEngine
from movie_util import concatenate_movies
from huggingface_hub import hf_hub_download
# chatgpt / llm imports
from llama_cpp import Llama
import gc
import argparse


parser = argparse.ArgumentParser(
    prog='AutoMovie',
    description='Generates Fully Automated AI Movies')

parser.add_argument('-t', '--t_trans', type=int,
                    help='Duration of a single transition', default=10) # 2 seconds for fast, 6 seconds for slow
parser.add_argument('-p', '--num_prompts', type=int,
                    help='Total number of prompts for SD', default=30)
parser.add_argument('--no_captions', action='store_false',
                    help='Do not add captions to the movie')
parser.add_argument('-l', '--local_llm', action='store_true',
                    help='Run a local llama model')
parser.add_argument('-m', '--ai_music', action='store_true',
                    help='Create music with MusicGen')
parser.add_argument('-w', '--watermark', action='store_true',
                    help='Add a Watermark')
parser.add_argument('--high_res', action='store_true',
                    help='Run 4x Upscaler')
parser.add_argument('--temp', type=float,
                    help='Temperature for the language model', default=1.2)  # 0.8)
parser.add_argument('--tmax', type=int,
                    help='Time resolution of the morph.', default=12)
parser.add_argument('--dstrength', type=float,
                    help='Depth strength of the model', default=0.75)   # Specifies how deep (in terms of diffusion iterations the first branching happens). 0.75 for alpha blendy, 0.55 for buttersmooth
parser.add_argument('--preset', type=str,
                    help='Run a pre-configured mode e.g. psytrance', default="")
parser.add_argument('--debug', type=bool,
                    help='Use a dummy story.', default=False)
parser.add_argument('--turbo', type=bool,
                    help='Use a sdxl turbo.', default=False)
parser.add_argument('--res', type=int, nargs=2,
                    help='Output resolution.', default=(1024, 1024))
parser.add_argument('--steps', type=int,
                    help='Steps used in the diffusion process.', default=20)
parser.add_argument('--gscale', type=float,
                    help='Guidance scale used in the diffusion process.', default=4)
args = parser.parse_args()

# update other args
debug = args.debug
resolution = args.res
upscale = args.high_res
add_captions = args.no_captions
duration_single_trans = args.t_trans
depth_strength = args.dstrength
temperature = args.temp
num_prompts = args.num_prompts
ai_music = args.ai_music
run_local = args.local_llm
add_watermark = args.watermark
t_compute_max_allowed = None if args.turbo else args.tmax  # Determines number of intermediary steps in latent blending. 12 for high quality, 8 or lower for faster runtime

# use presets

if args.preset != "":
    assert args.preset in presets.keys()
    globals().update(presets[args.preset])

# StableDiffusion / Latentbleeding Settings
custom_model_path = "/home/chris/workspace/sd_ckpts/juggernaut_sdxl/"
#fp_config = None
fps = 24
g_scale = args.gscale
num_steps = args.steps
high_res = False
out_name = "out.mp4"
watermark_path = "./techno3_alpha.png"

# %% Define vars for high-resolution pass
fp_ckpt_hires = hf_hub_download(repo_id="stabilityai/stable-diffusion-x4-upscaler", filename="x4-upscaler-ema.ckpt")
depth_strength_hires = 0.65
num_inference_steps_hires = 100
nmb_branches_final_hires = 5

# LLM Settings
max_tries = 3

# Local LLM
n_ctx = 2048  # Context window lenth
n_gpu_layers = 35  # Number of layers to push to the GPU.
local_llama_path = "./llama-2-70b-chat.Q3_K_M.gguf"

# Debug options
print(f"DSTRENGTH {depth_strength}")
# Init local LLM model when needed
llm = None
if run_local:
    llm = Llama(model_path=local_llama_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)

if debug:
    theme = "debug theme"
    raw_story = debug_story
    split_story = remove_prefixes_and_split(debug_story)
    split_prompts = remove_prefixes_and_split(debug_prompts)
    music_inject = prompt_inject = neg_prompt_inject = ""
else:
    if "theme" not in globals():
        theme = input("Please input the theme of the movie \n")
    if "prompt_inject" not in globals():
        prompt_inject = input("Input additional instructions for the prompts \n")
    if "neg_prompt_inject" not in globals():
        neg_prompt_inject = input("Input additional instructions for the negative image prompt\n")
    if "music_inject" not in globals():
        music_inject = input("Manually set the music to this song \n")
    print("Generating Story and Prompts...")
    # Create a story matching the prompts also using GPT
    for i in range(max_tries):
        # ChatGPT cannot output more than 30 prompts at a time. For now we just create separated stories.
        if num_prompts > 30:
            num_generations, rest = divmod(num_prompts, 30)
            raw_stories = []
            split_story = []
            for j in range(num_generations):
                raw_story = create_story(theme, 30, temperature=temperature, llm=llm)
                raw_stories.append(raw_story)
                split_story = split_story + remove_prefixes_and_split(raw_story)
            raw_story = create_story(theme, rest, temperature=temperature, llm=llm)
            raw_stories.append(raw_story)
            split_story = split_story + remove_prefixes_and_split(raw_story)

        else:
            raw_story = create_story(theme, num_prompts, temperature=temperature, llm=llm)
            split_story = remove_prefixes_and_split(raw_story)

        # Cut if too long or try again if too short
        if len(split_story) == num_prompts:
            break
        if len(split_story) > num_prompts:
            split_story = split_story[:num_prompts]
            break

    print(f"Theme: {theme}. Story:")
    for i, item in enumerate(split_story):
        print(f"{i + 1}. {item}.")

    # Create Prompts with GPT 3.5
    for i in range(max_tries):
        # If we have more than 30 prompts, we split up the generation
        # TODO: Make the new story depend on the old one. Maybe use ChatGPT instead of InstructGPT?
        if num_prompts > 30:
            split_prompts = []
            for story in raw_stories:
                raw_prompts = create_prompts(story, temperature=temperature, llm=llm)
                split_prompts = split_prompts + remove_prefixes_and_split(raw_prompts)

        else:
            raw_prompts = create_prompts(raw_story, temperature=temperature, llm=llm)
            split_prompts = remove_prefixes_and_split(raw_prompts)

        # Cut if too long or try again if too short
        if len(split_prompts) == num_prompts:
            break
        if len(split_prompts) > num_prompts:
            split_prompts = split_prompts[:num_prompts]
            break
        if len(split_prompts) < num_prompts:
            num_prompts = len(split_prompts)
            split_story = split_story[:num_prompts]
            break

    # Use prompt inject
    if prompt_inject != "":
        for j in range(len(split_prompts)):
            split_prompts[j] = prompt_inject + " " + split_prompts[j]

    # Print promtps
    print(f"Prompts:")
    for i, item in enumerate(split_prompts):
        print(f"{i + 1}. {item}.")

# Create song recommendation with LLM or create song description for musicGEN
if music_inject == "":
    raw_song_rec = create_music_recommendation(raw_story, llm=llm, gen_music=ai_music)
    song_rec = remove_prefixes_and_split(raw_song_rec)
    if len(song_rec) > 1:
        song_rec = song_rec[0]
else:
    song_rec = music_inject
print(f"Recommended song: {song_rec}")

# Clean up local LLM garbage to free up VRAM
if run_local:
    del llm
    gc.collect()
print("Generating movie...")

# Join the first two captions
if len(split_story) > 1:
    split_story = [split_story[0] + " " + split_story[1]] + split_story[2:]

if args.turbo:
    pretrained_model_name_or_path = "stabilityai/sdxl-turbo"
else:
    pretrained_model_name_or_path = custom_model_path#"stabilityai/stable-diffusion-xl-base-1.0"

pipe = AutoPipelineForText2Image.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16, variant="fp16")
pipe.to('cuda')
be = BlendingEngine(pipe, do_compile=True)


# Add default negative prompt
be.set_negative_prompt(get_negative_prompt(neg_prompt_inject))
be.guidance_scale = g_scale
be.num_inference_steps = num_steps
be.set_branching(depth_strength=depth_strength, t_compute_max_allowed=t_compute_max_allowed, nmb_max_branches=None)
be.set_dimensions(resolution)
seeds = np.random.randint(0, 954375479, num_prompts).tolist() # Use random seeds because why not?
parts = []
# Create transitions
for i in tqdm(range(len(split_prompts) - 1), desc="LowRes Progress"):
    blockPrint()
    # For a multi transition we can save some computation time and recycle the latents
    if i == 0:
        be.set_prompt1(split_prompts[i])
        be.set_prompt2(split_prompts[i + 1])
        recycle_img1 = False
    else:
        be.swap_forward()
        be.set_prompt2(split_prompts[i + 1])
        recycle_img1 = True

    fp_movie_part = f"tmp_part_{str(i).zfill(3)}.mp4"
    part_nr = f"tmp_part_{str(i).zfill(3)}"

    # Run latent blending and block console output from there
    be.run_transition(
        recycle_img1=recycle_img1,
        fixed_seeds=seeds[i:i + 2]
    )
    # Apply captions & save movie
    if add_captions:
        apply_caption(be, split_story[i])
    if add_watermark:
        apply_watermark(be, watermark_path=watermark_path)
    be.write_movie_transition(fp_movie_part, duration_single_trans * 2 if i == 0 else duration_single_trans, fps=fps)
    if upscale:
        be.write_imgs_transition(part_nr)
    parts.append(part_nr)
    enablePrint()

if upscale:
    pass
    #TODO: UPDATE TO NEW VERSION
    #lb = LatentBlending(sdh)
    #for dp_part in tqdm(parts, desc="HighRes Progress"):
    #    lb.run_upscaling(dp_part, depth_strength_hires, num_inference_steps_hires, nmb_branches_final_hires)

# Finally, concatente the result
if upscale:
    list_movie_parts = [os.path.join(p, "movie_highres.mp4") for p in parts]
else:
    list_movie_parts = [f"{p}.mp4" for p in parts]
concatenate_movies(out_name, list_movie_parts)

# Free up space, run gc on image models
del be

# Add sound (automusic from Youtube) / MusicGen
if ai_music:
    print("AI Generating Music...")
    gen_music(song_rec, serialize=True)
    input_audio = ffmpeg.input("musicgen_out.wav")
else:
    youtube2mp3(song_rec)
    input_audio = ffmpeg.input('soundtrack.mp3')
    print("Adding Music...")
input_video = ffmpeg.input('out.mp4')
if os.path.exists("final_movie.mp4"):
    os.remove("final_movie.mp4")
(
    ffmpeg.output(input_video, input_audio, 'final_movie.mp4', shortest=None, vcodec='copy')
    .run()
)

write_log(theme, prompt_inject, neg_prompt_inject, depth_strength, duration_single_trans)
