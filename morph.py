from util import *
# latent bleeding imports
import torch
from tqdm import tqdm

torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)
import warnings

warnings.filterwarnings('ignore')
from latent_blending import LatentBlending
from stable_diffusion_holder import StableDiffusionHolder
from movie_util import concatenate_movies
from huggingface_hub import hf_hub_download
# chatgpt imports
import openai

# Settings
fps = 24
num_prompts = 15
duration_single_trans = 7
depth_strength = 0.55  # Specifies how deep (in terms of diffusion iterations the first branching happens)
high_res = True


theme = input("Please input the theme of the movie \n")
openai.api_key = open("openai_apikey", "r").read()

# Create Prompts with GPT 3.5
prompts_req = create_prompts(theme, num_prompts)

# Create a story matching the prompts also using GPT
raw_prompts = prompts_req.choices[0].text
split_prompts = remove_prefixes(raw_prompts)
print(f"Theme: {theme}. Prompts: {raw_prompts}.")

story_req = create_story(raw_prompts)
raw_story = story_req.choices[0].text
split_story = remove_prefixes(raw_story)

print(f"Story: {raw_story}. Generating movie...")

# %% First let us spawn a stable diffusion holder. Uncomment your version of choice.
if high_res:
    fp_ckpt = hf_hub_download(repo_id="stabilityai/stable-diffusion-2-1", filename="v2-1_768-ema-pruned.ckpt")
else:
    fp_ckpt = hf_hub_download(repo_id="stabilityai/stable-diffusion-2-1-base", filename="v2-1_512-ema-pruned.ckpt")
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
        t_compute_max_allowed=t_compute_max_allowed
    )

    # Apply captions & save movie
    apply_caption(lb, split_story[i+1], high_res=high_res)
    lb.write_movie_transition(fp_movie_part, duration_single_trans, fps=fps)
    parts.append(part_nr)


# Finally, concatente the result
list_movie_parts = [f"{p}.mp4" for p in parts]
concatenate_movies(out_name, list_movie_parts)
