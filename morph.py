from util import *
from prompts import *
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
# chatgpt imports
import openai

# Settings
fps = 24
num_prompts = 15
duration_single_trans = 10
depth_strength = 0.55  # Specifies how deep (in terms of diffusion iterations the first branching happens)
high_res = False
debug = False
temperature = 0.6
max_tries = 3

openai.api_key = open("openai_apikey", "r").read()

if not debug:
    theme = input("Please input the theme of the movie \n")

    # Create a story matching the prompts also using GPT
    correctly_generated = False

    for i in range(max_tries):
        raw_story = create_story(theme, num_prompts, temperature=temperature)
        split_story = remove_prefixes(raw_story)
        if len(split_story) == num_prompts:
            break
    print(f"Theme: {theme}. Story: {raw_story}.")

    # Create Prompts with GPT 3.5
    for i in range(max_tries):
        raw_prompts = create_prompts(raw_story, temperature=temperature)
        split_prompts = remove_prefixes(raw_prompts)
        if len(split_prompts) == num_prompts:
            break

    print(f"Prompts: {raw_prompts}.")
    print("Generating movie...")

else:
    raw_prompts = '''1.Two women sitting in a bar on stools and chatting
2. Two women sitting close together in a bar, flirting and laughing
3. Two women walk together in the streets, hand in hand
4. Two women laying in a bed, still dressed, sensually kissing
5. Two women laying in a bed, slowly undressing each other
6. Two naked women laying naked in a bed, kissing each other sensually
7. Two women naked in a bed, eating each other out sensually
8. night scene, full body shot of a sexy naked nude girl, posing, look at a camera, (smile:0.7), [scarlett johansson:emma watson:0.3], white blue hair, ponytail, cute young face, 18 yo, soft volumetric lights, (backlit:1.3), (cinematic:1.3), intricate details, (ArtStation:1.3), Rutkowski
9. Vampire naked Queen, wet skin, backlit, intricate details, highly detailed, slate atmosphere, cinematic, dimmed colors, dark shot, muted colors, film grainy, lut, spooky
10. night scene, close up photo of a sexy naked girl, posing, look at a camera and smile, pink ponytail hair, (green eyes:0.8), cute young face, 18 yo, soft volumetric lights, (backlit:1.3), (cinematic:1.3), intricate details, (ArtStation:1.2)'''
    raw_story = '''1. Two women, Sarah and Emily, walked into a dimly lit bar, their eyes scanning the room for a place to sit.
2. After a few minutes, they spotted a lone bar stool and sat down, ordering drinks and striking up a conversation. As they talked, they realized they had a lot in common and their laughter filled the bar.
3. Sarah and Emily leave and get to the hotel where Emily resides
4. Sarah and Emily start to become sensual, they touch each other and kiss
5. They slowly start to take their clothes off and feel each other bodies
6. They lay naked on the bed and kiss very sensually.
7. Emily starts to eat out Sarah
8. test
9. test
10. test'''



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
sdh.guidance_scale = 7
sdh.num_inference_steps = 25

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
    apply_caption(lb, split_story[i], high_res=high_res)
    lb.write_movie_transition(fp_movie_part, duration_single_trans*2 if i == 0 else duration_single_trans, fps=fps)
    parts.append(part_nr)


# Finally, concatente the result
list_movie_parts = [f"{p}.mp4" for p in parts]
concatenate_movies(out_name, list_movie_parts)

# Add sound (automusic from Youtube)
recommended_song = create_music_recommendation(raw_story)
print(f"Recommended song: {recommended_song}")
youtube2mp3(recommended_song)
input_video = ffmpeg.input('out.mp4')
input_audio = ffmpeg.input('soundtrack.mp3')
if os.path.exists("final_movie.mp4"):
    os.remove("final_movie.mp4")
(
    ffmpeg.output(input_video, input_audio, 'final_movie.mp4',  shortest=None, vcodec='copy')
    .run()
)
