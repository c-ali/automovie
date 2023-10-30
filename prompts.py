neg_prompt2 = '''(((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, 
(extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs (mutated hands and 
fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquid tongue, 
disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, 
lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, bad hands, fused hand, 
missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fused ears, bad ears, 
poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, deformed, ugly, mutilated, disfigured, text, 
extra limbs, face cut, head cut, extra fingers, extra arms, poorly drawn face, mutation, bad proportions, 
cropped head, malformed limbs, mutated hands, fused fingers, long neck'''

neg_prompt = "cartoon, children, underage girl, painting, illustration, (worst quality, low quality, normal quality:2)"

debug_prompts = '''1.Two women sitting in a bar on stools and chatting
2. Two women sitting close together in a bar, flirting and laughing
3. Two women walk together in the streets, hand in hand
4. Two women laying in a bed, still dressed, sensually kissing
5. Two women laying in a bed, slowly undressing each other
6. Two naked women laying naked in a bed, kissing each other sensually
7. Two women naked in a bed, eating each other out sensually
8. night scene, full body shot of a sexy naked nude girl, posing, look at a camera, (smile:0.7), [scarlett johansson:emma watson:0.3], white blue hair, ponytail, cute young face, 18 yo, soft volumetric lights, (backlit:1.3), (cinematic:1.3), intricate details, (ArtStation:1.3), Rutkowski
9. Vampire naked Queen, wet skin, backlit, intricate details, highly detailed, slate atmosphere, cinematic, dimmed colors, dark shot, muted colors, film grainy, lut, spooky
10. night scene, close up photo of a sexy naked girl, posing, look at a camera and smile, pink ponytail hair, (green eyes:0.8), cute young face, 18 yo, soft volumetric lights, (backlit:1.3), (cinematic:1.3), intricate details, (ArtStation:1.2)'''
debug_story = '''1. Two women, Sarah and Emily, walked into a dimly lit bar, their eyes scanning the room for a place to sit.
2. After a few minutes, they spotted a lone bar stool and sat down, ordering drinks and striking up a conversation. As they talked, they realized they had a lot in common and their laughter filled the bar.
3. Sarah and Emily leave and get to the hotel where Emily resides
4. Sarah and Emily start to become sensual, they touch each other and kiss
5. They slowly start to take their clothes off and feel each other bodies
6. They lay naked on the bed and kiss very sensually.
7. Emily starts to eat out Sarah
8. test
9. test
10. test'''

def get_story_prompt(theme, num_prompts):
    return fr"Tell an interesting, breathtaking story on the theme of {theme}. The story should be structured as list of {num_prompts} where each item starts with a number (eg. 1. first prompt). Each item number should be one rather short sentence and the items are separated by a newline."

def get_music_prompt(raw_story):
    return fr"Give me a song recommendation for a sountrack illustrating the following story {raw_story}. The reply should be in the format '<song name> <artist> \n'."

def get_music_desc(raw_story):
    return fr"Give me a simple, (short) description of a music track that would fit the following story {raw_story} e.g. '4/4 100bpm 320kbps 48khz, Industrial/Electronic Soundtrack, Dark, Intense, Sci-Fi'"

def get_prompt_prompt(raw_story):
    return f"Create a list of descriptions matching the following story {raw_story}. They should be short, visual descriptions of images that could illustrate the story."\
               f"The amount of items on the list should be the same and each description, seaparated by a newline, should match the corresponding story point."\
               f"Good examples of descriptions would be: 1. full body picture of one tall blond girl, model, with space buns hair style, she wear a black short, and black patterns tights ,black lace top with intricated yellow details, black ankle boots in a rooftop in Paris, masterpiece,"\
               f"2. A moustached man with deep creases and wrinkles in his face, stunning ochre eyes, wild windswept long hair, taking a selfie near a faraway castle at sunset."\
               f"Close up of an adult woman as (Alice in Wonderland:1.1), dressed in tight attire, amidst an intense city landscape with a Blade Runner aesthetic, narrow street crowded with people, rain, steam, neon, dark, (small:1.5)"

def get_llama_postprompt(what="story"):
    return f"Assistant: Sure, here is your {what}:"
