from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy

#test_prompt = "4/4 100bpm 320kbps 48khz, Industrial/Electronic Soundtrack, Dark, Intense, Sci-Fi"
#test_prompt = "a techno song with rolling bass bpm:130"
#test_prompt = "4/4 100bpm 320kbps 48khz a techno song with rolling bass"

def gen_music(prompt, model_size="large", serialize=False, max_new_tokens=1500):
    # Max tokens is set to 1500 as it corresponds to a 30sec audio, the maximum possible time
    processor = AutoProcessor.from_pretrained(f"facebook/musicgen-{model_size}")
    model = MusicgenForConditionalGeneration.from_pretrained(f"facebook/musicgen-{model_size}").to('cuda')
    inputs = processor(text=prompt,padding=True,return_tensors="pt").to('cuda')
    audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
    if serialize:
        sampling_rate = model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())
    return audio_values