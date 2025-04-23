import os
import stable_diffusuvion_cpp as sdcpp

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

repo_id = "city96/FLUX.1-schnell-gguf"
filename = "flux1-schnell-Q2_K.gguf"
name = "flux1-schnell"

image_model = sdcpp.StableDiffusionCpp(repo_id="city96/FLUX.1-schnell-gguf", filename="flux1-schnell-Q2_K.gguf", n_threads=20)
prompt = "a cute cat"
image = image_model.generate(prompt=prompt, seed=42)
