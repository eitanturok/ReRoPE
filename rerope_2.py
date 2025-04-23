import os

from diffusers import DiffusionPipeline
import torch

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = DiffusionPipeline.from_pretrained("segmind/small-sd", torch_dtype=torch.float16).to(device)
prompt = "Portrait of a pretty girl"
image = pipeline(prompt).images[0]
image.save("my_image.png")
