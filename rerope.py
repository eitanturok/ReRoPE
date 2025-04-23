import os
from typing import List
from diffusers.models.embeddings import get_1d_rotary_pos_embed

import torch
from torch import nn
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoRotaryPosEmbed

# Faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to the GGUF checkpoint (can be a Hugging Face Hub URL or local file)
# from https://huggingface.co/collections/city96/gguf-image-model-quants-67199ef97bf1d9ca49033234
ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf"

# Load the transformer model from the GGUF file
print('loading transformer...')
transformer = FluxTransformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)

# Load the pipeline with the pre-trained model, replacing the transformer
print('loading pipe...')
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.to(device)
original_rope = pipe.transformer.pos_embed

# memory savings
pipe.enable_model_cpu_offload()

# inference
print('doing inference...')
prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, generator=torch.manual_seed(0), device=device).images[0]
image.save("flux-gguf.png")
print('finished.')
