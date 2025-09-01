#!/usr/bin/env python3
"""
qwen_edit_local_run.py

Standalone script to run Qwen-Image-Edit on a single local image file.

Requirements:
  pip install diffusers transformers huggingface_hub accelerate safetensors   # (or your preferred stack)
  The model weights should already be downloaded at workspace/models/qwen-image-edit

Usage:
  python qwen_edit_local_run.py
"""

import os
import random
from PIL import Image
import torch
torch.cuda.empty_cache()


# Try to import the Qwen pipeline and optional prompt polish helper.
# The demo repo had `from diffusers import QwenImageEditPipeline` and tools.prompt_utils.polish_edit_prompt
from diffusers import QwenImageEditPipeline
try:
    from tools.prompt_utils import polish_edit_prompt
except Exception:
    polish_edit_prompt = None  # fall back if function not present

# ----------------- Config -----------------
MODEL_PATH = "/workspace/models/qwen-image-edit"   # local model directory (same as your demo)
INPUT_IMAGE = "Test_of_Qwen.png"                  # input filename in current directory
OUTPUT_BASENAME = "Test_of_Qwen_output"           # output prefix (will append _0.png etc)
PROMPT = "Convert this into 3d claymation-esk version of the main object in the image."
NEGATIVE_PROMPT = " "   # hardcoded empty negative prompt as in your demo
OFFLOAD_FOLDER = "/mnt/offload"   # make sure this exists and is writable

# Inference hyperparams (adjust if needed)
NUM_INFERENCE_STEPS = 50
TRUE_GUIDANCE_SCALE = 4.0
NUM_IMAGES_PER_PROMPT = 1
SEED = 42
RANDOMIZE_SEED = False

# Device and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Device: {device}, dtype: {dtype}")

# ----------------- Load pipeline -----------------
print(f"Loading pipeline from: {MODEL_PATH} ... (this may take a while)")
pipe = QwenImageEditPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map="cuda",
    offload_folder=OFFLOAD_FOLDER,
    low_cpu_mem_usage=True,   # helps reduce memory during load
)
# pipe = pipe.to(device)
# Inspect placement
print("HF device map:", getattr(pipe, "hf_device_map", None))
pipe.safety_checker = None  # optional: disable safety checker if not needed or not available

# If xformers installed:
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("Enabled xFormers attention.")
except Exception as e:
    print("xFormers not enabled:", e)
    
# ----------------- Prepare inputs -----------------
if not os.path.isfile(INPUT_IMAGE):
    raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

input_img = Image.open(INPUT_IMAGE).convert("RGB")

prompt = PROMPT
if polish_edit_prompt is not None:
    try:
        # polish_edit_prompt in the original demo rewrites prompts for better stability
        prompt = polish_edit_prompt(prompt, input_img)
        print("Prompt after polish_edit_prompt:", prompt)
    except Exception as e:
        print("Warning: polish_edit_prompt failed, using original prompt. Error:", e)

if RANDOMIZE_SEED:
    seed = random.randint(0, 2**31 - 1)
else:
    seed = SEED

generator = torch.Generator(device=device).manual_seed(seed) if device.startswith("cuda") else torch.Generator().manual_seed(seed)

print("Running edit with:")
print("  prompt:", prompt)
print("  negative_prompt:", NEGATIVE_PROMPT)
print("  seed:", seed)
print("  steps:", NUM_INFERENCE_STEPS)
print("  guidance:", TRUE_GUIDANCE_SCALE)
print("  images per prompt:", NUM_IMAGES_PER_PROMPT)

# ----------------- Run pipeline -----------------
outputs = pipe(
    input_img,
    prompt=prompt,
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=NUM_INFERENCE_STEPS,
    generator=generator,
    true_cfg_scale=TRUE_GUIDANCE_SCALE,
    num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
)

images = outputs.images  # list of PIL images

# ----------------- Save outputs -----------------
for idx, img in enumerate(images):
    out_name = f"{OUTPUT_BASENAME}_{idx}.png"
    img.save(out_name)
    print(f"Saved output image: {out_name}")

print("Done.")
