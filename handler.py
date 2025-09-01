import os
import json
import torch
import runpod
import base64
from io import BytesIO
from PIL import Image, ImageOps
import math
import re

# --- Diffusers Imports ---
from diffusers import (
    QwenImageEditPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models import QwenImageTransformer2DModel

# --- Global Variables & Model Loading ---
pipe = None
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# --- Prompt Loading ---
# Load prompts from local files into memory once during cold start
OBJECT_PROMPTS = {}
TEXTURE_PROMPT = ""

try:
    with open("model_prompts.txt", "r") as f:
        prompt_data = json.load(f)
        # Organize prompts for easy access
        for stage_data in prompt_data.get("pipeline", []):
            stage_name = stage_data.get("name")
            if stage_name:
                OBJECT_PROMPTS[stage_name] = stage_data["prompt"]
    print("Successfully loaded object pipeline prompts.")
except Exception as e:
    print(f"Warning: Could not load model_prompts.txt. Object pipeline will fail. Error: {e}")

try:
    with open("texture_prompts.txt", "r") as f:
        TEXTURE_PROMPT = f.read()
    print("Successfully loaded texture pipeline prompt.")
except Exception as e:
    print(f"Warning: Could not load texture_prompts.txt. Texture pipeline will fail. Error: {e}")


def load_model():
    """
    Loads the model and pipeline into memory. This function is called only once
    when the worker cold starts.
    """
    global pipe
    
    model_name = "/workspace/models/Qwen-Image-Edit"
    lora_path = "/workspace/models/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors"

    print("Starting model load...")
    
    model = QwenImageTransformer2DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=torch_dtype
    )
    
    scheduler_config = {
        "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False,
        "max_image_seq_len": 8192, "max_shift": math.log(3), "num_train_timesteps": 1000,
        "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False,
        "time_shift_type": "exponential", "use_beta_sigmas": False, "use_dynamic_shifting": True,
        "use_exponential_sigmas": False, "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    pipe = QwenImageEditPipeline.from_pretrained(
        model_name, transformer=model, scheduler=scheduler, torch_dtype=torch_dtype
    )
    
    # FIX: Move pipeline to GPU before loading LoRA. Your implementation of this
    # was already correct. Moving the pipe to the GPU first is the right way to
    # speed up LoRA weight patching. The initial 10-minute load time is likely a
    # one-time cost on cold start as the model and LoRA weights are patched in memory.
    # Subsequent "warm" inferences will be much faster.
    pipe.to(device)
    
    print("Loading LoRA weights...")
    pipe.load_lora_weights(lora_path)
    
    print("Model loaded successfully.")
    return pipe

def run_inference(prompt, image, seed):
    """Helper function to run a single generation step."""
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Ensure the input image is a PIL image, not a list
    if isinstance(image, list):
        if len(image) > 0:
            image = image[0] # Take the first image if a list is passed
        else:
            raise ValueError("run_inference received an empty list for the image argument.")

    input_args = {
        "image": image,
        "prompt": json.dumps(prompt), # Prompts are JSON objects
        "generator": generator,
        "true_cfg_scale": 1.0,
        "negative_prompt": " ",
        "num_inference_steps": 8,
    }
    print(f"Running inference for: {prompt.get('summary', 'N/A')}")
    # The pipeline returns a list of images, we want the first one.
    output_images = pipe(**input_args).images[0] # FIX: The pipeline returns a list, so we get the first image
    print("Inference step complete.")
    return output_images

def fill_prompt_placeholders(prompt_template, vlm_data):
    """Replaces {{placeholders}} in a prompt string with data from the VLM JSON."""
    prompt_str = json.dumps(prompt_template)
    
    # Find all placeholders like {{field_name}}
    placeholders = re.findall(r"\{\{(\w+)\}\}", prompt_str)
    
    for placeholder in placeholders:
        if placeholder in vlm_data:
            value = vlm_data[placeholder]
            # Ensure the replacement value is a JSON-compatible string
            replacement = json.dumps(value).strip('"')
            prompt_str = prompt_str.replace(f"{{{{{placeholder}}}}}", replacement)
            
    return json.loads(prompt_str)

def threshold_and_apply_alpha(rgb_image, mask_image, threshold=10):
    """Converts a mask to B&W, thresholds it, and applies it as an alpha channel."""
    print("Applying threshold and alpha channel...")
    # Convert mask to grayscale and apply threshold
    mask_l = mask_image.convert("L")
    alpha_mask = mask_l.point(lambda p: 255 if p > threshold else 0)

    # Ensure the main image is in RGBA format
    rgba_image = rgb_image.convert("RGBA")
    
    # Apply the thresholded mask as the alpha channel
    rgba_image.putalpha(alpha_mask)
    print("Alpha channel applied.")
    return rgba_image

def handler(job):
    """Main function that RunPod serverless calls for each job."""
    global pipe
    
    if pipe is None:
        pipe = load_model()
        
    job_input = job['input']

    # --- Parse Inputs ---
    model_gen = job_input.get('model_gen', True)
    vlm_output = job_input.get('vlm_output', {})
    base64_images = job_input.get('images', []) # Expect a list
    seed = job_input.get('seed', 42)

    if not base64_images:
        return {"error": "Input 'images' list cannot be empty."}

    try:
        # FIX: Decode the FIRST image from the list. b64decode expects a single
        # string, not a list of strings.
        initial_image = Image.open(BytesIO(base64.b64decode(base64_images[0]))).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to decode base64 image: {e}"}

    # --- Pipeline Router ---
    if model_gen:
        # --- OBJECT PIPELINE ---
        print("--- Starting Object Pipeline ---")
        
        # STAGE 1: Isolate and Reorient
        stage1_prompt = fill_prompt_placeholders(OBJECT_PROMPTS['isolate_and_reorient'], vlm_output)
        stage1_image = run_inference(stage1_prompt, initial_image, seed)
        
        images_to_process = [(stage1_image, False)] # (image, has_skipped_stage2)

        # STAGE 1.5: Interior Cavity (Conditional)
        if vlm_output.get('is_transparent_container', False):
            print("Condition met: is_transparent_container is true. Running Stage 1.5.")
            stage1_5_prompt = fill_prompt_placeholders(OBJECT_PROMPTS['interior_cavity_isolation'], vlm_output)
            stage1_5_image = run_inference(stage1_5_prompt, stage1_image, seed)
            images_to_process.append((stage1_5_image, True)) # This path skips stage 2
        
        final_outputs = []
        for current_image, skip_stage_2 in images_to_process:
            
            stage2_image = current_image
            if not skip_stage_2:
                # STAGE 2: Material Tagging (Conditional Chain)
                print("--- Starting Stage 2 ---")
                stage2_prompts_config = {
                    'has_mirror': 'material_tag_replacement_mirror_only',
                    'has_countertop': 'material_tag_replacement_countertop_only',
                    'has_glass': 'material_tag_replacement_glass_only',
                    'has_clear_plastic': 'material_tag_replacement_clear_plastic_only'
                }
                
                for key, prompt_name in stage2_prompts_config.items():
                    if vlm_output.get(key, False):
                        print(f"Condition met: {key} is true. Running {prompt_name}.")
                        prompt = fill_prompt_placeholders(OBJECT_PROMPTS[prompt_name], vlm_output)
                        stage2_image = run_inference(prompt, stage2_image, seed)
            else:
                print("Skipping Stage 2 for transparent container interior.")

            # STAGE 3: Clean and Soften
            print("--- Starting Stage 3 ---")
            stage3_prompt = OBJECT_PROMPTS['clean_and_soften_diffuse'] # No placeholders
            stage3_image = run_inference(stage3_prompt, stage2_image, seed)

            # STAGE 4: Alpha Mask Generation
            print("--- Starting Stage 4 ---")
            stage4_prompt = OBJECT_PROMPTS['alpha_mask_generation'] # No placeholders
            stage4_mask = run_inference(stage4_prompt, stage3_image, seed)
            
            # Final Compositing
            final_image = threshold_and_apply_alpha(stage3_image, stage4_mask)
            final_outputs.append(final_image)

        # Encode all final images for output
        output_images_b64 = []
        for img in final_outputs:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            output_images_b64.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
            
        return {"images": output_images_b64}

    else:
        # --- TEXTURE PIPELINE ---
        print("--- Starting Texture Pipeline ---")
        prompt_str = TEXTURE_PROMPT.replace('{{VLM Output}}', json.dumps(vlm_output))
        
        # The texture prompt is a string, but our inference function expects a JSON object
        # We wrap it in a simple structure.
        texture_prompt_obj = {"summary": "Generate a seamless texture.", "instructions": prompt_str}
        
        output_image = run_inference(texture_prompt_obj, initial_image, seed)
        
        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"images": [img_str]}


# ---------------------------------------------------------------------------- #
#                                Local Testing                                 #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("--- Starting local test ---")

    # --- Configuration for Local Test ---
    test_image_paths = ["Test_of_Qwen.png"] # IMPORTANT: Change this to your test image(s)
    
    # Sample VLM output for testing the object pipeline
    sample_vlm_object_data = {
        "object_description": "A gray plastic bin with hexagonal holes",
        "is_transparent_container": False,
        "has_mirror": False,
        "has_countertop": False,
        "has_glass": False,
        "has_clear_plastic": False
    }

    # --- Helper to encode images ---
    def encode_images_to_base64(paths):
        encoded = []
        for path in paths:
            try:
                with open(path, "rb") as image_file:
                    encoded.append(base64.b64encode(image_file.read()).decode("utf-8"))
                print(f"Successfully encoded image from: {path}")
            except FileNotFoundError:
                print(f"ERROR: Test image not found at '{path}'.")
                return None
        return encoded

    encoded_images = encode_images_to_base64(test_image_paths)

    if encoded_images:
        # --- Test 1: Object Pipeline ---
        print("\n--- Testing Object Pipeline (model_gen=True) ---")
        object_job = {
            "input": {
                "model_gen": True,
                "vlm_output": sample_vlm_object_data,
                "images": encoded_images,
                "seed": 42
            }
        }
        object_result = handler(object_job)
        if "error" in object_result:
            print(f"Object pipeline test failed: {object_result['error']}")
        else:
            for i, img_b64 in enumerate(object_result.get("images", [])):
                output_filename = f"test_output_object_{i}.png"
                with open(output_filename, "wb") as f:
                    f.write(base64.b64decode(img_b64))
                print(f"Object pipeline output saved to {output_filename}")

        # --- Test 2: Texture Pipeline ---
        print("\n--- Testing Texture Pipeline (model_gen=False) ---")
        texture_job = {
            "input": {
                "model_gen": False,
                "vlm_output": {"material": "marble", "color": "white with gray veins"},
                "images": encoded_images,
                "seed": 42
            }
        }
        texture_result = handler(texture_job)
        if "error" in texture_result:
            print(f"Texture pipeline test failed: {texture_result['error']}")
        else:
            # FIX: Handle the list of images returned by the handler.
            # We expect one image, so we take the first element.
            img_b64_list = texture_result.get("images", [])
            if img_b64_list:
                output_filename = "test_output_texture.png"
                with open(output_filename, "wb") as f:
                    # FIX: Decode the first element of the list, not the list itself.
                    f.write(base64.b64decode(img_b64_list[0]))
                print(f"Texture pipeline output saved to {output_filename}")
            else:
                print("Texture pipeline test did not return any images.")