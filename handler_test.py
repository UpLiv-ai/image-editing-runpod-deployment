import os
import sys
import torch
import runpod
import base64
from io import BytesIO
from PIL import Image
import math

# --- Add Qwen-Image source to Python path to import the prompt polisher ---
# This assumes handler.py is in the same directory as the 'Qwen-Image' folder.
try:
    qwen_image_src_path = os.path.join(os.path.dirname(__file__), 'Qwen-Image', 'src')
    if qwen_image_src_path not in sys.path:
        sys.path.append(qwen_image_src_path)
    from examples.edit_demo import polish_edit_prompt
    print("Successfully imported official polish_edit_prompt function.")
except (ImportError, ModuleNotFoundError):
    print("Warning: Could not import 'polish_edit_prompt'. Falling back to simple prompt enhancement.")
    polish_edit_prompt = None

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
    pipe.to(device)
    
    pipe.load_lora_weights(lora_path)
    
    
    print("Model loaded successfully.")
    return pipe

def handler(job):
    """
    Main function that RunPod serverless calls for each job.
    """
    global pipe
    
    if pipe is None:
        pipe = load_model()
        
    job_input = job['input']

    prompt = job_input.get('prompt', "Make this image cinematic.")
    base64_image = job_input.get('image')
    enhance_prompt = job_input.get('enhance_prompt', True)

    if not base64_image:
        return {"error": "No image provided in the input."}
        
    try:
        input_image = Image.open(BytesIO(base64.b64decode(base64_image))).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to decode base64 image: {e}"}

    # --- Prompt Enhancement ---
    if enhance_prompt and polish_edit_prompt is not None:
        # Use the official, imported prompt polisher function
        final_prompt = polish_edit_prompt(prompt, input_image)
    else:
        # Fallback if the function couldn't be imported or if enhancement is disabled
        if enhance_prompt:
             final_prompt = prompt + ", while preserving the exact facial features, expression, clothing, and pose. Maintain the same background, natural lighting, and overall photographic composition and style."
        else:
            final_prompt = prompt

    generator = torch.Generator(device=device).manual_seed(job_input.get('seed', 42))

    input_args = {
        "image": input_image,
        "prompt": final_prompt,
        "generator": generator,
        "true_cfg_scale": job_input.get('cfg', 1.0),
        "negative_prompt": " ",
        "num_inference_steps": job_input.get('steps', 8),
    }

    print(f"Original prompt: {prompt}")
    print(f"Executing with final prompt: {final_prompt}")
    output_image = pipe(**input_args).images
    print("Inference complete.")

    buffered = BytesIO()
    output_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"image": img_str}


# ---------------------------------------------------------------------------- #
#                                Local Testing                                 #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("--- Starting local test ---")

    # --- 1. Set the path to your local test image ---
    test_image_path = "path/to/your/test_image.png" # IMPORTANT: Change this path

    # --- 2. Encode the image to base64 ---
    try:
        with open(test_image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        print(f"Successfully encoded image from: {test_image_path}")
    except FileNotFoundError:
        print(f"ERROR: Test image not found at '{test_image_path}'. Please update the path.")
        encoded_string = None
    except Exception as e:
        print(f"An error occurred while encoding the image: {e}")
        encoded_string = None

    if encoded_string:
        # --- 3. Simulate a serverless job payload ---
        test_job = {
            "input": {
                "prompt": "Convert this into 3d claymation-esk version of the main object in the image.",
                "image": encoded_string,
                "seed": 1234,
                "steps": 8,
                "cfg": 1.0,
                "enhance_prompt": True 
            }
        }

        # --- 4. Call the handler function ---
        result = handler(test_job)

        # --- 5. Process the result ---
        if "error" in result:
            print(f"Handler returned an error: {result['error']}")
        else:
            output_image_data = base64.b64decode(result['image'])
            output_filename = "test_output_local.png"
            with open(output_filename, "wb") as f:
                f.write(output_image_data)
            print(f"Test complete. Image saved to {output_filename}")

    # To run the server locally for API testing (e.g., with Postman):
    # print("\n--- Starting server for API testing ---")
    # runpod.serverless.start({"handler": handler})