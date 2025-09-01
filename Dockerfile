# 1. Base Image
# Use the official Runpod PyTorch image as specified. This provides the CUDA, cuDNN, and Python environment.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# 2. System Dependencies
# We need to install 'git' because one of the Python packages (diffusers) is installed directly from a GitHub repository.
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# 3. Set Working Directory
# Set a working directory inside the container to keep the project files organized.
WORKDIR /app

# 4. Copy Application Files
# Copy your handler and prompt files from your local machine into the container's working directory.
# This assumes the Dockerfile is in the same directory as these files.
COPY handler.py .
COPY model_prompts.txt .
COPY texture_prompts.txt .

# 5. Install Python Dependencies
# Install all the required Python packages in a single layer to keep the image size smaller.
# The `--no-cache-dir` flag prevents pip from storing the wheel cache, reducing the final image size.
RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/diffusers \
    transformers \
    accelerate \
    peft \
    runpod

# 6. Set the Default Command
# This command is executed when the container starts.
# It runs your handler script, which will initialize the model and start listening for jobs from Runpod.
# The "-u" flag is for unbuffered Python output, which is best for logging in a containerized environment.
CMD ["python", "-u", "handler.py"]
