from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen-Image-Edit",
    repo_type="model",                       # optional; defaults to "model"
    local_dir="models/Qwen-Image-Edit",      # directory where files are stored
    local_dir_use_symlinks=False             # ensure no symlinks, replicate files
)
