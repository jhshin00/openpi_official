from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="openvla/modified_libero_rlds",
    repo_type="dataset",
    local_dir="/ssd1/openpi_official/datasets/libero_raw",
    resume_download=True,
)
