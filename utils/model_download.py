from huggingface_hub import snapshot_download
from pathlib import Path

snapshot_download(
    repo_id="meta-llama/Llama-3.1-70B",
    local_dir="/scratch/mj6ux/.cache/models/meta-llama_Llama-3.1-70B"
)

# scratch_cache_dir = Path("/scratch/mj6ux/.cache/models/")
# scratch_cache_dir.mkdir(parents=True, exist_ok=True)

# meta_model_versions = [
#     "meta-llama/Llama-3.1-8B",
#     "meta-llama/Llama-3.1-70B",
#     "meta-llama/Llama-3.1-405B"
# ]

# mistral_model_versions = [
#     "mistralai/Mistral-Large-Instruct-2407"
# ]

# def download_models(model_versions, base_dir):
#     for model_version in model_versions:
#         model_dir = base_dir / model_version.replace("/", "_")
#         try:
#             print(f"Attempting to download: {model_version} to {model_dir}")
#             snapshot_download(
#                 repo_id=model_version,
#                 allow_patterns=["params.json", "consolidated-*.safetensors", "tokenizer.model.v3"],
#                 local_dir=model_dir
#             )
#             print(f"Successfully downloaded: {model_version}")
#         except Exception as e:
#             print(f"Failed to download {model_version}: {e}")

# download_models(meta_model_versions, scratch_cache_dir)

# download_models(mistral_model_versions, scratch_cache_dir)


