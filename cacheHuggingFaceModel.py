import os
import shutil
from huggingface_hub import snapshot_download

# Optional: Trigger Hugging Face cache migration if required
try:
    from transformers.utils import move_cache
    move_cache()
except ImportError:
    # Older versions of transformers may not have move_cache
    pass
except Exception as e:
    print(f"Cache migration warning: {e}")

def main():
    model_name = "BAAI/bge-base-en-v1.5"
    cache_dir = snapshot_download(repo_id=model_name)
    destination_dir = "bge-base-en-v1.5"
    if not os.path.exists(destination_dir):
        shutil.copytree(cache_dir, destination_dir)
        print(f"Model moved to {destination_dir}")
    else:
        print(f"Destination already exists at {destination_dir}")

if __name__ == "__main__":
    main()