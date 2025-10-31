"""
Download Jina ONNX models using Hugging Face Hub

This script downloads ONNX model files from Jina AI repositories:
- jinaai/jina-embeddings-v3
- jinaai/jina-reranker-v2-base-multilingual

Models are saved to the Triton model repository structure.

Optional: set authentication in your environment if the repository requires it.
"""
import sys
from pathlib import Path
from typing import Tuple, Optional


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_model(repo_id: str, file_path: str, target_dir: Path, model_name: str) -> bool:
    """
    Download a model file from Hugging Face Hub
    
    Args:
        repo_id: HuggingFace repository ID
        file_path: Path to file in repository
        target_dir: Local target directory
        model_name: Display name for logging
        
    Returns:
        True if successful, False otherwise
    """
    from huggingface_hub import hf_hub_download
    
    ensure_dir(target_dir)

    print(f"\n{'='*50}", flush=True)
    print(f"Downloading {model_name}", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"Repository: {repo_id}", flush=True)
    print(f"File: {file_path}", flush=True)
    print(f"Target directory: {target_dir}\n", flush=True)

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✓ Downloaded to: {downloaded_path}", flush=True)
        
        expected_file = target_dir / file_path
        if expected_file.exists():
            file_size = expected_file.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"Successfully downloaded: {expected_file}", flush=True)
            print(f"  File size: {file_size:.2f} MB", flush=True)
            return True
        else:
            print(f"Model file not found at expected location: {expected_file}", flush=True)
            return False
            
    except Exception as e:
        print(f"Download failed: {e}", flush=True)
        return False


def main() -> int:
    print("=" * 60, flush=True)
    print("Jina AI Models Download Script", flush=True)
    print("=" * 60, flush=True)
    
    success = True
    
    embeddings_success = download_model(
        repo_id="jinaai/jina-embeddings-v3",
        file_path="onnx/model_fp16.onnx",
        target_dir=Path("model_repository/jina-embeddings-v3/1"),
        model_name="Jina Embeddings v3"
    )
    success = success and embeddings_success
    
    reranker_success = download_model(
        repo_id="jinaai/jina-reranker-v2-base-multilingual",
        file_path="onnx/model_fp16.onnx",
        target_dir=Path("model_repository/jina-reranker-v2/1"),
        model_name="Jina Reranker v2 Base Multilingual"
    )
    success = success and reranker_success

    print(f"\n{'='*60}", flush=True)
    if success:
        print("All models downloaded successfully!", flush=True)
    else:
        print("Some models failed to download", flush=True)
    print("=" * 60, flush=True)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
