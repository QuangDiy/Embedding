"""
Download Jina-Embeddings-v3 ONNX model file using Hugging Face Hub

This script downloads only the model_fp16.onnx file from `jinaai/jina-embeddings-v3`
and saves it to `model_repository/jina-embeddings-v3/1/` inside the Triton model
repository structure.

Optional: set authentication in your environment if the repository requires it.
"""
import sys
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ huggingface_hub is not installed. Install with: pip install huggingface_hub", flush=True)
        return 1

    repo_id = "jinaai/jina-embeddings-v3"
    file_path = "onnx/model_fp16.onnx"
    target_dir = Path("model_repository/jina-embeddings-v3/1")
    ensure_dir(target_dir)

    print("==========================================", flush=True)
    print("Download Jina-Embeddings-v3 ONNX Model", flush=True)
    print("==========================================\n", flush=True)
    print(f"Repository: {repo_id}", flush=True)
    print(f"File: {file_path}", flush=True)
    print(f"Target directory: {target_dir}\n", flush=True)

    print("Downloading model file...", flush=True)

    try:
        # Download the file directly to the target directory
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✓ Downloaded to: {downloaded_path}", flush=True)
    except Exception as e:
        print(f"❌ Download failed: {e}", flush=True)
        return 1

    # Verify the file was downloaded
    model_file = target_dir / "onnx" / "model_fp16.onnx"
    if model_file.exists():
        file_size = model_file.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"\n✓ Successfully downloaded: {model_file}", flush=True)
        print(f"  File size: {file_size:.2f} MB", flush=True)
    else:
        # Check alternative location (hf_hub_download might create nested structure)
        alt_model_file = target_dir / "model_fp16.onnx"
        if alt_model_file.exists():
            file_size = alt_model_file.stat().st_size / (1024 * 1024)
            print(f"\n✓ Successfully downloaded: {alt_model_file}", flush=True)
            print(f"  File size: {file_size:.2f} MB", flush=True)
        else:
            print(f"❌ Model file not found at expected location", flush=True)
            print(f"   Searched: {model_file}", flush=True)
            print(f"   Also searched: {alt_model_file}", flush=True)
            return 1

    print("\n==========================================", flush=True)
    print("Download completed!", flush=True)
    print("==========================================", flush=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


