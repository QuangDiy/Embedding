"""
Script to verify all required files for Jina-Embeddings-v3 model
"""
import os
import sys
from pathlib import Path

# Required files
REQUIRED_FILES = [
    "model.onnx",
    "model.onnx_data",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json"
]

MODEL_DIR = Path("model_repository/jina-embeddings-v3/1")

def check_files():
    """Check all required files"""
    print("=" * 60)
    print("Checking Model Files - Jina-Embeddings-v3 ONNX")
    print("=" * 60)
    print()
    
    missing_files = []
    found_files = []
    
    print(f"Model directory: {MODEL_DIR.absolute()}")
    print()
    
    for filename in REQUIRED_FILES:
        filepath = MODEL_DIR / filename
        
        if filepath.exists():
            size = filepath.stat().st_size
            size_mb = size / (1024 * 1024)
            
            if size_mb < 1:
                size_str = f"{size / 1024:.2f} KB"
            else:
                size_str = f"{size_mb:.2f} MB"
            
            print(f"✓ {filename:30s} ({size_str})")
            found_files.append(filename)
        else:
            print(f"✗ {filename:30s} (MISSING)")
            missing_files.append(filename)
    
    print()
    print("=" * 60)
    print(f"Result: {len(found_files)}/{len(REQUIRED_FILES)} files")
    print("=" * 60)
    
    if missing_files:
        print()
        print("⚠️  MISSING FILES:")
        for f in missing_files:
            print(f"   - {f}")
        print()
        print("📥 Download all files from:")
        print("   https://huggingface.co/jinaai/jina-embeddings-v3/tree/main/onnx")
        print()
        print("📂 Place them in directory:")
        print(f"   {MODEL_DIR.absolute()}")
        print()
        return False
    else:
        print()
        print("✅ ALL FILES ARE READY!")
        print()
        print("Next steps:")
        print("  1. Run: docker-compose build")
        print("  2. Run: docker-compose up -d")
        print("  3. Test: curl http://localhost:8000/health")
        print()
        return True


def verify_onnx_structure():
    """Verify ONNX model structure"""
    model_path = MODEL_DIR / "model.onnx"
    
    if not model_path.exists():
        return
    
    try:
        import onnx
        
        print()
        print("=" * 60)
        print("Checking ONNX Model Structure")
        print("=" * 60)
        print()
        
        model = onnx.load(str(model_path))
        
        print("Inputs:")
        for inp in model.graph.input:
            print(f"  - Name: {inp.name}")
            print(f"    Type: {inp.type.tensor_type.elem_type}")
            dims = [d.dim_value if d.dim_value > 0 else -1 for d in inp.type.tensor_type.shape.dim]
            print(f"    Shape: {dims}")
        
        print()
        print("Outputs:")
        for out in model.graph.output:
            print(f"  - Name: {out.name}")
            print(f"    Type: {out.type.tensor_type.elem_type}")
            dims = [d.dim_value if d.dim_value > 0 else -1 for d in out.type.tensor_type.shape.dim]
            print(f"    Shape: {dims}")
        
        print()
        print("✓ Model structure verified")
        print()
        
    except ImportError:
        print()
        print("⚠️  Package 'onnx' is not installed")
        print("   Install: pip install onnx")
        print()
    except Exception as e:
        print()
        print(f"❌ Error reading ONNX model: {e}")
        print()


def verify_tokenizer():
    """Verify tokenizer files"""
    try:
        from transformers import AutoTokenizer
        
        print()
        print("=" * 60)
        print("Checking Tokenizer")
        print("=" * 60)
        print()
        
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR.absolute()))
        
        # Test tokenization
        test_text = "Hello, this is a test!"
        tokens = tokenizer(test_text, return_tensors="np")
        
        print(f"Test text: {test_text}")
        print(f"Token IDs shape: {tokens['input_ids'].shape}")
        print(f"Attention mask shape: {tokens['attention_mask'].shape}")
        print()
        print("✓ Tokenizer verified and working")
        print()
        
    except ImportError:
        print()
        print("⚠️  Package 'transformers' is not installed")
        print("   Install: pip install transformers")
        print()
    except Exception as e:
        print()
        print(f"❌ Error loading tokenizer: {e}")
        print()


if __name__ == "__main__":
    # Check files
    files_ok = check_files()
    
    if files_ok:
        # Verify ONNX structure
        verify_onnx_structure()
        
        # Verify tokenizer
        verify_tokenizer()
        
        sys.exit(0)
    else:
        sys.exit(1)

