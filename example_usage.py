"""
Example usage of Jina-Embeddings-v3 Embedding API
"""
import requests
import json


def test_single_text():
    """Test with single text"""
    url = "http://localhost:8000/v1/embeddings"
    
    payload = {
        "input": "Hello, world! This is a test of the Jina-Embeddings-v3 model.",
        "model": "jina-embeddings-v3",
        "task": "text-matching"
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        embedding = result['data'][0]['embedding']
        print(f"✓ Single text embedding")
        print(f"  - Status: Success")
        print(f"  - Embedding dimension: {len(embedding)}")
        print(f"  - First 5 values: {embedding[:5]}")
        print(f"  - Tokens used: {result['usage']['total_tokens']}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")


def test_multiple_texts():
    """Test with multiple texts (batching)"""
    url = "http://localhost:8000/v1/embeddings"
    
    texts = [
        "What is machine learning?",
        "Triton Inference Server provides optimized inference.",
        "FastAPI is a modern web framework for Python.",
        "ONNX is an open format for machine learning models."
    ]
    
    payload = {
        "input": texts,
        "model": "jina-embeddings-v3",
        "task": "text-matching"
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Multiple texts embedding (batch of {len(texts)})")
        print(f"  - Status: Success")
        print(f"  - Number of embeddings: {len(result['data'])}")
        
        for i, item in enumerate(result['data']):
            print(f"  - Text {i+1}: dimension = {len(item['embedding'])}")
        
        print(f"  - Total tokens: {result['usage']['total_tokens']}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")


def test_with_openai_client():
    """Test using OpenAI Python client"""
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key="dummy-key",
            base_url="http://localhost:8000/v1"
        )
        
        # Note: OpenAI client doesn't support custom task parameter
        # Use requests library for full control
        response = client.embeddings.create(
            input="Testing with OpenAI client library",
            model="jina-embeddings-v3"
        )
        
        print(f"\n✓ OpenAI client test")
        print(f"  - Status: Success")
        print(f"  - Embedding dimension: {len(response.data[0].embedding)}")
        print(f"  - First 5 values: {response.data[0].embedding[:5]}")
        
    except ImportError:
        print(f"\n⚠ OpenAI client not installed")
        print(f"  Install with: pip install openai")
    except Exception as e:
        print(f"\n✗ Error with OpenAI client: {e}")


def test_health():
    """Test health endpoint"""
    url = "http://localhost:8000/health"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Health check")
        print(f"  - Status: {result['status']}")
        print(f"  - Triton: {result['triton_server']}")
        print(f"  - Model: {result['model']}")
    else:
        print(f"✗ Health check failed: {response.status_code}")


def test_list_models():
    """Test list models endpoint"""
    url = "http://localhost:8000/v1/models"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ List models")
        print(f"  - Available models: {len(result['data'])}")
        for model in result['data']:
            print(f"    - {model['id']}")
    else:
        print(f"✗ List models failed: {response.status_code}")


def compute_similarity(text1: str, text2: str):
    """Calculate similarity between 2 texts"""
    import numpy as np
    
    url = "http://localhost:8000/v1/embeddings"
    
    payload = {
        "input": [text1, text2],
        "model": "jina-embeddings-v3",
        "task": "text-matching"
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        emb1 = np.array(result['data'][0]['embedding'])
        emb2 = np.array(result['data'][1]['embedding'])
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        print(f"\n✓ Similarity computation")
        print(f"  - Text 1: {text1[:50]}...")
        print(f"  - Text 2: {text2[:50]}...")
        print(f"  - Cosine similarity: {similarity:.4f}")
    else:
        print(f"✗ Error: {response.status_code}")


def test_different_tasks():
    """Test with different task types"""
    url = "http://localhost:8000/v1/embeddings"
    
    tasks = ["retrieval.query", "retrieval.passage", "text-matching", "classification"]
    text = "Machine learning is a subset of artificial intelligence."
    
    print(f"\n✓ Testing different tasks")
    print(f"  Text: {text[:50]}...")
    
    for task in tasks:
        payload = {
            "input": text,
            "model": "jina-embeddings-v3",
            "task": task
        }
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            embedding = result['data'][0]['embedding']
            print(f"  - Task '{task}': dimension = {len(embedding)}")
        else:
            print(f"  - Task '{task}': Error {response.status_code}")


if __name__ == "__main__":
    print("=" * 60)
    print("Jina-Embeddings-v3 Embedding API - Example Usage")
    print("=" * 60)
    
    # Run tests
    test_health()
    test_list_models()
    test_single_text()
    test_multiple_texts()
    test_with_openai_client()
    
    # Similarity example
    compute_similarity(
        "The cat is sleeping on the couch.",
        "A feline is resting on the sofa."
    )
    
    # Test different tasks
    test_different_tasks()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print("\nAvailable tasks:")
    print("  - retrieval.query: For search queries")
    print("  - retrieval.passage: For documents to be retrieved")
    print("  - text-matching: For similarity/matching tasks")
    print("  - classification: For classification tasks")
    print("  - separation: For separation tasks")

