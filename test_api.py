"""
Test script for Jina-Embeddings-v3 API
"""
import requests
import json
import time

API_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("\n=== Testing Health Check ===")
    start = time.time()
    response = requests.get(f"{API_URL}/health")
    elapsed = (time.time() - start) * 1000
    print(f"Status Code: {response.status_code}")
    print(f"Response Time: {elapsed:.2f}ms")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_list_models():
    """Test list models endpoint"""
    print("\n=== Testing List Models ===")
    start = time.time()
    response = requests.get(f"{API_URL}/v1/models")
    elapsed = (time.time() - start) * 1000
    print(f"Status Code: {response.status_code}")
    print(f"Response Time: {elapsed:.2f}ms")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_single_text_embedding():
    """Test embedding generation for a single text"""
    print("\n=== Testing Single Text Embedding ===")
    
    payload = {
        "input": "What is the capital of France?",
        "model": "jina-embeddings-v3",
        "task": "retrieval.query"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    start = time.time()
    response = requests.post(f"{API_URL}/v1/embeddings", json=payload)
    elapsed = (time.time() - start) * 1000
    print(f"Status Code: {response.status_code}")
    print(f"Response Time: {elapsed:.2f}ms")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Model: {result['model']}")
        print(f"Number of embeddings: {len(result['data'])}")
        print(f"Embedding dimension: {len(result['data'][0]['embedding'])}")
        print(f"Usage: {json.dumps(result['usage'], indent=2)}")
        print(f"First 10 values of embedding: {result['data'][0]['embedding'][:10]}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_batch_text_embeddings():
    """Test embedding generation for multiple texts"""
    print("\n=== Testing Batch Text Embeddings ===")
    
    payload = {
        "input": [
            "What is the capital of France?",
            "Paris is the capital of France.",
            "Machine learning is a subset of artificial intelligence."
        ],
        "model": "jina-embeddings-v3",
        "task": "retrieval.query"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    start = time.time()
    response = requests.post(f"{API_URL}/v1/embeddings", json=payload)
    elapsed = (time.time() - start) * 1000
    print(f"Status Code: {response.status_code}")
    print(f"Response Time: {elapsed:.2f}ms")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Model: {result['model']}")
        print(f"Number of embeddings: {len(result['data'])}")
        print(f"Embedding dimension: {len(result['data'][0]['embedding'])}")
        print(f"Usage: {json.dumps(result['usage'], indent=2)}")
        
        for i, data in enumerate(result['data']):
            print(f"\nEmbedding {i} (index {data['index']}):")
            print(f"  First 10 values: {data['embedding'][:10]}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_different_tasks():
    """Test different task types"""
    print("\n=== Testing Different Task Types ===")
    
    tasks = ["retrieval.query", "retrieval.passage", "classification", "text-matching"]
    text = "Machine learning is transforming the world."
    
    for task in tasks:
        payload = {
            "input": text,
            "model": "jina-embeddings-v3",
            "task": task
        }
        
        print(f"\nTask: {task}")
        start = time.time()
        response = requests.post(f"{API_URL}/v1/embeddings", json=payload)
        elapsed = (time.time() - start) * 1000
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {elapsed:.2f}ms")
        
        if response.status_code == 200:
            result = response.json()
            embedding = result['data'][0]['embedding']
            print(f"Embedding dimension: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")
        else:
            print(f"Error: {response.text}")

def compute_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings"""
    import numpy as np
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def test_semantic_similarity():
    """Test semantic similarity between texts"""
    print("\n=== Testing Semantic Similarity ===")
    
    texts = [
        "What is artificial intelligence?",
        "Define AI and machine learning",
        "How to cook pasta?"
    ]
    
    payload = {
        "input": texts,
        "model": "jina-embeddings-v3",
        "task": "text-matching"
    }
    
    start = time.time()
    response = requests.post(f"{API_URL}/v1/embeddings", json=payload)
    elapsed = (time.time() - start) * 1000
    print(f"Response Time: {elapsed:.2f}ms")
    
    if response.status_code == 200:
        result = response.json()
        embeddings = [data['embedding'] for data in result['data']]
        
        print(f"\nTexts:")
        for i, text in enumerate(texts):
            print(f"  {i}: {text}")
        
        print(f"\nSimilarity matrix:")
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                similarity = compute_similarity(embeddings[i], embeddings[j])
                print(f"  Text {i} <-> Text {j}: {similarity:.4f}")
    else:
        print(f"Error: {response.text}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Jina-Embeddings-v3 API Test Suite")
    print("=" * 60)
    
    try:
        # Basic tests
        test_health()
        test_list_models()
        
        # Embedding tests
        test_single_text_embedding()
        test_batch_text_embeddings()
        test_different_tasks()
        
        # Advanced tests
        test_semantic_similarity()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API at http://localhost:8000")
        print("Make sure the API is running with: docker-compose up")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

