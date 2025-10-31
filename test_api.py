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

def test_basic_rerank():
    """Test basic reranking functionality"""
    print("\n=== Testing Basic Rerank ===")
    
    payload = {
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a branch of artificial intelligence.",
            "Python is a popular programming language.",
            "Deep learning is a subset of machine learning.",
            "The weather is nice today.",
            "Neural networks are used in deep learning."
        ],
        "model": "jina-reranker-v2",
        "top_n": 3
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    start = time.time()
    response = requests.post(f"{API_URL}/v1/rerank", json=payload)
    elapsed = (time.time() - start) * 1000
    print(f"Status Code: {response.status_code}")
    print(f"Response Time: {elapsed:.2f}ms")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Model: {result['model']}")
        print(f"Number of results: {len(result['data'])}")
        print(f"\nReranked documents (top {payload['top_n']}):")
        for i, item in enumerate(result['data']):
            print(f"  {i+1}. [Index: {item['index']}, Score: {item['relevance_score']:.4f}]")
            if 'document' in item and item['document']:
                print(f"     {item['document']}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_rerank_without_return_documents():
    """Test reranking without returning document text"""
    print("\n=== Testing Rerank without Document Text ===")
    
    payload = {
        "query": "artificial intelligence",
        "documents": [
            "AI is transforming the world.",
            "Cooking recipes for beginners.",
            "Machine learning algorithms."
        ],
        "model": "jina-reranker-v2",
        "return_documents": False
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    start = time.time()
    response = requests.post(f"{API_URL}/v1/rerank", json=payload)
    elapsed = (time.time() - start) * 1000
    print(f"Status Code: {response.status_code}")
    print(f"Response Time: {elapsed:.2f}ms")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nResults (indices and scores only):")
        for i, item in enumerate(result['data']):
            print(f"  {i+1}. Index: {item['index']}, Score: {item['relevance_score']:.4f}")
            print(f"     Document included: {'document' in item and item['document'] is not None}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_rerank_with_dict_documents():
    """Test reranking with dictionary documents"""
    print("\n=== Testing Rerank with Dict Documents ===")
    
    payload = {
        "query": "climate change solutions",
        "documents": [
            {"text": "Renewable energy is key to fighting climate change.", "id": "doc1"},
            {"text": "Pizza recipes from Italy.", "id": "doc2"},
            {"text": "Carbon capture technology can help reduce emissions.", "id": "doc3"}
        ],
        "model": "jina-reranker-v2"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    start = time.time()
    response = requests.post(f"{API_URL}/v1/rerank", json=payload)
    elapsed = (time.time() - start) * 1000
    print(f"Status Code: {response.status_code}")
    print(f"Response Time: {elapsed:.2f}ms")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nReranked documents:")
        for i, item in enumerate(result['data']):
            print(f"  {i+1}. Index: {item['index']}, Score: {item['relevance_score']:.4f}")
            if 'document' in item and item['document']:
                print(f"     Document: {item['document']}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_rerank_score_ordering():
    """Test that rerank scores are properly ordered"""
    print("\n=== Testing Rerank Score Ordering ===")
    
    payload = {
        "query": "python programming",
        "documents": [
            "Python is a high-level programming language.",
            "Java is also a programming language.",
            "The sky is blue.",
            "Python programming tutorials for beginners.",
            "Cooking with olive oil."
        ],
        "model": "jina-reranker-v2"
    }
    
    start = time.time()
    response = requests.post(f"{API_URL}/v1/rerank", json=payload)
    elapsed = (time.time() - start) * 1000
    print(f"Response Time: {elapsed:.2f}ms")
    
    if response.status_code == 200:
        result = response.json()
        scores = [item['relevance_score'] for item in result['data']]
        
        print(f"\nScores in order:")
        for i, (item, score) in enumerate(zip(result['data'], scores)):
            print(f"  {i+1}. Score: {score:.4f} (Index: {item['index']})")
        
        is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        print(f"\nScores properly sorted (descending): {is_sorted}")
        
        if is_sorted:
            print("[OK] Scores are correctly ordered from highest to lowest")
        else:
            print("[FAIL] Scores are NOT properly ordered")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_rerank_top_n():
    """Test top_n parameter with different values"""
    print("\n=== Testing Rerank top_n Parameter ===")
    
    documents = [
        "Machine learning is a subset of AI.",
        "Python is used for data science.",
        "The cat sat on the mat.",
        "Deep learning uses neural networks.",
        "JavaScript is for web development."
    ]
    
    for top_n in [1, 2, 3, None]:
        payload = {
            "query": "artificial intelligence and machine learning",
            "documents": documents,
            "model": "jina-reranker-v2",
            "top_n": top_n
        }
        
        print(f"\nTesting with top_n={top_n}")
        response = requests.post(f"{API_URL}/v1/rerank", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            expected_count = top_n if top_n is not None else len(documents)
            actual_count = len(result['data'])
            print(f"  Expected results: {expected_count}, Got: {actual_count}")
            if actual_count == expected_count:
                print(f"  [OK] Correct number of results")
            else:
                print(f"  [FAIL] Incorrect number of results")
        else:
            print(f"  Error: {response.text}")
            return False
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Jina AI API Test Suite (Embeddings + Reranking)")
    print("=" * 60)
    
    try:
        test_health()
        test_list_models()
        
        test_single_text_embedding()
        test_batch_text_embeddings()
        test_different_tasks()
        
        test_semantic_similarity()
        
        test_basic_rerank()
        test_rerank_without_return_documents()
        test_rerank_with_dict_documents()
        test_rerank_score_ordering()
        test_rerank_top_n()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Cannot connect to API at http://localhost:8000")
        print("Make sure the API is running with: docker-compose up")
    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    main()
