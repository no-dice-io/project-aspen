import requests
import json
from time import time

# Server URL
url = "http://localhost:8000/predict"

# Example data for reranking
rerank_request = {
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a type of AI that learns from data",
        "Python is a programming language",
        "ML systems improve through experience"
    ],
    "kwargs": {},
    "verbose": True
}

# Example data for LLM using chat format
llm_request = {
    "messages": [
        {"role": "user", "content": "Explain what machine learning is in one sentence."}
    ],
    "kwargs": {
        "model": "qwen",  # or whatever model you're using
        "temperature": 0.7,
        "stream": True  # Enable streaming
    },
    "verbose": True
}

# Headers
headers = {
    "Content-Type": "application/json",
    "Accept": "text/event-stream"  # For streaming responses
}

# Send reranking request
print("\nSending reranking request...")
rerank_start = time()
rerank_response = requests.post(
    url,
    headers={"Content-Type": "application/json"},  # Regular headers for reranking
    json=rerank_request,
    stream=True
)
for line in rerank_response.iter_lines():
    if line:
        chunk = line.decode('utf-8')
        print(chunk)
        #print(chunk.get('response', ''))
rerank_elapsed = time() - rerank_start
print(f"Reranking time: {rerank_elapsed:.2f} seconds")

# Send LLM request with streaming
print("\nSending LLM request...")
llm_start = time()
with requests.post(
    url,
    headers=headers,
    json=llm_request,
    stream=True
) as llm_response:
    
    print("Streaming LLM Response:")
    for line in llm_response.iter_lines():
        if line:
            chunk = json.loads(line)
            print(chunk.get('response', ''), end='', flush=True)

llm_elapsed = time() - llm_start
time_elapsed = rerank_elapsed + llm_elapsed

elapsed = f"""
    Reranking time: {rerank_elapsed:.2f} seconds
    LLM time: {llm_elapsed:.2f} seconds
    Total time elapsed: {time_elapsed:.2f} seconds
"""
print(elapsed)