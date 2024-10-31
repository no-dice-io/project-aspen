import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "batch": {
            "query": "What is machine learning?",
            "passages": [
                "Machine learning is a branch of AI focused on data-driven learning.",
                "Natural language processing deals with text understanding.",
                "Machine learning allows computers to improve through experience.",
            ]
        }
    }
)

results = response.json()
print(results)
for passage, score in zip(results["ranked_passages"], results["scores"]):
    print(f"Score: {score:.4f} | Passage: {passage}")