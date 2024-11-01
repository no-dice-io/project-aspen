import asyncio
import aiohttp
import json
from typing import Dict, Any

async def make_request(session: aiohttp.ClientSession, data: Dict[str, Any], request_id: int) -> None:
    """Make async request to reranker endpoint"""
    try:
        async with session.post("http://localhost:8000/predict", json=data) as response:
            results = await response.json()
            print(f"\nRequest {request_id} Results:")
            print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error in request {request_id}: {str(e)}")

async def main():
    # Define three different test requests
    requests = [
        {
            "batch": {
                "query": "What is machine learning?",
                "passages": [
                    "Machine learning is a branch of AI focused on data-driven learning.",
                    "Natural language processing deals with text understanding.",
                    "Machine learning allows computers to improve through experience.",
                ]
            }
        },
        {
            "batch": {
                "query": "How does deep learning work?",
                "passages": [
                    "Deep learning uses neural networks with multiple layers.",
                    "Traditional machine learning requires manual feature engineering.",
                    "Deep learning automatically learns hierarchical representations.",
                    "Neural networks are inspired by biological brains.",
                ]
            }
        },
        {
            "batch": {
                "query": "What is natural language processing?",
                "passages": [
                    "NLP enables computers to understand human language.",
                    "Natural language processing combines linguistics and machine learning.",
                    "Language models are key components of NLP systems.",
                    "NLP applications include translation and text generation.",
                    "Transformers have revolutionized natural language processing.",
                ]
            }
        }
    ]

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, request_data in enumerate(requests, 1):
            task = asyncio.create_task(make_request(session, request_data, i))
            tasks.append(task)
        
        print("Sending requests...")
        await asyncio.gather(*tasks)
        print("\nAll requests completed!")

if __name__ == "__main__":
    asyncio.run(main())