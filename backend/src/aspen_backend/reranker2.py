import litserve as ls
from sentence_transformers import CrossEncoder
import numpy as np
from typing import Optional, List, Dict, Any

class RerankerAPI(ls.LitAPI):
    def setup(self, device: Optional[str] = "cpu"):
        # Initialize CrossEncoder with a pre-trained model
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
    def decode_request(self, request: Dict[str, Any]) -> List[List[str]]:
        # Handle single request format
        if isinstance(request, list):
            request = request[0]
        
        data = request.get("batch", {})
        query = data.get("query", "")
        documents = data.get("passages", [])
        
        # Create pairs of (query, document) for CrossEncoder
        return [[query, doc] for doc in documents]
    
    def predict(self, payload: List[List[str]]) -> Dict[str, Any]:
        # CrossEncoder.predict already handles batching internally
        payloads = list()

        for batch in payload:
            query = batch[0][0]

            scores = self.model.predict(batch)
            # Get original documents from the pairs
            documents = [pair[1] for pair in batch]
            # Sort by scores
            ranked_indices = np.argsort(scores)[::-1]
            ranked_documents = [documents[i] for i in ranked_indices]
            ranked_scores = scores[ranked_indices]

            payload = {
                "query": query,
                "ranked_documents": ranked_documents,
                "scores": ranked_scores.tolist()
            }
            payloads.append(payload)

        return payloads
    
    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return output

if __name__ == "__main__":
    # Initialize with batch_size=1 since CrossEncoder handles batching internally
    server = ls.LitServer(
        RerankerAPI(),
        accelerator="cpu",
        max_batch_size=16
    )
    server.run(port=8000)