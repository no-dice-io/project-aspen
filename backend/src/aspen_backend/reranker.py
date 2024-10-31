import litserve as ls
from sentence_transformers import CrossEncoder
import numpy as np

class RerankingAPI(ls.LitAPI):
    def setup(self, device: str = "cpu"):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def decode_request(self, request) -> dict:
        # Handle both batched and unbatched requests
        print(request)
        if isinstance(request, list):
            request = request[0]  # Take first item if batched
            
        request = request.get("batch")
        if request is None:
            raise ValueError("Request must contain a 'batch' field")
        
        req_keys = ["query", "passages"]
        for key in req_keys:
            if key not in request:
                raise ValueError(f"Request must contain a '{key}' field")

        query = request.get("query")
        passages = request.get("passages")
        pairs = [[query, passage] for passage in passages]

        return {
            "query": query,
            "passages": passages,
            "pairs": pairs
        }
    
    def predict(self, batch: dict):
        # Handle both batched and unbatched inputs
        print(batch)
        if isinstance(batch, list):
            batch = batch[0]
            
        pairs = batch.get("pairs")
        passages = batch.get("passages")

        scores = self.model.predict(pairs)
        print(pairs, scores)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_scores = scores[ranked_indices]
        ranked_passages = [passages[i] for i in ranked_indices]
        
        return {
            "ranked_passages": ranked_passages,
            "scores": ranked_scores.tolist()
        }
    
    def encode_response(self, output):
        print(output)
        return {"hi I'm carl": "hi I'm carl"}   
    

if __name__ == "__main__":
    server = ls.LitServer(
        RerankingAPI(),
        accelerator="auto",
        max_batch_size=32
    )
    server.run(port=8000)