import litserve as ls
from sentence_transformers import CrossEncoder
import numpy as np
from typing import Optional
import os


class RerankingAPI(ls.LitAPI):
    def setup(self, device: Optional[str] = "cpu"):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    def decode_request(self, request: dict) -> list:
        # Handle both batched and unbatched requests
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

        payload =  [
            [query, passage] 
            for passage in passages
        ]

        print("decoding: ", payload)

        return payload
    
    def batch(self, inputs: list) -> list:

        print("batch: ", inputs)
        return inputs
    
    def predict(self, batch: list) -> list:
        # Handle both batched and unbatched inputs
        print("prepredict ", batch)
        batch = batch[0] if len(batch) == 1 else batch
        
        query = batch[0][0]
        passages = [passage for _, passage in batch]

        scores = self.model.predict(batch)
        ranked_indices = np.argsort(scores)[::-1]
        #ranked_scores = scores[ranked_indices]
        ranked_passages = [(passages[i], scores[i]) for i in ranked_indices]


        print("predict: ", ranked_passages)

        return ranked_passages
    
    def unbatch(self, output: list) -> list:
        
        print("unbatch: ", output)  
        return list(output)
    
    def encode_response(self, output):

        print(output)
        payload = {
            "ranked_passages": output[0]
            , "ranked_scores": output[1]
        }   
        print(payload)

        return payload

if __name__ == "__main__":
    server = ls.LitServer(
        RerankingAPI(),
        accelerator="auto",
        max_batch_size=32
    )
    server.run(port=8000)