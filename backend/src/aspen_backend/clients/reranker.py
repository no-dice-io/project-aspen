from pydantic import (
    BaseModel
    , Field
    , PrivateAttr
)
from typing import (
    Optional
    , List
    , Dict
    , Any
)
from sentence_transformers import CrossEncoder
import numpy as np

class RerankerAPI(BaseModel):
    default_model: Optional[str] = Field(
        'cross-encoder/ms-marco-MiniLM-L-6-v2'
        , description='Default model'
    )

    _model: CrossEncoder = PrivateAttr()

    def __init__(self, model: Optional[str] = None):
        
        super().__init__()
        
        model = model if model else self.default_model
        self._model = CrossEncoder(model)

    def stream_rerank(self, batch: list) -> List:
        
        # Create pairs of (query, document) for CrossEncoder
        query = batch[0][0]
        scores = self._model.predict(batch)
        # Get original documents from the pairs
        documents = [pair[1] for pair in batch]
        # Sort by scores
        ranked_indices = np.argsort(scores)[::-1]
        ranked_documents = [documents[i] for i in ranked_indices]
        ranked_scores = scores[ranked_indices]

        payload = {
            "query": query
            , "ranked_documents": ranked_documents
            , "scores": ranked_scores.tolist()
        }        

        return payload
    
    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return output