from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import logging

# At this point I probs could have turned this into a Pydantic model
# Maybe I'll refactor for that later
@dataclass
class AspenRequest:
    messages: Optional[List[Dict[str, str]]] = None
    model: Optional[str] = None

    query: Optional[str] = None
    documents: Optional[List[str]] = None

    verbose: Optional[bool] = False
    kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.messages is None and self.query is None:
            raise ValueError("Messages or query must be provided")
        
        if self.query is not None and self.documents is None:
            raise ValueError("Documents must be provided with query")
        
        if self.documents is not None and self.query is None:
            raise ValueError("Query must be provided with documents")  
        
    @property
    def request_type(self) -> str:
        if self.messages is not None:
            return "llm"
        elif self.query is not None:
            return "rerank"
        else:
            wrn = f"Unknown request type: {self}"
            logging.warning(wrn)

            return "unknown"
        
    @property
    def pairs(self) -> List[List[str]]:
        if self.request_type == "llm":
            return None
        
        return [[self.query, doc] for doc in self.documents]
    
    @property 
    def llm_kwargs(self) -> Dict[str, Any]:

        required_keys = ["messages"]
        payload = {
            "messages": self.messages
            , "model": self.model
            , "verbose": self.verbose
            , "kwargs": self.kwargs
        }

        for key in required_keys:
            if payload[key] is None:
                raise ValueError(f"Missing required key: {key}")
        
        return payload
    
    @property
    def request_info(self) -> str:
        if self.request_type == "llm":
            info = f"""
            Request to chat w/ model: {
                self.model 
                if self.model else 'default'
            }
            Messages: {self.messages}
            """
            return info
        elif self.request_type == "rerank":
            info = f"""
            Query: {self.query}
            Reranking {len(self.documents)} documents
            """
            return info
        else:
            return "Unknown request type"