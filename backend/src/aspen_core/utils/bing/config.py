from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import os

class BingConfig(BaseModel):
    key: str = Field(..., description="BING_API_KEY")
    base_uri: str = Field(..., description="BING_BASE_URI")
    endpoints: dict = {
        "web_search": "v7.0/search",
        "search": "v7.0/search",
        "news": "v7.0/news",
    }

    @classmethod
    def from_env(cls) -> "BingConfig":
        key = os.getenv("BING_API_KEY")
        base_uri = os.getenv("BING_BASE_URI", "https://api.bing.microsoft.com")

        for req in [key, base_uri]:
            assert req, f"Key/base url missing; key: {key}, base_url: {base_uri}"
            #if not req:
            #    err = f"Bing subscription key or base url not set. Please set env vars BING_API_KEY and BING_BASE_URI"
            #    raise ValueError(err)
        
        return cls(
            key=key
            , base_uri=base_uri
        )
    
    def endpoint(self, endpoint: str) -> Optional[str]:
        if endpoint not in self.endpoints:
            wrn = f"Endpoint {endpoint} is currently unsupported"
            logging.warning(wrn)
            return None
        
        path = os.path.join(
            self.base_uri
            , self.endpoints[endpoint]
        )

        return path
        

