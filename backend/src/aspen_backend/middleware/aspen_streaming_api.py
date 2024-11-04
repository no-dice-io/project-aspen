import litserve as ls
from typing import (
    Optional
    , Dict
    , List
    , Any
    , Generator
)
from aspen_backend.clients import (
    RerankerAPI
    , OllamaAPI
    , AspenRequest
)
import logging

logging.basicConfig(level=logging.INFO)

class AspenStreamingAPI(ls.LitAPI):
    def setup(self, device) -> None:
        self.reranker = RerankerAPI()
        self.ollama = OllamaAPI()
        self.logger = logging.getLogger(__name__)

    def decode_request(self, request: AspenRequest):
        if request.verbose:
            self.logger.info(request.request_info)

        return request

    def predict(self, request: AspenRequest) -> Generator[Dict[str, Any], None, None]:
        if request.request_type == "llm":
            for chunk in self.ollama.chat_stream(**request.llm_kwargs):
                if request.verbose:
                    info = f"LLM {chunk.get('model')} response: {chunk.get('response')}"
                    self.logger.info(
                        info if not chunk.get("done") else chunk.get("inference_metrics")
                    )
                yield chunk
        elif request.request_type == "rerank":
            response = self.reranker.stream_rerank(request.pairs)
            if request.verbose:
                self.logger.info(f"Reranker response: {response}")
            yield {
                "response": response
            }
        else:
            raise ValueError(f"Unsupported request type: {request.request_type}")

    def encode_response(self, output):
        for response in output:
            yield response