import ollama as ollm
from pydantic import (
    BaseModel
    , Field
    , PrivateAttr
    , model_validator
)
from typing import (
    Optional
    , List
    , Dict
    , Any
    , Generator
)

class OllamaAPI(BaseModel):
    endpoint: Optional[str] = Field(
        'http://localhost:11434'
        , description='Ollama endpoint'
    )
    default_model: Optional[str] = Field(
        'qwen'
        , description='Local Ollama model'
    )

    _client: ollm.Client = PrivateAttr()

    @model_validator(mode='after')
    def initialize(self):
        self._client = ollm.Client(
            host=self.endpoint
        )

        return self

    def decode_request(self, request: dict) -> List[str]:
        # Handle single request format
        if isinstance(request, list):
            request = request[0]
        
        return request.get("prompts", [])

    def generate_response(
        self
        , prompt: str
        , model: Optional[str] = None
        , stream: Optional[bool] = False
        , verbose: Optional[bool] = False
        , kwargs: Optional[Dict[str, Any]] = {}
    ) -> Dict[str, Any]:
        
        model = model if model else self.default_model

        response = self._client.generate(
            model=model
            , prompt=prompt
            , stream=stream
            , options=kwargs
        )

        payload = {
            'response': response.get("response")
            , 'model': model
        }

        if verbose:
            payload["inference_metrics"] = self.compute_inference_metrics(response)

        return payload
    
    def chat(
        self
        , messages: List[Dict[str, str]]
        , model: Optional[str] = None
        , stream: Optional[bool] = False
        , verbose: Optional[bool] = False
        , kwargs: Optional[Dict[str, Any]] = {}
    ) -> Dict[str, Any]:
        """
        Send a chat request to Ollama's chat endpoint.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Optional model name (defaults to self.default_model)
            stream: Whether to stream the response
            verbose: Whether to include inference metrics
            kwargs: Additional options to pass to Ollama
        """
        model = model if model else self.default_model
        
        # Validate message format
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("Messages must be a list of dicts with 'role' and 'content' keys")
        
        response = self._client.chat(
            model=model,
            messages=messages,
            stream=stream,
            options=kwargs
        )
        
        # Extract the assistant's response
        payload = {
            'response': response.get("message", {}).get("content"),
            'model': model,
            'messages': messages,  # Include conversation history
        }
        
        # Add chat-specific metadata if available
        if 'role' in response.get("message", {}):
            payload['role'] = response["message"]["role"]
        
        # Add inference metrics if verbose
        if verbose and not stream:
            payload["inference_metrics"] = self.compute_inference_metrics(response)
        
        return payload

    def chat_stream(
        self
        , messages: List[Dict[str, str]]
        , model: Optional[str] = None
        , verbose: Optional[bool] = False
        , kwargs: Optional[Dict[str, Any]] = {}
    ) -> Generator[str, None, None]:
        """
        Stream chat responses from Ollama's chat endpoint.
        """
        model = model if model else self.default_model
        
        response = self._client.chat(
            model=model,
            messages=messages,
            stream=True,
            options=kwargs
        )
        
        for chunk in response:
            chunk_payload = {
                'response': chunk.get("message", {}).get("content"),
                'model': model,
                'done': chunk.get("done", False)
            }
            
            if verbose and chunk.get("done"):
                chunk_payload["inference_metrics"] = self.compute_inference_metrics(chunk)
            
            yield chunk_payload
        
    def compute_inference_metrics(self, response: Dict[str, Any]) -> Dict[str, Any]:
        
        # Convert timing data from nanoseconds to milliseconds
        ns_to_ms = lambda x: x / 1_000_000

        # Extract relevant metrics from the response
        prompt_eval_duration = ns_to_ms(response['prompt_eval_duration'])  # in ms
        eval_duration = ns_to_ms(response['eval_duration'])  # in ms
        eval_count = response['eval_count']  # number of generated tokens

        # Calculate metrics
        ttft = prompt_eval_duration  # Time to First Token = prompt evaluation duration
        tpot = eval_duration / eval_count if eval_count > 0 else float('inf')  # Time Per Output Token
        latency = ttft + tpot * eval_count  # Latency = TTFT + (TPOT * num_output_tokens)
        throughput = eval_count / eval_duration * 1000 if eval_duration > 0 else 0  # Throughput in tokens per second

        # Generate metrics dictionary
        metrics = {
            'created_at': response['created_at']
            , 'ms_ttft': ttft
            , 'ms_tpot': tpot
            , 'ms_latency': latency
            , 'tks_throughput': throughput
            , 'tokens_generated': eval_count
            , 'prompt_eval_duration': prompt_eval_duration
            , 'eval_duration': eval_duration
            , 'total_tokens': eval_count + response['prompt_eval_count']
        }

        return metrics
    
    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return output
