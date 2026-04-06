"""LLM Client – Ollama API wrapper for local LLM inference."""

import json
from typing import Generator, Optional

import requests

from pragya.utils import load_config


class OllamaClient:
    """Client for interacting with a local Ollama instance."""
    
    def __init__(self, model: str = None, base_url: str = None):
        config = load_config()
        self.model = model or config["llm"]["model"]
        self.base_url = base_url or config["llm"]["base_url"]
        self.temperature = config["llm"]["temperature"]
        self.max_tokens = config["llm"]["max_tokens"]
        self.context_window = config["llm"]["context_window"]
    
    def _build_url(self, endpoint: str) -> str:
        return f"{self.base_url}{endpoint}"
    
    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = requests.get(self._build_url("/api/tags"), timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Check if our model is available (exact or partial match)
                model_base = self.model.split(":")[0]
                return any(model_base in name for name in model_names)
            return False
        except requests.ConnectionError:
            return False
    
    def list_models(self) -> list[str]:
        """List all available models in Ollama."""
        try:
            resp = requests.get(self._build_url("/api/tags"), timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return [m.get("name", "") for m in models]
            return []
        except requests.ConnectionError:
            return []
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> Generator[str, None, None]:
        """Generate a streaming response from the LLM.
        
        Args:
            prompt: User prompt text
            system_prompt: System prompt for context/instructions
            
        Yields:
            Response tokens as they arrive
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.context_window,
            },
        }
        
        try:
            resp = requests.post(
                self._build_url("/api/generate"),
                json=payload,
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
            
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
                        
        except requests.ConnectionError:
            yield "❌ Error: Cannot connect to Ollama. Make sure Ollama is running (ollama serve)."
        except requests.Timeout:
            yield "❌ Error: Ollama request timed out. The model may be loading."
        except Exception as e:
            yield f"❌ Error: {str(e)}"
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> str:
        """Generate a complete (non-streaming) response from the LLM.
        
        Args:
            prompt: User prompt text
            system_prompt: System prompt for context/instructions
            
        Returns:
            Complete response text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.context_window,
            },
        }
        
        try:
            resp = requests.post(
                self._build_url("/api/generate"),
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
            
        except requests.ConnectionError:
            return "❌ Error: Cannot connect to Ollama. Make sure Ollama is running."
        except Exception as e:
            return f"❌ Error: {str(e)}"
