import os
import time
import json
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import openai
import anthropic
import google.generativeai as genai


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self, api_key: str, model_name: str, **kwargs):
        self.api_key = api_key
        self.model_name = model_name
        self.kwargs = kwargs
    
    @abstractmethod
    def generate(self, prompt: str, **generation_kwargs) -> LLMResponse:
        """Generate text from the given prompt."""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **generation_kwargs) -> LLMResponse:
        """Generate response from chat messages."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider (GPT-4, GPT-3.5, etc.)."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, **generation_kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **generation_kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **generation_kwargs) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=generation_kwargs.get('temperature', 0.7),
                max_tokens=generation_kwargs.get('max_tokens', 1024),
                **{k: v for k, v in generation_kwargs.items() 
                   if k not in ['temperature', 'max_tokens']}
            )
            
            return LLMResponse(
                text=response.choices[0].message.content,
                usage=dict(response.usage) if response.usage else None,
                model=response.model,
                finish_reason=response.choices[0].finish_reason
            )
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-haiku-20240307", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(self, prompt: str, **generation_kwargs) -> LLMResponse:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=generation_kwargs.get('max_tokens', 1024),
                temperature=generation_kwargs.get('temperature', 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            
            return LLMResponse(
                text=response.content[0].text,
                usage={"input_tokens": response.usage.input_tokens, 
                      "output_tokens": response.usage.output_tokens},
                model=self.model_name,
                finish_reason=response.stop_reason
            )
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def chat(self, messages: List[Dict[str, str]], **generation_kwargs) -> LLMResponse:
        # Convert to Anthropic format (no system role)
        anthropic_messages = []
        for msg in messages:
            if msg["role"] != "system":
                anthropic_messages.append(msg)
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=generation_kwargs.get('max_tokens', 1024),
                temperature=generation_kwargs.get('temperature', 0.7),
                messages=anthropic_messages
            )
            
            return LLMResponse(
                text=response.content[0].text,
                usage={"input_tokens": response.usage.input_tokens, 
                      "output_tokens": response.usage.output_tokens},
                model=self.model_name,
                finish_reason=response.stop_reason
            )
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")


class GoogleProvider(BaseLLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate(self, prompt: str, **generation_kwargs) -> LLMResponse:
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=generation_kwargs.get('temperature', 0.7),
                max_output_tokens=generation_kwargs.get('max_tokens', 1024)
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return LLMResponse(
                text=response.text,
                model=self.model_name,
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else None
            )
        except Exception as e:
            raise Exception(f"Google API error: {str(e)}")
    
    def chat(self, messages: List[Dict[str, str]], **generation_kwargs) -> LLMResponse:
        # Convert to Google format
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return self.generate(prompt, **generation_kwargs)


class NebiusProvider(BaseLLMProvider):
    """Nebius AI Studio provider."""
    
    def __init__(self, api_key: str, model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.base_url = "https://api.studio.nebius.ai/v1"
    
    def generate(self, prompt: str, **generation_kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **generation_kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **generation_kwargs) -> LLMResponse:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": generation_kwargs.get('temperature', 0.7),
                "max_tokens": generation_kwargs.get('max_tokens', 1024)
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            return LLMResponse(
                text=result["choices"][0]["message"]["content"],
                usage=result.get("usage"),
                model=result.get("model"),
                finish_reason=result["choices"][0].get("finish_reason")
            )
        except Exception as e:
            raise Exception(f"Nebius API error: {str(e)}")


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API provider."""
    
    def __init__(self, api_key: str, model_name: str = "deepseek-chat", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.base_url = "https://api.deepseek.com/v1"
    
    def generate(self, prompt: str, **generation_kwargs) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **generation_kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **generation_kwargs) -> LLMResponse:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": generation_kwargs.get('temperature', 0.7),
                "max_tokens": generation_kwargs.get('max_tokens', 1024)
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            return LLMResponse(
                text=result["choices"][0]["message"]["content"],
                usage=result.get("usage"),
                model=result.get("model"),
                finish_reason=result["choices"][0].get("finish_reason")
            )
        except Exception as e:
            raise Exception(f"DeepSeek API error: {str(e)}")


class LocalTransformersProvider(BaseLLMProvider):
    """Local Transformers model provider for backward compatibility."""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        super().__init__("", model_name, **kwargs)
        self.device = device
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        device = self.device if self.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
    
    def generate(self, prompt: str, **generation_kwargs) -> LLMResponse:
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=generation_kwargs.get('max_tokens', 100),
                temperature=generation_kwargs.get('temperature', 0.7),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = generated_text[len(prompt):].strip()
        
        return LLMResponse(
            text=response_text,
            model=self.model_name
        )
    
    def chat(self, messages: List[Dict[str, str]], **generation_kwargs) -> LLMResponse:
        # Convert chat to single prompt
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return self.generate(prompt, **generation_kwargs)


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_type: str, api_key: str = None, model_name: str = None, **kwargs) -> BaseLLMProvider:
        """Create an LLM provider based on type."""
        
        if provider_type.lower() == "openai":
            return OpenAIProvider(api_key, model_name or "gpt-4o-mini", **kwargs)
        elif provider_type.lower() == "anthropic":
            return AnthropicProvider(api_key, model_name or "claude-3-haiku-20240307", **kwargs)
        elif provider_type.lower() == "google":
            return GoogleProvider(api_key, model_name or "gemini-1.5-flash", **kwargs)
        elif provider_type.lower() == "nebius":
            return NebiusProvider(api_key, model_name or "meta-llama/Meta-Llama-3.1-70B-Instruct", **kwargs)
        elif provider_type.lower() == "deepseek":
            return DeepSeekProvider(api_key, model_name or "deepseek-chat", **kwargs)
        elif provider_type.lower() == "local":
            return LocalTransformersProvider(model_name or "microsoft/DialoGPT-medium", **kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")


class RateLimitedLLMProvider:
    """Wrapper that adds rate limiting to any LLM provider."""
    
    def __init__(self, provider: BaseLLMProvider, requests_per_minute: int = 60):
        self.provider = provider
        self.requests_per_minute = requests_per_minute
        self.request_times = []
    
    def _check_rate_limit(self):
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.request_times.append(now)
    
    def generate(self, prompt: str, **generation_kwargs) -> LLMResponse:
        self._check_rate_limit()
        return self.provider.generate(prompt, **generation_kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **generation_kwargs) -> LLMResponse:
        self._check_rate_limit()
        return self.provider.chat(messages, **generation_kwargs) 