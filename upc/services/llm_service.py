# llm_service.py

from enum import Enum
from typing import NamedTuple, Optional, Type, Dict, Any, Union
from pydantic import BaseModel
import groq
import openai
import instructor
import json
from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
from upc.logger import setup_logger

logger = setup_logger(__name__)


# Configuration Classes
class ModelType(Enum):
    GROQ = "groq"
    OPENAI = "openai"
    CLAUDE = "claude"


class ModelConfig(NamedTuple):
    """Represents configuration for a specific LLM operation"""
    provider: ModelType
    model_name: str


class LLMConfig:
    """Central configuration for LLM services"""
    # Default configurations
    COMPLETION_CONFIG = ModelConfig(
        provider=ModelType.GROQ,
        model_name="llama-3.3-70b-versatile"
    )

    CLASSIFICATION_CONFIG = ModelConfig(
        provider=ModelType.OPENAI,
        model_name="gpt-4"
    )

    # Load from environment or use defaults
    @classmethod
    def get_completion_config(cls) -> ModelConfig:
        return ModelConfig(
            provider=ModelType(os.getenv("COMPLETION_PROVIDER", cls.COMPLETION_CONFIG.provider.value)),
            model_name=os.getenv("COMPLETION_MODEL", cls.COMPLETION_CONFIG.model_name)
        )

    @classmethod
    def get_classification_config(cls) -> ModelConfig:
        return ModelConfig(
            provider=ModelType(os.getenv("CLASSIFICATION_PROVIDER", cls.CLASSIFICATION_CONFIG.provider.value)),
            model_name=os.getenv("CLASSIFICATION_MODEL", cls.CLASSIFICATION_CONFIG.model_name)
        )

    @staticmethod
    def get_api_key(provider: ModelType) -> str:
        """Get API key for a specific provider"""
        key_mapping = {
            ModelType.GROQ: os.getenv("GROQ_API_KEY"),
            ModelType.OPENAI: os.getenv("OPENAI_API_KEY"),
            ModelType.CLAUDE: os.getenv("CLAUDE_API_KEY")
        }
        if key := key_mapping.get(provider):
            return key
        raise ValueError(f"No API key found for provider: {provider}")


# Response and Provider Classes
class LLMResponse(BaseModel):
    """Standardized response format"""
    content: str
    raw_response: Any
    metadata: Dict[str, Any] = {}


class BaseLLMProvider(ABC):
    """Abstract base class for providers"""

    @abstractmethod
    def generate_completion(
            self,
            prompt: str,
            max_tokens: int = 1000,
            temperature: float = 0.7,
            response_model: Optional[Type[BaseModel]] = None
    ) -> LLMResponse:
        pass


class GroqProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str):
        self.client = groq.Groq(api_key=api_key)
        self.model_name = model_name
        self.instructor_client = instructor.patch(self.client)

    def generate_completion(
            self,
            prompt: str,
            max_tokens: int = 1000,
            temperature: float = 0.7,
            response_model: Optional[Type[BaseModel]] = None
    ) -> LLMResponse:
        try:
            if response_model:
                response = self.instructor_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_model=response_model
                )
                return LLMResponse(
                    content=json.dumps(response.model_dump()),
                    raw_response=response,
                    metadata={"provider": "groq", "model": self.model_name}
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return LLMResponse(
                    content=response.choices[0].message.content,
                    raw_response=response,
                    metadata={"provider": "groq", "model": self.model_name}
                )
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise LLMAPIError(f"Groq API error: {str(e)}")


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.instructor_client = instructor.patch(self.client)

    def generate_completion(
            self,
            prompt: str,
            max_tokens: int = 1000,
            temperature: float = 0.7,
            response_model: Optional[Type[BaseModel]] = None
    ) -> LLMResponse:
        try:
            if response_model:
                response = self.instructor_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_model=response_model
                )
                return LLMResponse(
                    content=json.dumps(response.model_dump()),
                    raw_response=response,
                    metadata={"provider": "openai", "model": self.model_name}
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return LLMResponse(
                    content=response.choices[0].message.content,
                    raw_response=response,
                    metadata={"provider": "openai", "model": self.model_name}
                )
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise LLMAPIError(f"OpenAI API error: {str(e)}")


class LLMAPIError(Exception):
    """Custom exception for LLM API errors"""
    pass


# Main Client Class
class LLMClient:
    """Singleton LLM client for handling all LLM operations"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info("Initializing LLM Client")

        # Get configurations
        completion_config = LLMConfig.get_completion_config()
        classification_config = LLMConfig.get_classification_config()

        # Initialize providers
        provider_map = {
            ModelType.GROQ: GroqProvider,
            ModelType.OPENAI: OpenAIProvider
        }

        # Set up completion provider
        provider_class = provider_map.get(completion_config.provider)
        if not provider_class:
            raise ValueError(f"Unsupported completion provider: {completion_config.provider}")
        self.completion_provider = provider_class(
            api_key=LLMConfig.get_api_key(completion_config.provider),
            model_name=completion_config.model_name
        )

        # Set up classification provider
        provider_class = provider_map.get(classification_config.provider)
        if not provider_class:
            raise ValueError(f"Unsupported classification provider: {classification_config.provider}")
        self.classification_provider = provider_class(
            api_key=LLMConfig.get_api_key(classification_config.provider),
            model_name=classification_config.model_name
        )

        self._initialized = True
        logger.info("LLM Client initialized successfully")

    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion using configured provider"""
        logger.info("Generating completion")
        response = self.completion_provider.generate_completion(prompt, **kwargs)
        return response.content

    def generate_classification(self, prompt: str, response_model: Type[BaseModel], **kwargs) -> str:
        """Generate classification using configured provider"""
        logger.info("Generating classification")
        response = self.classification_provider.generate_completion(
            prompt=prompt,
            response_model=response_model,
            temperature=0.3,
            **kwargs
        )
        return response.content


# Convenience functions that match your existing interface
def make_api_call(prompt: str) -> str:
    """Drop-in replacement for existing make_api_call"""
    client = LLMClient()
    return client.generate_completion(prompt)


def make_classification_call(prompt: str, response_model: Type[BaseModel]) -> str:
    """Drop-in replacement for existing make_classification_call"""
    client = LLMClient()
    return client.generate_classification(prompt, response_model)