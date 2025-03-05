from dotenv import load_dotenv
import os
from enum import Enum

# Load environment variables from .env file
load_dotenv()

class ModelType(Enum):
    GROQ = "groq"
    OPENAI = "openai"
    CLAUDE = "claude"


class Config:
    # FastAPI settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8002"))

    # Selected model for general use
    # SELECTED_MODEL: ModelType = ModelType(os.getenv("SELECTED_MODEL", ModelType.GROQ.value))
    CLASSIFICATION_SELECTED_MODEL_GROQ = os.getenv("CLASSIFICATION_SELECTED_MODEL_GROQ", ModelType.GROQ.value)
    CLASSIFICATION_SELECTED_MODEL_OPENAI =os.getenv("CLASSIFICATION_SELECTED_MODEL_OPENAI", ModelType.GROQ.value)
    CLASSIFICATION_SELECTED_MODEL_CLIENT = os.getenv("CLASSIFICATION_SELECTED_MODEL_CLIENT",ModelType.GROQ.value)


    GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT = os.getenv("GENERAL_CONVERSATION_SELECTED_MODEL_CLIENT", "groq")
    GENERAL_CONVERSATION_SELECTED_MODEL_OPENAI =os.getenv("GENERAL_CONVERSATION_SELECTED_MODEL_OPENAI", "gpt-4o")
    GENERAL_CONVERSATION_SELECTED_MODEL_GROQ = os.getenv("GENERAL_CONVERSATION_SELECTED_MODEL_GROQ","llama3-70b-8192")
    GENERAL_CONVERSATION_CACHE_THRESHOLD = 1

    PLAN_EXTRACTOR_SELECTED_MODEL_CLIENT = os.getenv("PLAN_EXTRACTOR_SELECTED_MODEL_CLIENT", "groq")
    PLAN_EXTRACTOR_SELECTED_MODEL_GROQ = os.getenv("PLAN_EXTRACTOR_SELECTED_MODEL_GROQ", "llama3-70b-8192")
    PLAN_EXTRACTOR_SELECTED_MODEL_OPENAI = os.getenv("PLAN_EXTRACTOR_SELECTED_MODEL_OPENAI", "gpt-4o")

    CONFIRMATION_SELECTED_MODEL_CLIENT = os.getenv("PLAN_EXTRACTOR_SELECTED_MODEL_CLIENT", "groq")
    CONFIRMATION_SELECTED_MODEL_GROQ = os.getenv("PLAN_EXTRACTOR_SELECTED_MODEL_GROQ", "llama3-70b-8192")
    CONFIRMATION_SELECTED_MODEL_OPENAI = os.getenv("PLAN_EXTRACTOR_SELECTED_MODEL_OPENAI", "gpt-4o")

    MODEL_FINAL_MSG_SELECTED_MODEL_CLIENT = os.getenv("MODEL_FINAL_MSG_SELECTED_MODEL_CLIENT", "groq")
    MODEL_FINAL_MSG_SELECTED_MODEL_GROQ = os.getenv("MODEL_FINAL_MSG_SELECTED_MODEL_GROQ", "llama3-70b-8192")
    MODEL_FINAL_MSG_MODEL_OPENAI = os.getenv("MODEL_FINAL_MSG_MODEL_OPENAI", "gpt-4o")


    MODEL_IMAGE_CLIENT = os.getenv("MODEL_IMAGE_CLIENT", "groq")
    MODEL_IMAGE_GROQ = os.getenv("MODEL_IMAGE_GROQ", "llama3-70b-8192")
    MODEL_IMAGE_OPENAI = os.getenv("MODEL_IMAGE_OPENAI", "gpt-4o")


    # Image processing model
    IMAGE_PROCESSING_MODEL: ModelType = ModelType.OPENAI


    DEFAULT_GENERAL_RESPONSE = "hi would you like to create any  product"
    DEFAULT_FINAL_RESPONSE = "somthing happened please try later"

    # API keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY")

    # Model-specific settings
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    OPENAI_MODEL: str = "gpt-4o"
    CLAUDE_MODEL: str = "claude-2"
    OPENAI_IMAGE_MODEL: str = "gpt-4o-mini"

    @classmethod
    # def get_api_key(cls, model_type: ModelType = None) -> str:
    #     if model_type is None:
    #         model_type = cls.SELECTED_MODEL
    #
    #     if model_type == ModelType.GROQ:
    #         return cls.GROQ_API_KEY
    #     elif model_type == ModelType.OPENAI:
    #         return cls.OPENAI_API_KEY
    #     elif model_type == ModelType.CLAUDE:
    #         return cls.CLAUDE_API_KEY
    #     else:
    #         raise ValueError(f"Invalid model selected: {model_type}")

    @classmethod
    def get_model_name(cls, model_type: ModelType = None) -> str:
        if model_type is None:
            model_type = cls.SELECTED_MODEL

        if model_type == ModelType.GROQ:
            return cls.GROQ_MODEL
        elif model_type == ModelType.OPENAI:
            return cls.OPENAI_MODEL
        elif model_type == ModelType.CLAUDE:
            return cls.CLAUDE_MODEL
        else:
            raise ValueError(f"Invalid model selected: {model_type}")

    @classmethod
    def get_image_processing_model(cls) -> str:
        return cls.OPENAI_IMAGE_MODEL


# Add this to config.py

class FunctionConfig:
    FUNCTION_CONFIGS = {
        "extract_plan": {
            "name": "extract_plan",
            "description": "Extracts product plan details",
            "track_metrics": True,
            "expected_duration": 10.0  # expected duration in seconds
        },

        "make_api_call_general": {
            "name": "make_api_call_general",
            "description": "Makes general API calls",
            "track_metrics": True,
            "expected_duration": 5.5
        },

    }

    # Metric logging configuration
    METRIC_LOG_FILE = "logs/metrics.log"
    METRIC_LOG_FORMAT = "json"  # can be 'json' or 'text'
    METRIC_ALERT_THRESHOLD = 1.5  # Alert if execution time is 1.5x expected duration