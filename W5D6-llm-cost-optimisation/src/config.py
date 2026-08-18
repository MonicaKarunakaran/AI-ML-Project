import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ModelPricing:
    input_price_per_million: float
    output_price_per_million: float


MODEL_PRICING = {
    "gpt-4o": ModelPricing(
        input_price_per_million=5.00,
        output_price_per_million=15.00,
    ),
    "claude-sonnet-4": ModelPricing(
        input_price_per_million=3.00,
        output_price_per_million=15.00,
    ),
    "gemini-2.0-flash": ModelPricing(
        input_price_per_million=0.10,
        output_price_per_million=0.40,
    ),
    "gpt-4o-mini": ModelPricing(
        input_price_per_million=0.15,
        output_price_per_million=0.60,
    ),
}


# Local Ollama models have no API token cost.
LOCAL_MODELS = {
    "llama3.2:3b",
    "qwen2.5:3b",
}


INR_EXCHANGE_RATE = float(
    os.getenv("INR_EXCHANGE_RATE", "88.0")
)

CACHE_SIMILARITY_THRESHOLD = float(
    os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92")
)

CACHE_TTL = int(
    os.getenv("CACHE_TTL", "3600")
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b",
)

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
)

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379",
)

COST_LOG_FILE = os.getenv(
    "COST_LOG_FILE",
    "llm_cost_log.csv",
)