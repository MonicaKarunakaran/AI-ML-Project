from typing import Optional

import tiktoken

from src.config import (
    INR_EXCHANGE_RATE,
    LOCAL_MODELS,
    MODEL_PRICING,
)


def get_encoder(model: str = "gpt-4o"):
    """
    Return the tokenizer for the requested model.

    If the model is not directly supported by tiktoken,
    GPT-compatible o200k_base encoding is used.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_tokens(
    text: str,
    model: str = "gpt-4o",
) -> int:
    """
    Count the number of tokens in a text string.
    """
    if not text:
        return 0

    encoder = get_encoder(model)

    return len(
        encoder.encode(text)
    )


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> float:
    """
    Calculate estimated API cost in USD.

    Local Ollama models have zero API cost.
    """
    if model in LOCAL_MODELS:
        return 0.0

    if model not in MODEL_PRICING:
        raise ValueError(
            f"Unknown model: {model}"
        )

    pricing = MODEL_PRICING[model]

    input_cost = (
        input_tokens
        / 1_000_000
        * pricing.input_price_per_million
    )

    output_cost = (
        output_tokens
        / 1_000_000
        * pricing.output_price_per_million
    )

    return input_cost + output_cost


def usd_to_inr(
    usd_cost: float,
    exchange_rate: Optional[float] = None,
) -> float:
    """
    Convert USD to INR.
    """
    rate = (
        exchange_rate
        if exchange_rate is not None
        else INR_EXCHANGE_RATE
    )

    return usd_cost * rate


def calculate_request_cost(
    prompt: str,
    response: str,
    model: str = "gpt-4o",
) -> dict:
    """
    Count input/output tokens and calculate cost.
    """
    input_tokens = count_tokens(
        prompt,
        model,
    )

    output_tokens = count_tokens(
        response,
        model,
    )

    usd_cost = calculate_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
    )

    inr_cost = usd_to_inr(
        usd_cost
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd_cost": usd_cost,
        "inr_cost": inr_cost,
    }