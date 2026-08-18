from src.dashboard import log_request
from src.token_utils import calculate_cost, usd_to_inr


requests = [
    {
        "query": "What is machine learning?",
        "model": "gpt-4o-mini",
        "input_tokens": 20,
        "output_tokens": 80,
        "cache_hit": False,
        "compression_used": False,
    },
    {
        "query": "Explain supervised learning.",
        "model": "gpt-4o-mini",
        "input_tokens": 35,
        "output_tokens": 120,
        "cache_hit": False,
        "compression_used": False,
    },
    {
        "query": "What is the purpose of human life?",
        "model": "gpt-4o",
        "input_tokens": 1500,
        "output_tokens": 250,
        "cache_hit": False,
        "compression_used": False,
    },
    {
        "query": "What is the purpose of human existence?",
        "model": "gpt-4o",
        "input_tokens": 1500,
        "output_tokens": 250,
        "cache_hit": True,
        "compression_used": False,
    },
    {
        "query": "Explain duty and karma.",
        "model": "gpt-4o",
        "input_tokens": 900,
        "output_tokens": 200,
        "cache_hit": False,
        "compression_used": True,
    },
    {
        "query": "Explain spiritual development.",
        "model": "gpt-4o-mini",
        "input_tokens": 500,
        "output_tokens": 150,
        "cache_hit": False,
        "compression_used": True,
    },
]


for item in requests:

    cost_usd = calculate_cost(
        input_tokens=item["input_tokens"],
        output_tokens=item["output_tokens"],
        model=item["model"],
    )

    cost_inr = usd_to_inr(
        cost_usd
    )

    log_request(
        query=item["query"],
        model=item["model"],
        input_tokens=item["input_tokens"],
        output_tokens=item["output_tokens"],
        cache_hit=item["cache_hit"],
        compression_used=item["compression_used"],
        cost_usd=cost_usd,
        cost_inr=cost_inr,
    )


print(
    "Cost log generated successfully."
)

print(
    "File: llm_cost_log.csv"
)