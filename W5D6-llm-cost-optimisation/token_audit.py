import pandas as pd

from src.token_utils import (
    count_tokens,
    calculate_cost,
    usd_to_inr,
)


# ---------------------------------------------------------
# 5 different prompt sizes
# ---------------------------------------------------------

prompts = {
    "Small": """
What is machine learning?
""",

    "Medium": """
Explain machine learning in simple terms.
Describe supervised learning, unsupervised learning,
and reinforcement learning with one example for each.
""",

    "RAG": """
Based on the Bhagavad Gita document, explain the
importance of performing one's duty. Explain why a person
should perform their prescribed duty without being
attached to the results of their actions.
""",

    "Large RAG": """
Based on the Bhagavad Gita document, explain the following:
the purpose of human life, the importance of performing
one's duty, karma, knowledge, devotion, attachment to
results, spiritual development, and the relationship
between action and consciousness. Give a clear explanation
using the information available in the document.
""",

    "Very Large RAG": """
Using the Bhagavad Gita document as the primary context,
provide a detailed explanation of the purpose of human
life, consciousness, duty, karma, knowledge, devotion,
spiritual development, material attachment, the importance
of performing one's prescribed duties, detachment from the
results of actions, and the relationship between individual
actions and spiritual realization. Explain how these ideas
are connected and provide a structured answer based only
on the supplied context.
""",
}


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_GPT4O = "gpt-4o"
MODEL_GPT4O_MINI = "gpt-4o-mini"

# Estimated output tokens for comparison
ESTIMATED_OUTPUT_TOKENS = 300


# ---------------------------------------------------------
# Run token audit
# ---------------------------------------------------------

results = []

for prompt_name, prompt in prompts.items():

    input_tokens = count_tokens(
        prompt,
        MODEL_GPT4O,
    )

    # GPT-4o
    gpt4o_cost_usd = calculate_cost(
        input_tokens=input_tokens,
        output_tokens=ESTIMATED_OUTPUT_TOKENS,
        model=MODEL_GPT4O,
    )

    gpt4o_cost_inr = usd_to_inr(
        gpt4o_cost_usd
    )

    # GPT-4o mini
    mini_cost_usd = calculate_cost(
        input_tokens=input_tokens,
        output_tokens=ESTIMATED_OUTPUT_TOKENS,
        model=MODEL_GPT4O_MINI,
    )

    mini_cost_inr = usd_to_inr(
        mini_cost_usd
    )

    results.append(
        {
            "Prompt": prompt_name,
            "Input Tokens": input_tokens,
            "Estimated Output Tokens": ESTIMATED_OUTPUT_TOKENS,
            "GPT-4o Cost (USD)": round(
                gpt4o_cost_usd,
                6,
            ),
            "GPT-4o Cost (INR)": round(
                gpt4o_cost_inr,
                4,
            ),
            "GPT-4o Mini Cost (USD)": round(
                mini_cost_usd,
                6,
            ),
            "GPT-4o Mini Cost (INR)": round(
                mini_cost_inr,
                4,
            ),
        }
    )


# ---------------------------------------------------------
# Create DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(results)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("W5D6 — LLM TOKEN COST AUDIT")
print("=" * 70)

print(df.to_string(index=False))


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

df.to_csv(
    "token_cost_audit.csv",
    index=False,
)

print("\n" + "=" * 70)
print("Audit saved to: token_cost_audit.csv")
print("=" * 70)