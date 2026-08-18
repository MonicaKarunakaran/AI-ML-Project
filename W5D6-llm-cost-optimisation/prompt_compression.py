import pandas as pd

from src.token_utils import (
    count_tokens,
    calculate_cost,
    usd_to_inr,
)
from src.compressor import compress_prompt


# ---------------------------------------------------------
# Create approximately 2,000-token RAG context
# ---------------------------------------------------------

base_context = """
The Bhagavad Gita explains the nature of human life,
consciousness, duty, action, knowledge, devotion and
spiritual realization.

A person should perform their prescribed duties carefully
and responsibly. The teaching emphasizes that a person
should not become excessively attached to the results of
their actions. Instead, one should focus on performing
the appropriate duty with sincerity and discipline.

Action is an important part of human life. People cannot
remain completely inactive because material nature and
ordinary responsibilities require continuous activity.
Therefore, the important principle is to perform action
while maintaining the correct consciousness.

The Bhagavad Gita also explains karma. Karma refers to
action and its consequences. Actions performed with
attachment can create further bondage, while actions
performed with the proper understanding and without
selfish attachment can support spiritual development.

Knowledge is another important concept. Understanding
the difference between temporary material conditions
and the deeper nature of consciousness helps a person
develop wisdom.

Devotion is also emphasized. Devotional consciousness
allows a person to connect their activities with a higher
spiritual purpose rather than simply pursuing personal
material rewards.

The purpose of human life is therefore presented as more
than material achievement. Human beings have the ability
to develop knowledge, discipline, consciousness and
devotion.

A person should understand their responsibilities and
perform their duties according to their situation.
Avoiding action simply because the result is uncertain
is not considered a productive solution.

The document repeatedly connects duty with discipline.
Performing one's duty sincerely can help develop a stable
mind and reduce selfish attachment.

The results of action are not always completely under a
person's control. A person can control their effort,
intention and attitude, but external circumstances may
affect the final result.

Therefore, focusing entirely on outcomes can produce
anxiety, disappointment and attachment. A better approach
is to focus on performing the duty properly.

The teachings also distinguish between selfish action
and action performed with a higher purpose. Selfish action
is primarily concerned with personal reward, while
selfless action can contribute to spiritual progress.

Human consciousness is influenced by thoughts, desires
and actions. By controlling these influences and
developing knowledge, a person can gradually improve
their understanding.

The relationship between knowledge and action is also
important. Knowledge helps a person understand why an
action should be performed, while disciplined action
allows that knowledge to be applied in practical life.

The document presents spiritual development as a gradual
process. It involves understanding one's responsibilities,
controlling attachment, developing knowledge and directing
one's actions toward a higher purpose.

The central idea is that duty should not be abandoned
because of fear, uncertainty or attachment to results.
Instead, a person should perform their responsibilities
with sincerity and maintain appropriate awareness.

This approach encourages discipline and reduces excessive
concern about success and failure.

A person who performs actions with a balanced mind can
develop greater stability. Such stability can support
knowledge and spiritual understanding.

The document also discusses the importance of controlling
desire. Excessive desire can lead to attachment, anger,
confusion and poor judgment.

When a person becomes attached to a particular result,
their emotional state can depend heavily on whether that
result occurs.

By reducing attachment and focusing on duty, a person can
develop greater mental stability.

The teachings therefore provide a framework for combining
ordinary responsibilities with spiritual development.

The important concepts of duty, action, knowledge,
consciousness and devotion are connected rather than
being completely separate ideas.

A person can perform ordinary activities while still
developing spiritually if those activities are performed
with appropriate intention and awareness.

The document encourages people to understand their role,
perform their responsibilities and avoid unnecessary
attachment.

The final goal is a state in which action, knowledge and
consciousness work together toward spiritual realization.
"""


# Repeat context to create a realistic long RAG prompt.
long_context = base_context * 6

question = """
Question:
According to the document, what is the purpose of human
life, why should a person perform their prescribed duty,
and how does performing one's duty without attachment to
results contribute to spiritual development?

Answer using only the supplied context.
"""


original_prompt = (
    "Context:\n"
    + long_context
    + "\n\n"
    + question
)


# ---------------------------------------------------------
# Count original tokens
# ---------------------------------------------------------

original_tokens = count_tokens(
    original_prompt,
    "gpt-4o",
)


# ---------------------------------------------------------
# Compress using LLMLingua
# ---------------------------------------------------------

print("\nCompressing prompt using LLMLingua...")
print("Compression rate: 0.4")

compressed_prompt = compress_prompt(
    original_prompt,
    rate=0.4,
)


compressed_tokens = count_tokens(
    compressed_prompt,
    "gpt-4o",
)


# ---------------------------------------------------------
# Calculate reduction
# ---------------------------------------------------------

tokens_saved = (
    original_tokens
    - compressed_tokens
)

if original_tokens > 0:
    reduction_percent = (
        tokens_saved
        / original_tokens
        * 100
    )
else:
    reduction_percent = 0


# ---------------------------------------------------------
# Cost comparison
# ---------------------------------------------------------

estimated_output_tokens = 300

baseline_cost = calculate_cost(
    original_tokens,
    estimated_output_tokens,
    "gpt-4o",
)

optimized_cost = calculate_cost(
    compressed_tokens,
    estimated_output_tokens,
    "gpt-4o",
)

cost_saved = (
    baseline_cost
    - optimized_cost
)

cost_reduction_percent = (
    cost_saved
    / baseline_cost
    * 100
    if baseline_cost > 0
    else 0
)


# ---------------------------------------------------------
# Quality comparison
# ---------------------------------------------------------

quality_note = (
    "Manual quality check required. "
    "The compressed prompt should preserve "
    "the main context and answer requirements."
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("W5D6 — LLMLINGUA COMPRESSION RESULTS")
print("=" * 70)

print(
    f"Original tokens   : {original_tokens}"
)

print(
    f"Compressed tokens : {compressed_tokens}"
)

print(
    f"Tokens saved      : {tokens_saved}"
)

print(
    f"Token reduction   : {reduction_percent:.2f}%"
)

print(
    f"\nBaseline cost     : ${baseline_cost:.6f}"
)

print(
    f"Optimised cost    : ${optimized_cost:.6f}"
)

print(
    f"Cost saved        : ${cost_saved:.6f}"
)

print(
    f"Cost reduction    : {cost_reduction_percent:.2f}%"
)

print(
    f"\nBaseline INR      : ₹{usd_to_inr(baseline_cost):.4f}"
)

print(
    f"Optimised INR     : ₹{usd_to_inr(optimized_cost):.4f}"
)

print("\nQuality check:")
print(quality_note)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results = pd.DataFrame(
    [
        {
            "Original Tokens": original_tokens,
            "Compressed Tokens": compressed_tokens,
            "Tokens Saved": tokens_saved,
            "Token Reduction %": round(
                reduction_percent,
                2,
            ),
            "Baseline Cost USD": round(
                baseline_cost,
                6,
            ),
            "Optimised Cost USD": round(
                optimized_cost,
                6,
            ),
            "Cost Saved USD": round(
                cost_saved,
                6,
            ),
            "Cost Reduction %": round(
                cost_reduction_percent,
                2,
            ),
            "Baseline Cost INR": round(
                usd_to_inr(baseline_cost),
                4,
            ),
            "Optimised Cost INR": round(
                usd_to_inr(optimized_cost),
                4,
            ),
            "Quality Check": quality_note,
        }
    ]
)

results.to_csv(
    "compression_results.csv",
    index=False,
)

print(
    "\nResults saved to: compression_results.csv"
)