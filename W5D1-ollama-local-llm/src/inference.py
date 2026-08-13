from pathlib import Path

from src.ollama_client import chat_with_ollama
from src.utils import load_prompts, save_results


MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """
You are a helpful AI and Machine Learning mentor.

Explain technical concepts clearly and simply.
Use beginner-friendly language and short examples.
When code is requested, provide clean Python examples.
Avoid unnecessary complexity.
"""


def run_inference() -> list[dict]:
    """Run the five prompts through the local LLM."""

    prompt_file = Path("data/prompts.txt")

    prompts = load_prompts(str(prompt_file))

    results = []

    for index, prompt in enumerate(prompts, start=1):

        print("\n" + "=" * 70)
        print(f"PROMPT {index}")
        print("=" * 70)
        print(prompt)

        response = chat_with_ollama(
            prompt=prompt,
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
        )

        print("\nRESPONSE:")
        print(response)

        results.append(
            {
                "prompt_number": index,
                "prompt": prompt,
                "model": MODEL,
                "response": response,
            }
        )

    return results


if __name__ == "__main__":
    results = run_inference()

    save_results(
        results,
        "outputs/llama3.2_results.md",
    )

    print("\nResults saved successfully.")