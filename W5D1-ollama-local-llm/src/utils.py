from pathlib import Path


def load_prompts(file_path: str) -> list[str]:
    """Load prompts from a text file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {file_path}"
        )

    with path.open("r", encoding="utf-8") as file:
        prompts = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return prompts


def save_results(
    results: list[dict],
    output_file: str,
) -> None:
    """Save inference results as Markdown."""

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:

        file.write("# Ollama Inference Results\n\n")

        for result in results:
            file.write(f"## Prompt {result['prompt_number']}\n\n")
            file.write(
                f"**Question:** {result['prompt']}\n\n"
            )
            file.write(
                f"**Model:** {result['model']}\n\n"
            )
            file.write(
                f"**Response:**\n\n{result['response']}\n\n"
            )
            file.write("---\n\n")