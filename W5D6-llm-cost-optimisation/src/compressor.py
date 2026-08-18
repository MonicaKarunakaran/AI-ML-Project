from typing import Optional


class PromptCompressor:
    """
    Wrapper around LLMLingua.

    Uses CPU explicitly because the internship environment
    does not have a CUDA-enabled PyTorch installation.
    """

    def __init__(self):
        try:
            from llmlingua import PromptCompressor as LLMLinguaCompressor
        except ImportError as exc:
            raise ImportError(
                "LLMLingua is not installed.\n"
                "Install it using:\n"
                "python -m pip install llmlingua"
            ) from exc

        self.compressor = LLMLinguaCompressor(
            device_map="cpu"
        )

    def compress(
        self,
        text: str,
        rate: float = 0.4,
    ) -> str:
        """
        Compress a prompt using LLMLingua.

        rate=0.4 means the compressor targets
        approximately 40% of the original token budget.
        """

        result = self.compressor.compress_prompt(
            text,
            rate=rate,
        )

        if isinstance(result, dict):
            return result.get(
                "compressed_prompt",
                text,
            )

        return str(result)


def compress_prompt(
    text: str,
    rate: float = 0.4,
) -> str:
    """
    Convenience function used by the project.
    """

    compressor = PromptCompressor()

    return compressor.compress(
        text=text,
        rate=rate,
    )