import json

from langchain_core.output_parsers import BaseOutputParser


class JsonListParser(BaseOutputParser[list[str]]):
    """Parse and validate an LLM response containing a JSON list of strings."""

    def parse(self, text: str) -> list[str]:
        try:
            result = json.loads(text)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid JSON. Expected a JSON list of strings."
            ) from exc

        if not isinstance(result, list):
            raise ValueError("Expected a JSON list.")

        if not all(isinstance(item, str) for item in result):
            raise ValueError("All items in the JSON list must be strings.")

        return result