import json


def parse_response(response: str) -> dict:
    """
    Parse the LLM response as JSON.

    Expected format:
    {
        "answer": "...",
        "sources": []
    }
    """

    if isinstance(response, dict):
        return response

    response = response.strip()

    # Remove markdown JSON fences if the model adds them
    if response.startswith("```json"):
        response = response[7:]

    if response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    response = response.strip()

    try:
        result = json.loads(response)

        if isinstance(result, dict):
            return {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
            }

    except json.JSONDecodeError:
        pass

    # Fallback if Ollama returns normal text instead of JSON
    return {
        "answer": response,
        "sources": [],
    }