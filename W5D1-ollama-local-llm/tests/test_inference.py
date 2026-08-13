from src.utils import load_prompts


def test_load_prompts():
    prompts = load_prompts("data/prompts.txt")

    assert len(prompts) == 5
    assert all(isinstance(prompt, str) for prompt in prompts)
    assert all(prompt.strip() for prompt in prompts)