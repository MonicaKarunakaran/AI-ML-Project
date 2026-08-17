import pytest

from src.token_utils import (
    calculate_cost,
    calculate_request_cost,
    count_tokens,
    usd_to_inr,
)


def test_count_tokens():
    text = (
        "What is the purpose "
        "of human life?"
    )

    tokens = count_tokens(
        text
    )

    assert tokens > 0


def test_empty_text():
    assert count_tokens(
        ""
    ) == 0


def test_gpt4o_cost():
    cost = calculate_cost(
        input_tokens=1000,
        output_tokens=500,
        model="gpt-4o",
    )

    expected = (
        (1000 / 1_000_000 * 5.00)
        + (500 / 1_000_000 * 15.00)
    )

    assert cost == pytest.approx(
        expected
    )


def test_gpt4o_mini_cost():
    cost = calculate_cost(
        input_tokens=1000,
        output_tokens=500,
        model="gpt-4o-mini",
    )

    expected = (
        (1000 / 1_000_000 * 0.15)
        + (500 / 1_000_000 * 0.60)
    )

    assert cost == pytest.approx(
        expected
    )


def test_local_model_cost():
    cost = calculate_cost(
        input_tokens=1000,
        output_tokens=500,
        model="llama3.2:3b",
    )

    assert cost == 0.0


def test_usd_to_inr():
    result = usd_to_inr(
        1.0,
        88.0,
    )

    assert result == 88.0


def test_request_cost():
    result = calculate_request_cost(
        prompt="Hello",
        response="Hi there",
        model="gpt-4o",
    )

    assert result["input_tokens"] > 0
    assert result["output_tokens"] > 0
    assert result["usd_cost"] >= 0
    assert result["inr_cost"] >= 0