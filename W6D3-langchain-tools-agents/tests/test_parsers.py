import pytest

from src.parsers.json_list_parser import JsonListParser


def test_parse_valid_json_list():
    parser = JsonListParser()

    result = parser.parse('["LangChain", "Agents", "Tools"]')

    assert result == ["LangChain", "Agents", "Tools"]


def test_parse_empty_json_list():
    parser = JsonListParser()

    result = parser.parse("[]")

    assert result == []


def test_reject_invalid_json():
    parser = JsonListParser()

    with pytest.raises(ValueError):
        parser.parse("not valid json")


def test_reject_non_list_json():
    parser = JsonListParser()

    with pytest.raises(ValueError):
        parser.parse('{"topic": "LangChain"}')


def test_reject_non_string_items():
    parser = JsonListParser()

    with pytest.raises(ValueError):
        parser.parse('["LangChain", 123]')
        