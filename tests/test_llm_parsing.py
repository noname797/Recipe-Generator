"""
Unit tests for utils.llm_parsing.parse_llm_yaml_with_retry, using fake LLM
clients/parsers so no real network/AWS calls are made.
"""
from unittest.mock import MagicMock

import pytest

from utils.llm_parsing import parse_llm_yaml_with_retry


class _FakeLLMClient:
    """Minimal stand-in for BedrockClient: has .invoke() and .llm."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.llm = MagicMock()  # underlying chat model, used by OutputFixingParser

    def invoke(self, prompt):
        return self._responses.pop(0)


def test_parse_succeeds_on_first_try(monkeypatch):
    parser = MagicMock()
    parser.parse.return_value = {"ok": True}

    client = _FakeLLMClient(["some yaml"])
    result = parse_llm_yaml_with_retry(client, parser, "prompt", "TestAgent")

    assert result == {"ok": True}
    parser.parse.assert_called_once_with("some yaml")


def test_parse_recovers_via_output_fixing_parser(monkeypatch):
    parser = MagicMock()
    parser.parse.side_effect = ValueError("bad yaml")

    fixing_parser_instance = MagicMock()
    fixing_parser_instance.parse.return_value = {"fixed": True}
    monkeypatch.setattr(
        "utils.llm_parsing.OutputFixingParser.from_llm",
        lambda parser, llm: fixing_parser_instance,
    )

    client = _FakeLLMClient(["malformed yaml"])
    result = parse_llm_yaml_with_retry(client, parser, "prompt", "TestAgent")

    assert result == {"fixed": True}


def test_parse_retries_llm_invocation_then_succeeds(monkeypatch):
    parser = MagicMock()
    # First attempt: direct parse fails.
    # Second attempt: direct parse succeeds.
    parser.parse.side_effect = [ValueError("bad yaml"), {"ok": True}]

    fixing_parser_instance = MagicMock()
    fixing_parser_instance.parse.side_effect = ValueError("still bad")
    monkeypatch.setattr(
        "utils.llm_parsing.OutputFixingParser.from_llm",
        lambda parser, llm: fixing_parser_instance,
    )

    client = _FakeLLMClient(["malformed yaml", "good yaml"])
    result = parse_llm_yaml_with_retry(
        client, parser, "prompt", "TestAgent", max_retries=1
    )

    assert result == {"ok": True}
    assert client._responses == []  # both queued responses were consumed


def test_parse_raises_after_exhausting_retries(monkeypatch):
    parser = MagicMock()
    parser.parse.side_effect = ValueError("always bad")

    fixing_parser_instance = MagicMock()
    fixing_parser_instance.parse.side_effect = ValueError("always bad too")
    monkeypatch.setattr(
        "utils.llm_parsing.OutputFixingParser.from_llm",
        lambda parser, llm: fixing_parser_instance,
    )

    client = _FakeLLMClient(["r1", "r2"])
    with pytest.raises(RuntimeError, match="TestAgent"):
        parse_llm_yaml_with_retry(client, parser, "prompt", "TestAgent", max_retries=1)
