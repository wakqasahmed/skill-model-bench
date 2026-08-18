"""Tests for the OpenRouter-backed multi-model runner. All HTTP calls are
mocked — no real network access and no OPENROUTER_API_KEY required."""

from unittest.mock import Mock, patch

import pytest
import requests

from skill_model_bench.runner import run_models


def _mock_response(content: str, prompt_tokens: int, completion_tokens: int, cost: float):
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
        },
    }
    return response


@patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
@patch("skill_model_bench.runner.requests.post")
def test_successful_multi_model_call(mock_post):
    mock_post.side_effect = [
        _mock_response("hello from A", 10, 5, 0.0001),
        _mock_response("hello from B", 12, 6, 0.0002),
    ]

    results = run_models("say hello", ["model-a", "model-b"])

    assert len(results) == 2

    assert results[0]["model"] == "model-a"
    assert results[0]["response_text"] == "hello from A"
    assert results[0]["prompt_tokens"] == 10
    assert results[0]["completion_tokens"] == 5
    assert results[0]["cost_usd"] == 0.0001
    assert results[0]["error"] is None
    assert isinstance(results[0]["latency_seconds"], float)

    assert results[1]["model"] == "model-b"
    assert results[1]["response_text"] == "hello from B"
    assert results[1]["prompt_tokens"] == 12
    assert results[1]["completion_tokens"] == 6
    assert results[1]["cost_usd"] == 0.0002
    assert results[1]["error"] is None

    # Never leak the API key into request bodies/headers we didn't intend.
    for call in mock_post.call_args_list:
        assert call.kwargs["headers"]["Authorization"] == "Bearer test-key"


@patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
@patch("skill_model_bench.runner.requests.post")
def test_one_model_errors_others_succeed(mock_post):
    rate_limited = Mock()
    rate_limited.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")

    def side_effect(*args, **kwargs):
        payload = kwargs["json"]
        if payload["model"] == "flaky-model":
            raise requests.ConnectionError("connection reset")
        return _mock_response("all good", 8, 4, 0.00005)

    mock_post.side_effect = side_effect

    results = run_models("do a thing", ["flaky-model", "stable-model"])

    assert len(results) == 2

    flaky = results[0]
    assert flaky["model"] == "flaky-model"
    assert flaky["error"] is not None
    assert flaky["response_text"] is None
    assert flaky["prompt_tokens"] is None
    assert flaky["completion_tokens"] is None
    assert flaky["cost_usd"] is None
    assert flaky["latency_seconds"] is None

    stable = results[1]
    assert stable["model"] == "stable-model"
    assert stable["error"] is None
    assert stable["response_text"] == "all good"
    assert stable["prompt_tokens"] == 8
    assert stable["completion_tokens"] == 4
    assert stable["cost_usd"] == 0.00005


@patch("skill_model_bench.runner.requests.post")
def test_missing_api_key_raises_before_any_http_call(mock_post, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        run_models("say hello", ["model-a"])

    mock_post.assert_not_called()
