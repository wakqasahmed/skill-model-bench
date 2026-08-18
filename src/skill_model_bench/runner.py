"""OpenRouter-backed multi-model runner.

Calls a list of OpenRouter models with the same prompt and returns a
per-model result: response text, token usage, cost, latency, and any error.

Cost reporting: OpenRouter's chat completions endpoint is OpenAI-compatible.
This module requests `"usage": {"include": true}` in the request body, which
per OpenRouter's documented "usage accounting" extension causes the response's
`usage` object to include a `cost` field (USD) alongside `prompt_tokens` /
`completion_tokens` — no separate call to `/api/v1/generation` is needed.
This shape is inferred from OpenRouter's public docs, not confirmed against a
live response in this environment (no network access here) — if `usage.cost`
is absent from a real response, `cost_usd` simply falls back to `None`.
"""

import os
import time

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 60


def _empty_result(model: str, error: str) -> dict:
    return {
        "model": model,
        "response_text": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cost_usd": None,
        "latency_seconds": None,
        "error": error,
    }


def _call_model(model: str, prompt: str, api_key: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "usage": {"include": True},
    }

    start = time.monotonic()
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        latency = time.monotonic() - start
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return _empty_result(model, str(exc))

    try:
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
    except (KeyError, IndexError, TypeError) as exc:
        return _empty_result(model, f"unexpected response shape: {exc}")

    return {
        "model": model,
        "response_text": choice,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost_usd": usage.get("cost"),
        "latency_seconds": latency,
        "error": None,
    }


def run_models(prompt: str, models: list[str]) -> list[dict]:
    """Call each model in `models` with `prompt` via OpenRouter.

    Reads OPENROUTER_API_KEY from the environment only. Raises RuntimeError
    immediately if it is unset. A single model's HTTP failure is recorded in
    that model's result entry (`error` set) and does not abort the batch.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Export it before running the benchmark."
        )

    return [_call_model(model, prompt, api_key) for model in models]
