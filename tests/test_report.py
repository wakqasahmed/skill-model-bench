"""Tests for the evidence-labeled report builder.

No real promptfoo binary, network access, or OPENROUTER_API_KEY is required --
these tests only construct small hand-built dicts shaped like promptfoo's
real ``eval --output results.json`` output (results.results[] with
provider.id, success, cost, latencyMs, testCase.metadata.ground_truth).
"""

import json

from skill_model_bench.report import build_report, render_markdown


def _result(model, success, ground_truth, cost=0.001, latency_ms=100.0):
    return {
        "provider": {"id": model, "label": ""},
        "success": success,
        "score": 1 if success else 0,
        "cost": cost,
        "latencyMs": latency_ms,
        "gradingResult": {"pass": success, "score": 1 if success else 0, "reason": ""},
        "testCase": {
            "vars": {},
            "assert": [],
            "metadata": {"ground_truth": ground_truth, "scenario_id": "s"},
        },
    }


def _promptfoo_output(results):
    return {
        "evalId": "eval-test",
        "results": {"results": results},
        "config": {},
    }


def test_all_fixture_scoring():
    results = [
        _result("openrouter:openai/gpt-4o-mini", True, "fixture"),
        _result("openrouter:openai/gpt-4o-mini", True, "fixture"),
        _result("openrouter:openai/gpt-4o-mini", False, "fixture"),
    ]
    report = build_report(_promptfoo_output(results), quality_bar=0.5)

    stats = report["models"]["openrouter:openai/gpt-4o-mini"]
    assert stats["total_tests"] == 3
    assert stats["passed_tests"] == 2
    assert stats["pass_rate"] == 2 / 3
    assert stats["fixture_count"] == 3
    assert stats["judge_count"] == 0
    assert stats["evidence"] == "fixture_only"
    assert stats["judge_only_warning"] is False


def test_mixed_fixture_and_judge_scoring_for_same_model():
    results = [
        _result("openrouter:openai/gpt-4o-mini", True, "fixture"),
        _result("openrouter:openai/gpt-4o-mini", True, "fixture"),
        _result("openrouter:openai/gpt-4o-mini", True, "judge"),
        _result("openrouter:openai/gpt-4o-mini", False, "judge"),
    ]
    report = build_report(_promptfoo_output(results), quality_bar=0.5)

    stats = report["models"]["openrouter:openai/gpt-4o-mini"]
    assert stats["total_tests"] == 4
    assert stats["fixture_count"] == 2
    assert stats["judge_count"] == 2
    assert stats["fixture_fraction"] == 0.5
    assert stats["judge_fraction"] == 0.5
    assert stats["evidence"] == "mixed"
    assert stats["judge_only_warning"] is False


def test_judge_only_scoring_flags_weaker_evidence():
    results = [
        _result("openrouter:openai/gpt-4o-mini", True, "judge"),
        _result("openrouter:openai/gpt-4o-mini", True, "judge"),
    ]
    report = build_report(_promptfoo_output(results), quality_bar=0.5)

    stats = report["models"]["openrouter:openai/gpt-4o-mini"]
    assert stats["fixture_count"] == 0
    assert stats["judge_count"] == 2
    assert stats["evidence"] == "judge_only"
    assert stats["judge_only_warning"] is True

    markdown = render_markdown(report)
    assert "judge-only" in markdown.lower()
    assert "weaker evidence" in markdown.lower()


def test_no_model_clears_the_bar():
    results = [
        _result("openrouter:openai/gpt-4o-mini", False, "fixture", cost=0.001),
        _result("openrouter:openai/gpt-4o-mini", True, "fixture", cost=0.001),
        _result("openrouter:anthropic/claude-3.5-haiku", False, "fixture", cost=0.002),
        _result("openrouter:anthropic/claude-3.5-haiku", False, "fixture", cost=0.002),
    ]
    report = build_report(_promptfoo_output(results), quality_bar=0.9)

    recommendation = report["recommendation"]
    assert recommendation["qualifies"] is False
    assert recommendation["model"] is None
    assert "no qualifying config" in recommendation["reason"].lower()

    markdown = render_markdown(report)
    assert "no qualifying config" in markdown.lower()


def test_recommendation_picks_cheapest_model_that_clears_bar():
    results = [
        _result("openrouter:openai/gpt-4o-mini", True, "fixture", cost=0.0005),
        _result("openrouter:openai/gpt-4o-mini", True, "fixture", cost=0.0005),
        _result("openrouter:anthropic/claude-3.5-haiku", True, "fixture", cost=0.002),
        _result("openrouter:anthropic/claude-3.5-haiku", True, "fixture", cost=0.002),
    ]
    report = build_report(_promptfoo_output(results), quality_bar=0.8)

    recommendation = report["recommendation"]
    assert recommendation["qualifies"] is True
    assert recommendation["model"] == "openrouter:openai/gpt-4o-mini"


def test_markdown_output_discloses_fixture_judge_split_not_just_json():
    results = [
        _result("openrouter:openai/gpt-4o-mini", True, "fixture"),
        _result("openrouter:openai/gpt-4o-mini", True, "judge"),
    ]
    report = build_report(_promptfoo_output(results), quality_bar=0.5)
    markdown = render_markdown(report)

    assert "fixture" in markdown.lower()
    assert "judge" in markdown.lower()
    assert "50%" in markdown
    assert "openrouter:openai/gpt-4o-mini" in markdown
    # The Markdown table itself is the disclosure surface -- not just the
    # underlying JSON-serializable dict.
    assert json.dumps(report) not in markdown


def test_build_report_accepts_file_path(tmp_path):
    results = [_result("openrouter:openai/gpt-4o-mini", True, "fixture")]
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(_promptfoo_output(results)))

    report = build_report(results_path, quality_bar=0.5)

    assert report["models"]["openrouter:openai/gpt-4o-mini"]["total_tests"] == 1
