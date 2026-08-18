"""Tests for the fixture-to-promptfoo config generator.

No real promptfoo binary, network access, or OPENROUTER_API_KEY is required —
these tests only construct skill directories on disk and assert on the
generated config's structure after round-tripping it through YAML.
"""

import json

import pytest
import yaml

from skill_model_bench.assertions.fixture_check import get_assert
from skill_model_bench.config_gen import DEFAULT_JUDGE_PROVIDER, generate_config, write_config

SKILL_MD = "# Example Skill\n\nSome rules the model must follow.\n"

FIXTURES = [
    {
        "id": "follow-01",
        "scenario": "The response cites its evidence directly.",
        "expected": "follow",
    },
    {
        "id": "violates-01",
        "scenario": "The response fabricates a score with no evidence.",
        "expected": "violates",
        "violates_gate": "2. No-fabricated-scores gate",
    },
]


def _make_skill_dir(tmp_path, with_fixtures: bool):
    skill_dir = tmp_path / "example-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(SKILL_MD)
    if with_fixtures:
        fixtures_dir = skill_dir / "eval" / "fixtures"
        fixtures_dir.mkdir(parents=True)
        (fixtures_dir / "held-out-scenarios.json").write_text(json.dumps(FIXTURES))
    return skill_dir


def test_generate_config_with_fixtures_uses_python_assertions(tmp_path):
    skill_dir = _make_skill_dir(tmp_path, with_fixtures=True)
    models = ["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku"]

    config = generate_config(skill_dir, models)

    assert config["providers"] == [
        "openrouter:openai/gpt-4o-mini",
        "openrouter:anthropic/claude-3.5-haiku",
    ]
    assert len(config["prompts"]) == 1
    assert "{{scenario}}" in config["prompts"][0]
    assert SKILL_MD.strip() in config["prompts"][0]

    tests = config["tests"]
    assert len(tests) == len(FIXTURES)

    for test_case, fixture in zip(tests, FIXTURES):
        assert test_case["vars"]["scenario"] == fixture["scenario"]
        assert test_case["vars"]["expected"] == fixture["expected"]
        if "violates_gate" in fixture:
            assert test_case["vars"]["violates_gate"] == fixture["violates_gate"]

        assert test_case["metadata"]["ground_truth"] == "fixture"

        asserts = test_case["assert"]
        assert len(asserts) == 1
        assert asserts[0]["type"] == "python"
        assert asserts[0]["value"].startswith("file://")
        assert asserts[0]["value"].endswith("fixture_check.py")


def test_generate_config_yaml_round_trip_with_fixtures(tmp_path):
    skill_dir = _make_skill_dir(tmp_path, with_fixtures=True)
    output_path = tmp_path / "promptfoo-config.yaml"

    write_config(skill_dir, ["openai/gpt-4o-mini"], output_path)

    loaded = yaml.safe_load(output_path.read_text())
    assert loaded["providers"] == ["openrouter:openai/gpt-4o-mini"]
    assert len(loaded["tests"]) == len(FIXTURES)
    for test_case in loaded["tests"]:
        assert test_case["metadata"]["ground_truth"] == "fixture"


def test_generate_config_without_fixtures_falls_back_to_llm_rubric(tmp_path):
    skill_dir = _make_skill_dir(tmp_path, with_fixtures=False)

    config = generate_config(skill_dir, ["openai/gpt-4o-mini"])

    tests = config["tests"]
    assert len(tests) == 1
    test_case = tests[0]

    assert test_case["metadata"]["ground_truth"] == "judge"

    asserts = test_case["assert"]
    assert len(asserts) == 1
    assert asserts[0]["type"] == "llm-rubric"
    assert isinstance(asserts[0]["value"], str) and asserts[0]["value"]
    assert asserts[0]["provider"] == DEFAULT_JUDGE_PROVIDER


def test_generate_config_judge_provider_is_configurable(tmp_path):
    skill_dir = _make_skill_dir(tmp_path, with_fixtures=False)

    config = generate_config(
        skill_dir,
        ["openai/gpt-4o-mini"],
        judge_provider="openrouter:openai/gpt-4o",
    )

    assert config["tests"][0]["assert"][0]["provider"] == "openrouter:openai/gpt-4o"


def test_generate_config_raises_without_skill_md(tmp_path):
    skill_dir = tmp_path / "no-skill-md"
    skill_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        generate_config(skill_dir, ["openai/gpt-4o-mini"])


def test_generate_config_empty_fixtures_file_falls_back_to_judge(tmp_path):
    skill_dir = _make_skill_dir(tmp_path, with_fixtures=False)
    fixtures_dir = skill_dir / "eval" / "fixtures"
    fixtures_dir.mkdir(parents=True)
    (fixtures_dir / "held-out-scenarios.json").write_text("[]")

    config = generate_config(skill_dir, ["openai/gpt-4o-mini"])

    assert config["tests"][0]["metadata"]["ground_truth"] == "judge"


class _DictContext(dict):
    """Simulate promptfoo's context object via dict-style access."""


def test_fixture_check_passes_for_follow_scenario_without_violation_language():
    context = _DictContext(vars={"expected": "follow"})
    result = get_assert("The response fully complies with the skill's rules.", context)
    assert result["pass"] is True
    assert result["score"] == 1.0


def test_fixture_check_fails_for_follow_scenario_with_violation_language():
    context = _DictContext(vars={"expected": "follow"})
    result = get_assert("This response violates gate 2.", context)
    assert result["pass"] is False


def test_fixture_check_passes_for_violates_scenario_with_gate_match():
    context = _DictContext(
        vars={"expected": "violates", "violates_gate": "2. No-fabricated-scores gate"}
    )
    output = "This response violates the 2. no-fabricated-scores gate by inventing a score."
    result = get_assert(output, context)
    assert result["pass"] is True


def test_fixture_check_fails_for_violates_scenario_missing_gate_reference():
    context = _DictContext(
        vars={"expected": "violates", "violates_gate": "2. No-fabricated-scores gate"}
    )
    result = get_assert("This response violates the rules.", context)
    assert result["pass"] is False


def test_fixture_check_fails_for_violates_scenario_with_no_violation_flagged():
    context = _DictContext(vars={"expected": "violates"})
    result = get_assert("Everything here looks compliant.", context)
    assert result["pass"] is False
