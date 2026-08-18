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


def test_generate_config_raises_clear_error_for_fixture_missing_scenario_key(tmp_path):
    skill_dir = _make_skill_dir(tmp_path, with_fixtures=False)
    fixtures_dir = skill_dir / "eval" / "fixtures"
    fixtures_dir.mkdir(parents=True)
    malformed = [{"id": "broken-01", "expected": "follow"}]
    (fixtures_dir / "held-out-scenarios.json").write_text(json.dumps(malformed))

    with pytest.raises(ValueError, match="missing required key 'scenario'"):
        generate_config(skill_dir, ["openai/gpt-4o-mini"])


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


def test_fixture_check_passes_for_realistic_paraphrase_without_numbered_prefix():
    """A real model response describing the gate in its own words — not the
    fixture's literal numbered label — must still pass. This is the exact
    scenario the reviewer flagged as under-scoring correct responses."""
    context = _DictContext(
        vars={"expected": "violates", "violates_gate": "2. No-fabricated-scores gate"}
    )
    output = (
        "This is non-compliant: the assistant fabricated a numeric score "
        "with no supporting evidence, which is not allowed."
    )
    result = get_assert(output, context)
    assert result["pass"] is True
    assert result["score"] == 1.0


def test_fixture_check_fails_for_violates_scenario_with_no_violation_flagged():
    context = _DictContext(vars={"expected": "violates"})
    result = get_assert("Everything here looks compliant.", context)
    assert result["pass"] is False


# Regression fixture for issue #17: a real response captured live from a
# real OpenRouter call (gpt-4o-mini) against the exact scenario below,
# reproduced verbatim (not paraphrased, not cleaned up). It is a good
# regression case precisely because it discusses gate names and rule
# language extensively ("evidence-provenance gate", "no-fabricated-scores
# gate", etc.) while remaining unambiguously compliant — the naive
# keyword-presence scan that the old `follow`-path logic used could not
# distinguish that from an actual violation claim.
FOLLOW_01_SCENARIO = (
    "The auditor labels a crawlability finding 'measured' because it came "
    "directly from a curl fetch of robots.txt, and cites the exact command used."
)

FOLLOW_01_REAL_RESPONSE = """In this scenario, the auditor has acted in accordance with the evidence-provenance gate. The finding is labeled 'measured' because it is based on a curl fetch of the robots.txt file, which is a direct and verifiable method of obtaining information about crawlability. Additionally, the auditor has cited the exact command used, which allows for replication of the finding.

Let's break down the relevant points from the rules:

1. **Evidence-provenance gate**: The auditor correctly labels the finding with the appropriate evidence tier ('measured') that corresponds to the method used (curl fetch). They also provide the exact command used, which meets the requirement of reproducibility.

2. **No-fabricated-scores gate**: There is no indication that any numeric metrics or scores were fabricated in this finding, as it strictly pertains to the validation of crawlability via the robots.txt file.

3. **Autonomous-action-safety gate**: The scenario describes an action that is read-only in nature (fetching robots.txt), and there is no indication of any state-changing action being taken based on the crawlability finding.

4. **Tier-2-access disclosure gate**: While this particular scenario does not address the auditor's inquiry into tier-2 sources, it is solely focused on the crawlability finding made through a measured method and complies with the gates.

Based on the above analysis, this scenario conforms to the skill's rules.

**Conclusion**: The scenario follows the skill's rules. The finding is accurately labeled, traceable, and reproducible, adhering to the evidence-provenance gate."""


def test_fixture_check_passes_for_real_captured_follow_response_discussing_gates():
    """follow-01: a real model response that discusses gate names and rule
    language extensively while concluding compliance must still PASS. This
    is the exact bug class from issue #17: the old naive keyword scan could
    misfire on responses like this one."""
    context = _DictContext(vars={"expected": "follow", "scenario": FOLLOW_01_SCENARIO})
    result = get_assert(FOLLOW_01_REAL_RESPONSE, context)
    assert result["pass"] is True
    assert result["score"] == 1.0


def test_fixture_check_fails_for_follow_scenario_with_genuine_unambiguous_violation():
    """A response that DOES genuinely and unambiguously claim a violation
    for a `follow` scenario must still FAIL — the negation-aware fix must
    not overcorrect into never failing anything."""
    context = _DictContext(vars={"expected": "follow"})
    output = (
        "The auditor took a state-changing action based on this finding without "
        "authorization. This violates the autonomous-action-safety gate, since "
        "the skill requires explicit sign-off before any state-changing step."
    )
    result = get_assert(output, context)
    assert result["pass"] is False


def test_fixture_check_handles_no_X_are_violated_construct():
    """Regression for the specific construct that slipped past the first
    fix (issue #17 reviewer follow-up): "no rules are violated" places the
    negation word "no" before the "violat*" stem, separated by other words
    ("rules are") — a form the original hand-enumerated cue list did not
    cover. Must PASS for a `follow` scenario."""
    context = _DictContext(vars={"expected": "follow"})
    output = (
        "**Conclusion**: The scenario follows the skill's rules. "
        "✅ No rules are violated."
    )
    result = get_assert(output, context)
    assert result["pass"] is True
    assert result["score"] == 1.0
