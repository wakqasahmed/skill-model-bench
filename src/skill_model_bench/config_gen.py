"""Generate a promptfoo YAML config from a skill directory and a model list.

promptfoo (https://www.promptfoo.dev) is an external Node.js CLI, not a Python
dependency of this project — it is not installed or invoked here. This module
only produces the config that ``promptfoo eval`` would later consume; running
that eval is out of scope for this issue.

Config shape follows promptfoo's documented schema:
- providers: OpenRouter models as ``openrouter:<model-id>``
  (https://www.promptfoo.dev/docs/providers/openrouter/).
- prompts: a single prompt built from the skill's SKILL.md content plus a
  ``{{scenario}}`` var substitution.
- tests: one test case per fixture scenario (or one llm-rubric-graded case
  per pseudo-scenario if no fixtures exist), each carrying
  ``metadata.ground_truth`` set to ``"fixture"`` or ``"judge"``
  (https://www.promptfoo.dev/docs/configuration/test-cases/).
- assert: a custom Python assertion (``type: python``, ``file://`` reference,
  https://www.promptfoo.dev/docs/configuration/expected-outputs/python/) when
  fixtures exist, otherwise ``type: llm-rubric``
  (https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/llm-rubric/).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

DEFAULT_JUDGE_PROVIDER = "openrouter:openai/gpt-4o-mini"

_ASSERTIONS_DIR = Path(__file__).resolve().parent / "assertions"
_FIXTURE_CHECK_PATH = _ASSERTIONS_DIR / "fixture_check.py"


def _load_fixtures(skill_dir: Path) -> Optional[List[Dict[str, Any]]]:
    fixtures_path = skill_dir / "eval" / "fixtures" / "held-out-scenarios.json"
    if not fixtures_path.is_file():
        return None
    data = json.loads(fixtures_path.read_text())
    if not isinstance(data, list) or not data:
        return None
    return data


def _load_skill_md(skill_dir: Path) -> str:
    skill_md_path = skill_dir / "SKILL.md"
    if not skill_md_path.is_file():
        raise FileNotFoundError(f"No SKILL.md found in {skill_dir}")
    return skill_md_path.read_text()


def _build_prompt(skill_md: str) -> str:
    return (
        f"{skill_md}\n\n"
        "---\n\n"
        "Given the skill above, evaluate the following scenario and state "
        "clearly whether it follows the skill's rules or violates them, "
        "citing which gate is violated if any:\n\n{{scenario}}"
    )


def _fixture_test_case(entry: Dict[str, Any]) -> Dict[str, Any]:
    scenario_vars = {
        "scenario": entry["scenario"],
        "expected": entry.get("expected"),
    }
    if "violates_gate" in entry:
        scenario_vars["violates_gate"] = entry["violates_gate"]

    return {
        "description": entry.get("id", "fixture-scenario"),
        "vars": scenario_vars,
        "assert": [
            {
                "type": "python",
                "value": f"file://{_FIXTURE_CHECK_PATH}",
            }
        ],
        "metadata": {
            "ground_truth": "fixture",
            "scenario_id": entry.get("id"),
        },
    }


def _judge_test_case(
    skill_name: str,
    judge_provider: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    scenario_text = (
        f"Assess whether the '{skill_name}' skill is being followed correctly "
        "in a general-purpose scenario for this skill (no fixtures were "
        "available for this skill, so no specific held-out scenario text "
        "exists)."
    )
    rubric = (
        "The response correctly identifies whether the described scenario "
        "follows or violates the skill's rules, and justifies the verdict "
        "by referencing the skill's own guidance."
    )
    return {
        "description": f"{skill_name}-judge-fallback",
        "vars": {"scenario": scenario_text},
        "assert": [
            {
                "type": "llm-rubric",
                "value": rubric,
                "provider": judge_provider,
            }
        ],
        "metadata": {
            "ground_truth": "judge",
            "scenario_id": f"{skill_name}-judge-fallback",
        },
    }


def generate_config(
    skill_dir: Union[str, Path],
    models: List[str],
    judge_provider: Union[str, Dict[str, Any]] = DEFAULT_JUDGE_PROVIDER,
) -> Dict[str, Any]:
    """Build a promptfoo config dict for the given skill and model list.

    Args:
        skill_dir: Path to a skill directory containing SKILL.md and,
            optionally, eval/fixtures/held-out-scenarios.json.
        models: OpenRouter model identifiers, e.g. "openai/gpt-4o-mini".
        judge_provider: promptfoo provider id (or object) used to grade
            llm-rubric assertions when a skill has no fixtures. Defaults to
            a cheap/fast model rather than the most expensive available one.
    """
    skill_dir = Path(skill_dir)
    skill_md = _load_skill_md(skill_dir)
    fixtures = _load_fixtures(skill_dir)

    config: Dict[str, Any] = {
        "providers": [f"openrouter:{model}" for model in models],
        "prompts": [_build_prompt(skill_md)],
    }

    if fixtures:
        config["tests"] = [_fixture_test_case(entry) for entry in fixtures]
    else:
        config["tests"] = [_judge_test_case(skill_dir.name, judge_provider)]

    return config


def write_config(
    skill_dir: Union[str, Path],
    models: List[str],
    output_path: Union[str, Path],
    judge_provider: Union[str, Dict[str, Any]] = DEFAULT_JUDGE_PROVIDER,
) -> Path:
    """Generate a config and write it to output_path as YAML. Returns the path."""
    config = generate_config(skill_dir, models, judge_provider=judge_provider)
    output_path = Path(output_path)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return output_path
