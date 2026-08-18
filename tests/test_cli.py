"""Tests for the CLI wrapper around promptfoo.

No real promptfoo binary, network access, or OPENROUTER_API_KEY is required --
``subprocess.run`` and ``shutil.which`` are mocked throughout.
"""

import json
from unittest.mock import patch

import pytest

from skill_model_bench.cli import main


SKILL_MD = "# Example Skill\n\nSome rules the model must follow.\n"


def _make_skill_dir(tmp_path):
    skill_dir = tmp_path / "example-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(SKILL_MD)
    return skill_dir


def _fake_promptfoo_results():
    return {
        "evalId": "eval-test",
        "results": {
            "results": [
                {
                    "provider": {"id": "openrouter:openai/gpt-4o-mini", "label": ""},
                    "success": True,
                    "score": 1,
                    "cost": 0.001,
                    "latencyMs": 100.0,
                    "gradingResult": {"pass": True, "score": 1, "reason": ""},
                    "testCase": {
                        "vars": {},
                        "assert": [],
                        "metadata": {"ground_truth": "judge", "scenario_id": "s"},
                    },
                }
            ]
        },
        "config": {},
    }


def test_happy_path_runs_promptfoo_and_prints_report(tmp_path, capsys):
    skill_dir = _make_skill_dir(tmp_path)

    def fake_run(command, check=False):
        output_path = command[command.index("--output") + 1]
        with open(output_path, "w") as f:
            json.dump(_fake_promptfoo_results(), f)
        return None

    with patch("skill_model_bench.cli.shutil.which", side_effect=lambda name: "/usr/bin/promptfoo" if name == "promptfoo" else None), \
         patch("skill_model_bench.cli.subprocess.run", side_effect=fake_run) as mock_run:
        with pytest.raises(SystemExit) as exc_info:
            main([str(skill_dir), "--model", "openai/gpt-4o-mini"])

    assert exc_info.value.code == 0
    mock_run.assert_called_once()
    command = mock_run.call_args[0][0]
    assert command[0] == "/usr/bin/promptfoo"
    assert "eval" in command

    captured = capsys.readouterr()
    assert "Model benchmark report" in captured.out
    assert "openrouter:openai/gpt-4o-mini" in captured.out


def test_happy_path_writes_json_report_when_output_given(tmp_path):
    skill_dir = _make_skill_dir(tmp_path)
    output_path = tmp_path / "report.json"

    def fake_run(command, check=False):
        results_path = command[command.index("--output") + 1]
        with open(results_path, "w") as f:
            json.dump(_fake_promptfoo_results(), f)
        return None

    with patch("skill_model_bench.cli.shutil.which", side_effect=lambda name: "/usr/bin/promptfoo" if name == "promptfoo" else None), \
         patch("skill_model_bench.cli.subprocess.run", side_effect=fake_run):
        with pytest.raises(SystemExit) as exc_info:
            main([str(skill_dir), "--model", "openai/gpt-4o-mini", "--output", str(output_path)])

    assert exc_info.value.code == 0
    report = json.loads(output_path.read_text())
    assert "openrouter:openai/gpt-4o-mini" in report["models"]


def test_nonzero_exit_with_valid_results_still_produces_report(tmp_path, capsys):
    """promptfoo exits non-zero (e.g. code 100) whenever a test case fails --
    that's normal signal, not a crash. As long as results.json was written
    and parses correctly, the CLI must proceed to report generation instead
    of raising/exiting with an error."""
    skill_dir = _make_skill_dir(tmp_path)

    def fake_run(command, check=False):
        output_path = command[command.index("--output") + 1]
        with open(output_path, "w") as f:
            json.dump(_fake_promptfoo_results(), f)

        class _CompletedProcess:
            returncode = 100

        return _CompletedProcess()

    with patch("skill_model_bench.cli.shutil.which", side_effect=lambda name: "/usr/bin/promptfoo" if name == "promptfoo" else None), \
         patch("skill_model_bench.cli.subprocess.run", side_effect=fake_run) as mock_run:
        with pytest.raises(SystemExit) as exc_info:
            main([str(skill_dir), "--model", "openai/gpt-4o-mini"])

    assert exc_info.value.code == 0
    mock_run.assert_called_once()

    captured = capsys.readouterr()
    assert "Model benchmark report" in captured.out
    assert "openrouter:openai/gpt-4o-mini" in captured.out


def test_falls_back_to_npx_when_promptfoo_binary_missing(tmp_path):
    skill_dir = _make_skill_dir(tmp_path)

    def fake_run(command, check=False):
        results_path = command[command.index("--output") + 1]
        with open(results_path, "w") as f:
            json.dump(_fake_promptfoo_results(), f)
        return None

    def fake_which(name):
        return "/usr/bin/npx" if name == "npx" else None

    with patch("skill_model_bench.cli.shutil.which", side_effect=fake_which), \
         patch("skill_model_bench.cli.subprocess.run", side_effect=fake_run) as mock_run:
        with pytest.raises(SystemExit) as exc_info:
            main([str(skill_dir), "--model", "openai/gpt-4o-mini"])

    assert exc_info.value.code == 0
    command = mock_run.call_args[0][0]
    assert command[0] == "/usr/bin/npx"
    assert "promptfoo@latest" in command


def test_neither_promptfoo_nor_npx_prints_manual_instructions_and_exits_zero(tmp_path, capsys):
    skill_dir = _make_skill_dir(tmp_path)

    with patch("skill_model_bench.cli.shutil.which", return_value=None), \
         patch("skill_model_bench.cli.subprocess.run") as mock_run:
        with pytest.raises(SystemExit) as exc_info:
            main([str(skill_dir), "--model", "openai/gpt-4o-mini"])

    assert exc_info.value.code == 0
    mock_run.assert_not_called()

    captured = capsys.readouterr()
    assert "Generated promptfoo config:" in captured.out
    assert "promptfoo eval -c" in captured.out
    assert "--output results.json" in captured.out


def test_results_flag_skips_running_promptfoo_entirely(tmp_path, capsys):
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(_fake_promptfoo_results()))

    with patch("skill_model_bench.cli.subprocess.run") as mock_run, \
         patch("skill_model_bench.cli.write_config") as mock_write_config:
        with pytest.raises(SystemExit) as exc_info:
            main(["--results", str(results_path)])

    assert exc_info.value.code == 0
    mock_run.assert_not_called()
    mock_write_config.assert_not_called()

    captured = capsys.readouterr()
    assert "Model benchmark report" in captured.out
    assert "openrouter:openai/gpt-4o-mini" in captured.out


def test_bad_skill_directory_produces_clear_error_not_traceback(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["/nonexistent/skill/dir", "--model", "openai/gpt-4o-mini"])

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "not found" in captured.err.lower()


def test_no_models_specified_produces_clear_error(tmp_path, capsys):
    skill_dir = _make_skill_dir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        main([str(skill_dir)])

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "no models" in captured.err.lower()


def test_missing_skill_md_produces_clear_error(tmp_path, capsys):
    skill_dir = tmp_path / "no-skill-md"
    skill_dir.mkdir()

    with patch("skill_model_bench.cli.shutil.which", return_value=None):
        with pytest.raises(SystemExit) as exc_info:
            main([str(skill_dir), "--model", "openai/gpt-4o-mini"])

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "Error:" in captured.err


def test_help_prints_usage_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "skill-model-bench" in captured.out
    assert "--model" in captured.out
