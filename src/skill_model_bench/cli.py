"""CLI entrypoint: wires config_gen -> promptfoo subprocess -> report together.

Usage (see ``skill-model-bench --help`` for the authoritative list):

    skill-model-bench <skill_dir> --model openai/gpt-4o-mini --model anthropic/claude-3.5-haiku
    skill-model-bench --skill-dir <path> --models openai/gpt-4o-mini,anthropic/claude-3.5-haiku
    skill-model-bench --results existing-results.json

If neither a ``promptfoo`` binary nor ``npx`` is found on PATH, the generated
config is written to disk and the exact manual command is printed instead of
running anything -- this tool does not assume Node.js/promptfoo is bundled or
always present.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from .config_gen import DEFAULT_JUDGE_PROVIDER, write_config
from .report import DEFAULT_QUALITY_BAR, build_report, render_markdown


class CliError(Exception):
    """A predictable, user-facing error -- caught in main() and reported cleanly."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skill-model-bench",
        description=(
            "Benchmark OpenRouter models against an agent skill's own fixtures "
            "(or a judge-model fallback), using promptfoo as the eval runner."
        ),
    )
    parser.add_argument(
        "skill_dir",
        nargs="?",
        default=None,
        help="Path to the skill directory (containing SKILL.md). Not required with --results.",
    )
    parser.add_argument(
        "--skill-dir",
        dest="skill_dir_opt",
        default=None,
        help="Alternative to the positional skill_dir argument.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        action="append",
        default=[],
        help="An OpenRouter model identifier, e.g. openai/gpt-4o-mini. Repeatable.",
    )
    parser.add_argument(
        "--models",
        dest="models",
        default=None,
        help="Comma-separated OpenRouter model identifiers, alternative to repeated --model.",
    )
    parser.add_argument(
        "--quality-bar",
        type=float,
        default=DEFAULT_QUALITY_BAR,
        help=f"Minimum pass rate a model must clear to be recommendable (default: {DEFAULT_QUALITY_BAR}).",
    )
    parser.add_argument(
        "--judge-provider",
        default=DEFAULT_JUDGE_PROVIDER,
        help="promptfoo provider id used to grade llm-rubric assertions when a skill has no fixtures.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the JSON report to this path (in addition to printing Markdown to stdout).",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Path to write the generated promptfoo config (default: a temp file).",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Path to an existing promptfoo results JSON file. Skips generating a config and "
        "running promptfoo entirely; goes straight to reporting.",
    )
    return parser


def _resolve_models(args: argparse.Namespace) -> List[str]:
    models: List[str] = list(args.model)
    if args.models:
        models.extend(part.strip() for part in args.models.split(",") if part.strip())
    if not models:
        raise CliError(
            "No models specified. Pass at least one model via --model or --models."
        )
    return models


def _resolve_skill_dir(args: argparse.Namespace) -> Path:
    raw = args.skill_dir_opt or args.skill_dir
    if not raw:
        raise CliError(
            "No skill directory given. Pass it as a positional argument or via --skill-dir."
        )
    skill_dir = Path(raw)
    if not skill_dir.is_dir():
        raise CliError(f"Skill directory not found: {skill_dir}")
    return skill_dir


def _promptfoo_command(config_path: Path, output_path: Path) -> Optional[List[str]]:
    promptfoo_bin = shutil.which("promptfoo")
    if promptfoo_bin:
        return [promptfoo_bin, "eval", "-c", str(config_path), "--output", str(output_path)]

    npx_bin = shutil.which("npx")
    if npx_bin:
        return [
            npx_bin,
            "--yes",
            "promptfoo@latest",
            "eval",
            "-c",
            str(config_path),
            "--output",
            str(output_path),
        ]

    return None


def _valid_results_file(results_path: Path) -> bool:
    """True if ``results_path`` exists and parses as promptfoo's results shape."""
    if not results_path.is_file():
        return False
    try:
        raw = json.loads(results_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(raw, dict) and "results" in raw


def run(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.results:
        report = build_report(args.results, quality_bar=args.quality_bar)
    else:
        skill_dir = _resolve_skill_dir(args)
        models = _resolve_models(args)

        # Not a context-managed TemporaryDirectory: on the "neither promptfoo
        # nor npx found" path we print this directory's config path for the
        # user to run manually, so it must still exist after we return.
        tmp_dir = Path(tempfile.mkdtemp(prefix="skill-model-bench-"))
        config_path = Path(args.config_path) if args.config_path else tmp_dir / "promptfoo-config.yaml"
        write_config(skill_dir, models, config_path, judge_provider=args.judge_provider)

        results_path = tmp_dir / "results.json"
        command = _promptfoo_command(config_path, results_path)

        if command is None:
            print(f"Generated promptfoo config: {config_path}")
            print(
                "Neither 'promptfoo' nor 'npx' was found on PATH. Run the eval yourself:\n"
                f"  promptfoo eval -c {config_path} --output results.json"
            )
            return 0

        completed = subprocess.run(command, check=False)

        # promptfoo exits non-zero (documented exit code 100) whenever any
        # test case fails or the pass rate is below threshold -- that's
        # normal signal, not a crash, and this tool's whole job is to
        # measure and report exactly that. Only a missing/unparseable
        # results file indicates a genuine crash worth treating as fatal.
        if not _valid_results_file(results_path):
            raise CliError(
                f"promptfoo did not produce a valid results file at {results_path} "
                f"(exit code {completed.returncode})."
            )

        report = build_report(results_path, quality_bar=args.quality_bar)

    print(render_markdown(report))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))

    return 0


def main(argv: Optional[List[str]] = None) -> None:
    try:
        exit_code = run(argv)
    except CliError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    else:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
