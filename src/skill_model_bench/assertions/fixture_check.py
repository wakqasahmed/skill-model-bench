"""Custom promptfoo Python assertion for fixture-backed skill scenarios.

promptfoo invokes ``get_assert(output, context)`` for a ``type: python`` assert
entry (see https://www.promptfoo.dev/docs/configuration/expected-outputs/python/).
This module is referenced from generated configs as
``file://.../fixture_check.py`` and scores a model's response against the
scenario's own `expected: follow|violates` (+ `violates_gate`) ground truth,
the same deterministic phrase-matching style used by the `contract_check.py`
scripts in wakqasahmed/verified-audit-skills, adapted to score a live model
response instead of a skill's own SKILL.md text.
"""

from typing import Any, Dict, Union

VIOLATION_PHRASES = (
    "violat",
    "non-compliant",
    "not compliant",
    "breaks the rule",
    "fails to comply",
    "does not follow",
    "in breach of",
)


def _get(context: Any, key: str, default: Any = None) -> Any:
    if isinstance(context, dict):
        return context.get(key, default)
    return getattr(context, key, default)


def get_assert(output: str, context: Any) -> Dict[str, Union[bool, float, str]]:
    vars_ = _get(context, "vars", {}) or {}
    expected = _get(vars_, "expected")
    violates_gate = _get(vars_, "violates_gate")

    text = (output or "").lower()
    flagged_violation = any(phrase in text for phrase in VIOLATION_PHRASES)

    if expected == "violates":
        gate_matched = True
        if violates_gate:
            gate_matched = violates_gate.lower() in text
        passed = flagged_violation and gate_matched
        if passed:
            reason = f"Response flags a violation and references gate {violates_gate!r}."
        elif not flagged_violation:
            reason = "Response does not flag a violation, but expected='violates'."
        else:
            reason = f"Response flags a violation but never references gate {violates_gate!r}."
    elif expected == "follow":
        passed = not flagged_violation
        reason = (
            "Response does not flag a violation, consistent with expected='follow'."
            if passed
            else "Response flags a violation, but expected='follow'."
        )
    else:
        passed = False
        reason = f"Unrecognized expected value: {expected!r}"

    return {"pass": passed, "score": 1.0 if passed else 0.0, "reason": reason}
