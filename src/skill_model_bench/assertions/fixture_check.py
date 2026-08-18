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

import re
from typing import Any, Dict, List, Union

VIOLATION_PHRASES = (
    "violat",
    "non-compliant",
    "not compliant",
    "breaks the rule",
    "fails to comply",
    "does not follow",
    "in breach of",
)

# Fixture ``violates_gate`` strings are internal ground-truth labels, often
# written with a numbered-list prefix (e.g. "2. No-fabricated-scores gate").
# A real model response is not expected to reproduce that label verbatim —
# it will describe the same gate in its own words. Matching the literal
# string (including the number) would systematically under-score correct
# responses, which defeats the purpose of having fixture ground truth at
# all. Instead we strip the numbering, drop generic/short filler words, and
# require a majority of the remaining significant keywords to appear
# anywhere in the response. This is robust to real phrasing while still
# requiring the response to name the *specific* gate rather than just
# asserting "this violates the rules" in general.
_GATE_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")
_GATE_STOPWORDS = {
    "gate",
    "gates",
    "rule",
    "rules",
    "the",
    "a",
    "an",
    "of",
    "for",
    "and",
    "no",
}


def _gate_keywords(violates_gate: str) -> List[str]:
    stripped = _GATE_PREFIX_RE.sub("", violates_gate)
    words = re.split(r"[^a-zA-Z0-9]+", stripped.lower())
    return [w for w in words if len(w) >= 3 and w not in _GATE_STOPWORDS]


def _gate_matched(violates_gate: Any, text: str) -> bool:
    if not violates_gate:
        return True
    keywords = _gate_keywords(violates_gate)
    if not keywords:
        return True
    hits = sum(1 for keyword in keywords if keyword in text)
    return hits >= max(1, (len(keywords) + 1) // 2)


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
        gate_matched = _gate_matched(violates_gate, text)
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
