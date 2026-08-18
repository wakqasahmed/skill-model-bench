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


# For an `expected: "follow"` scenario, the bare presence of a violation
# phrase anywhere in the response is not evidence of anything: a correct
# response commonly *names the gate it checked* and states that it was
# satisfied ("does not violate the evidence-provenance gate", "no violation
# of the no-fabricated-scores gate was found"), or quotes the rule text
# itself while walking through why the scenario is compliant. A bare
# substring scan over the whole response (the old `flagged_violation = any(
# phrase in text ...)`) cannot tell "X violates the rule" apart from "X does
# not violate the rule" and fails every response that so much as discusses
# the rule it is complying with — which is how real models write compliant
# answers.
#
# The fix scopes the scan to one sentence at a time and requires a clear,
# *unnegated* violation claim within that sentence before treating it as a
# violation signal. A violation phrase sitting in a sentence that also
# carries a negation/compliance cue ("does not violate", "complies with",
# "no indication that ... violated", "would violate" as a hypothetical) is
# treated as a compliance statement, not a violation claim. Only a sentence
# with a violation phrase and *no* accompanying negation/compliance cue is
# scored as an actual violation claim. Gate names or rule text appearing on
# their own (with no violation phrase at all, as in the real response used
# in the regression test below) never trigger anything — this deliberately
# avoids inventing a second "gate-name mention" heuristic, which is exactly
# the kind of naive scan that would misfire on a compliant response that
# discusses gates extensively while explaining why it passed.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

_COMPLIANCE_CUES = (
    " not ",
    "n't ",
    "no violation",
    "no indication",
    "does not",
    "did not",
    "without violating",
    "never violat",
    "complies",
    "compliant",
    "complied",
    "satisfies",
    "satisfied",
    "satisfying",
    "in accordance",
    "consistent with",
    "conforms",
    "conforming",
    "follows the",
    "adheres",
    "adhering",
    "adhered",
    "meets the requirement",
    "would violate",
    "would be violated",
    "would breach",
    "could violate",
)


def _has_unnegated_violation_claim(text: str) -> bool:
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        if not any(phrase in sentence for phrase in VIOLATION_PHRASES):
            continue
        if any(cue in sentence for cue in _COMPLIANCE_CUES):
            continue
        return True
    return False


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
        unnegated_violation = _has_unnegated_violation_claim(text)
        passed = not unnegated_violation
        reason = (
            "Response does not make an unnegated violation claim, consistent with "
            "expected='follow'."
            if passed
            else "Response makes a clear, unnegated violation claim, but expected='follow'."
        )
    else:
        passed = False
        reason = f"Unrecognized expected value: {expected!r}"

    return {"pass": passed, "score": 1.0 if passed else 0.0, "reason": reason}
