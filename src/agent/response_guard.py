"""Deterministic output safety layer (P3.2B): keep model prose consistent with the canonical status.

The certification adapter maps a live run to the observation the ten gates consume. Gate 8's automated
prose checks (A1-A5) read ``CaseObservation.final_text``; when the canonical review status is not
resolved (``REVIEW_REQUIRED`` / ``NEEDS_INFO``), a definitive compliance conclusion in that prose is
recorded as ``A5`` and fails the case. Prompt guidance reduces how often a model writes such a claim but
cannot guarantee it, so this module adds the approved **deterministic** layer between the model output
and Gate 8:

    model output -> response_guard -> check_final_response (Gate 8)

Scope and guarantees:

* pure and offline - no network, no credential, no I/O, no state, no randomness;
* **only** the affected sentence is replaced; every other sentence is preserved verbatim, so product
  category, rule IDs, findings, evidence and risk wording outside that sentence survive untouched;
* the canonical result (``CaseObservation.analysis_result``) is never modified - the guard rewrites the
  advisory prose only and invents no new facts;
* uncertainty statements are never touched (they contain no compliance conclusion);
* an already-resolved status is passed through unchanged.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, NamedTuple

# The approved replacement for a sentence that asserts a compliance conclusion the canonical status
# does not support. Wording is fixed: no fact, rule or timing is invented.
GUARD_REPLACEMENT = ("Evidence is insufficient to confirm compliance at this stage. "
                     "Further human review is required.")

# Unresolved canonical statuses. Gate 8's A5 compliance-claim check applies to both, so the guard does.
UNRESOLVED_STATUSES = frozenset({"REVIEW_REQUIRED", "NEEDS_INFO"})

# Definitive compliance conclusions (lower-case substrings). This is deliberately a local vocabulary:
# the certification core must never be imported by the agent layer. Sentences that preserve uncertainty
# ("evidence is insufficient", "further review is required", "cannot determine compliance") match none
# of these and are left alone.
COMPLIANCE_CONCLUSION_PATTERNS: tuple[str, ...] = (
    "is compliant", "are compliant", "fully compliant", "compliant with",
    "safe to import", "approved for import", "certified for import",
    "requirements are satisfied", "requirement is satisfied",
    "meets all requirements", "meets the requirements",
    "no compliance obligations",
)

# Sentence segmentation mirrors the Gate 8 check: split on sentence punctuation or a line break.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


class GuardFinding(NamedTuple):
    """One rewritten sentence: index, the unsafe substring matched, and the original sentence."""

    index: int
    pattern: str
    sentence: str


class GuardResult(NamedTuple):
    """Guarded text plus the deterministic findings that justify every rewrite."""

    text: str
    findings: tuple[GuardFinding, ...] = ()
    activated: bool = False


def _normalized_status(canonical_status: Any) -> str:
    """Accept a plain string, an enum member or ``None`` and return an upper-case status name."""
    if canonical_status is None:
        return ""
    return str(getattr(canonical_status, "value", canonical_status)).strip().upper()


def should_guard(canonical_status: Any) -> bool:
    """True only for an unresolved canonical status (the statuses Gate 8's A5 check applies to)."""
    return _normalized_status(canonical_status) in UNRESOLVED_STATUSES


def matched_conclusion(sentence: str) -> str | None:
    """Return the compliance-conclusion pattern found in one sentence, or ``None``."""
    lowered = sentence.lower()
    return next((pattern for pattern in COMPLIANCE_CONCLUSION_PATTERNS if pattern in lowered), None)


def rule_status_information(analysis_result: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    """Canonical ``(rule_id, rule_status)`` pairs, read-only, for callers that want to log the basis."""
    findings = ((analysis_result or {}).get("verified") or {}).get("compliance_information") or []
    return tuple((str(item.get("rule_id")), str(item.get("rule_status")))
                 for item in findings if isinstance(item, Mapping))


def guard_response_with_findings(
    final_text: Any,
    canonical_status: Any,
    rule_status_information: Iterable[Any] = (),
) -> GuardResult:
    """Apply the guard and report what was rewritten.

    ``rule_status_information`` is accepted for traceability (the canonical rule statuses the decision
    rests on) and never changes the outcome: activation depends solely on ``canonical_status``.
    """
    text = "" if final_text is None else str(final_text)
    if not should_guard(canonical_status) or not text.strip():
        return GuardResult(text, (), False)
    sentences = [part for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    if not sentences:                                   # whitespace-only text: nothing to rewrite
        return GuardResult(text, (), False)
    findings: list[GuardFinding] = []
    rewritten: list[str] = []
    for index, sentence in enumerate(sentences):
        pattern = matched_conclusion(sentence)
        if pattern is None:
            rewritten.append(sentence)
        else:
            findings.append(GuardFinding(index, pattern, sentence))
            rewritten.append(GUARD_REPLACEMENT)
    if not findings:
        return GuardResult(text, (), False)
    return GuardResult(" ".join(rewritten), tuple(findings), True)


def guard_response(
    final_text: Any,
    canonical_status: Any,
    rule_status_information: Iterable[Any] = (),
) -> str:
    """Return the guarded response text (the approved deterministic layer's single output)."""
    return guard_response_with_findings(final_text, canonical_status, rule_status_information).text


def guard_summary(result: GuardResult) -> str:
    """One sanitized line for the record/logs: what was rewritten and why, never the raw prose."""
    if not result.activated:
        return "response guard: not activated"
    patterns = ", ".join(sorted({finding.pattern for finding in result.findings}))
    return f"response guard: {len(result.findings)} compliance conclusion(s) rewritten ({patterns})"


def guard_observation_text(analysis_result: Mapping[str, Any] | None, final_text: Any) -> tuple[str, GuardResult]:
    """Convenience wrapper for the adapter: derive the canonical status, then guard the prose."""
    status = ((analysis_result or {}).get("review") or {}).get("status")
    result = guard_response_with_findings(final_text, status,
                                         rule_status_information(analysis_result))
    return result.text, result
