"""Deterministic output safety layer (P3.2B): keep model prose consistent with the canonical status.

The certification adapter maps a live run to the observation the ten gates consume. Gate 8's automated
prose checks (A1-A5) read ``CaseObservation.final_text``; when the canonical review status is not
resolved (``REVIEW_REQUIRED`` / ``NEEDS_INFO``), a definitive compliance conclusion in that prose is
recorded as ``A5`` and fails the case. The same gate fails definite *current-obligation* wording about a
rule whose canonical lifecycle is not a current effective obligation (PROPOSED / WATCHLIST / SUPERSEDED /
UNKNOWN). Prompt guidance reduces how often a model writes either form of claim but cannot guarantee it,
so this module adds the approved **deterministic** layer between the model output and Gate 8:

    canonical lifecycle/status -> model output -> response_guard -> check_final_response (Gate 8)

Scope and guarantees:

* pure and offline - no network, no credential, no I/O, no state, no randomness;
* **only** the affected sentence is replaced; every other sentence is preserved verbatim, so product
  category, rule IDs, findings, evidence and risk wording outside that sentence survive untouched;
* the canonical result (``CaseObservation.analysis_result``) is never modified - the guard rewrites the
  advisory prose only and invents no new facts;
* a rewrite restates the canonical lifecycle verbatim (``<rule_id> is <STATUS> and is not a current
  effective obligation``); it never invents a regulatory conclusion;
* a rule that is canonically EFFECTIVE is never lifecycle-rewritten, so a legitimate explanation of a
  real current obligation is preserved;
* uncertainty statements are never touched (they contain no compliance conclusion);
* an already-resolved status with no lifecycle promotion is passed through unchanged.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

# The approved replacement for a sentence that asserts a compliance conclusion the canonical status
# does not support. Wording is fixed: no fact, rule or timing is invented.
GUARD_REPLACEMENT = ("Evidence is insufficient to confirm compliance at this stage. "
                     "Further human review is required.")

# Unresolved canonical statuses. Gate 8's A5 compliance-claim check applies to both, so the guard does.
UNRESOLVED_STATUSES = frozenset({"REVIEW_REQUIRED", "NEEDS_INFO"})

# Canonical rule lifecycles that are NOT a current effective obligation. ``UNKNOWN`` means "not
# established as effective", so definite current-obligation wording about it is guarded too.
NON_CURRENT_LIFECYCLE_STATUSES = frozenset({"PROPOSED", "WATCHLIST", "SUPERSEDED", "UNKNOWN"})

# The canonical lifecycle and applicability values that together make a confirmed current obligation.
EFFECTIVE_LIFECYCLE = "EFFECTIVE"
APPLICABLE_STATUS = "APPLICABLE"

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

# Definite CURRENT-OBLIGATION wording. Generic on purpose: no rule id, no regulation name. The
# certification layer reuses this exact vocabulary for its A5 check.
OBLIGATION_PATTERNS: tuple[str, ...] = (
    "must comply", "must be complied with", "must meet", "must be met",
    "has to meet", "have to meet", "must be followed", "must be satisfied",
    "is required", "are required", "currently required", "required now", "now required",
    "is mandatory", "are mandatory", "is obligatory", "are obligatory", "is compulsory",
    "currently effective", "is in force", "are in force", "is the law",
)

# Context words that make an obligation phrase safe in the approved output: a negation (the sentence is
# stating the rule is NOT a current obligation) or a review/monitoring subject (the approved next step).
_NEGATIONS = frozenset({"not", "never", "no", "longer", "isn't", "aren't", "doesn't", "don't",
                        "cannot", "can't", "without"})
_REVIEW_SUBJECTS = frozenset({"review", "reviews", "monitoring", "attention", "verification",
                              "confirmation", "follow-up", "followup", "documentation", "evidence",
                              "clarification", "assessment", "remediation"})
# A negation is only honoured when it directly modifies the phrase ("is NOT currently effective"),
# so unrelated negation earlier in the sentence can never mask a real current-obligation claim
# ("R-ELEC-019 is not optional and is required" stays a violation). The review/monitoring subject is
# looked for a little further back so "human review of R-ELEC-019 is required" stays safe.
_NEGATION_WINDOW = 1
_SUBJECT_WINDOW = 4

# Sentence segmentation mirrors the Gate 8 check: split on sentence punctuation or a line break.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN = re.compile(r"[a-z0-9'\-]+")


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


class RuleAuthority(NamedTuple):
    """Read-only canonical authority for one rule: lifecycle and applicability, verbatim.

    A rule may be described as a **current confirmed obligation for this product** only when the
    canonical deterministic result says both ``rule_status == EFFECTIVE`` *and*
    ``applicability_status == APPLICABLE``. Every other combination - including an EFFECTIVE rule
    whose applicability is ``NOT_APPLICABLE`` / ``NEEDS_INFO`` / ``REVIEW_REQUIRED`` or absent - is
    not confirmed, so definite current-obligation wording about it is not authority-backed.
    """

    rule_id: str
    rule_status: str = ""
    applicability_status: str = ""

    @property
    def is_confirmed_current_obligation(self) -> bool:
        return (self.rule_status == EFFECTIVE_LIFECYCLE
                and self.applicability_status == APPLICABLE_STATUS)


def confirmed_current_obligation(rule: RuleAuthority) -> bool:
    """Canonical definition, shared by the guard and the Gate 8 checker (single source of truth)."""
    return rule.is_confirmed_current_obligation


def _coerce_authority(item: Any) -> RuleAuthority | None:
    """Accept a ``RuleAuthority``, a mapping, or a ``(rule_id, rule_status[, applicability])`` tuple."""
    if isinstance(item, RuleAuthority):
        return item
    if isinstance(item, Mapping):
        rule_id = item.get("rule_id")
        rule_status = item.get("rule_status")
        applicability = item.get("applicability_status")
    else:
        try:
            rule_id = item[0]
            rule_status = item[1]
            applicability = item[2] if len(item) > 2 else ""
        except (TypeError, ValueError, IndexError, KeyError):
            return None
    rule_id = str(rule_id or "").strip()
    if not rule_id:
        return None
    return RuleAuthority(rule_id, _normalized_status(rule_status), _normalized_status(applicability))


def rule_authorities(items: Iterable[Any] | None) -> tuple[RuleAuthority, ...]:
    """Normalize canonical authority entries, dropping duplicates and preserving order."""
    authorities: list[RuleAuthority] = []
    seen: set[str] = set()
    for item in items or ():
        authority = _coerce_authority(item)
        if authority is None or authority.rule_id in seen:
            continue
        seen.add(authority.rule_id)
        authorities.append(authority)
    return tuple(authorities)


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


def _prefix_tokens(sentence_lower: str, index: int, window: int) -> list[str]:
    """The few tokens immediately preceding a match — the only context a heuristic may rely on."""
    return _TOKEN.findall(sentence_lower[:index])[-window:]


def definite_current_obligation_phrases(sentence: str) -> tuple[str, ...]:
    """Obligation phrases this sentence ASSERTS definitely and currently.

    An occurrence is not counted when it is directly negated ("is not currently effective", "must not
    comply") or when its subject is the review/monitoring step the approved wording requires ("human
    review is required", "human review of R-ELEC-019 is required") rather than the rule or product
    itself. A plain substring test fails exactly that safe lifecycle wording, so the immediate context
    of each occurrence is checked instead — a distant negation never masks a real claim.
    """
    lowered = str(sentence or "").lower()
    phrases: list[str] = []
    for pattern in OBLIGATION_PATTERNS:
        start = 0
        while True:
            index = lowered.find(pattern, start)
            if index < 0:
                break
            start = index + len(pattern)
            if any(token in _NEGATIONS for token in _prefix_tokens(lowered, index, _NEGATION_WINDOW)):
                continue
            if any(token in _REVIEW_SUBJECTS for token in _prefix_tokens(lowered, index, _SUBJECT_WINDOW)):
                continue
            phrases.append(pattern)
    return tuple(phrases)


def has_definite_current_obligation(sentence: str) -> bool:
    """True when the sentence asserts a definite current obligation."""
    return bool(definite_current_obligation_phrases(sentence))


def non_current_rules(rule_status_information: Iterable[Any]) -> tuple[tuple[str, str], ...]:
    """Canonical ``(rule_id, rule_status)`` pairs whose lifecycle is not a current obligation.

    Kept for callers that only have lifecycle information; the guard itself uses
    :func:`rule_authority_information`, which also carries canonical applicability.
    """
    pairs: list[tuple[str, str]] = []
    for authority in rule_authorities(rule_status_information):
        if authority.rule_status not in NON_CURRENT_LIFECYCLE_STATUSES:
            continue
        pair = (authority.rule_id, authority.rule_status)
        if pair not in pairs:
            pairs.append(pair)
    return tuple(pairs)


def rule_authority_information(analysis_result: Mapping[str, Any] | None) -> tuple[RuleAuthority, ...]:
    """Read-only canonical projection: ``rule_id`` + ``rule_status`` + ``applicability_status``.

    Joins the already-produced canonical analysis result only - no compliance engine is imported,
    instantiated or re-run, and the result itself is never modified. A rule that appears in the
    applicability verdicts but not in the findings is included with an unknown lifecycle, which is
    never a confirmed current obligation.
    """
    result = analysis_result or {}
    findings = ((result.get("verified") or {}).get("compliance_information")) or []
    applicability = {
        str(rule.get("rule_id")): _normalized_status(rule.get("applicability_status"))
        for rule in ((result.get("applicability") or {}).get("rules")) or []
        if isinstance(rule, Mapping) and rule.get("rule_id")
    }
    authorities: list[RuleAuthority] = []
    seen: set[str] = set()
    for finding in findings:
        if not isinstance(finding, Mapping):
            continue
        rule_id = str(finding.get("rule_id") or "").strip()
        if not rule_id or rule_id in seen:
            continue
        seen.add(rule_id)
        authorities.append(RuleAuthority(rule_id, _normalized_status(finding.get("rule_status")),
                                         applicability.get(rule_id, "")))
    for rule_id, status in applicability.items():
        if rule_id not in seen:
            seen.add(rule_id)
            authorities.append(RuleAuthority(rule_id, "", status))
    return tuple(authorities)


def _rule_clause(rule: RuleAuthority) -> str:
    """State only the canonical facts that exist for one rule."""
    if rule.rule_status and rule.applicability_status:
        return (f"{rule.rule_id} is {rule.rule_status} with canonical applicability "
                f"{rule.applicability_status}")
    if rule.rule_status:
        return f"{rule.rule_id} is {rule.rule_status}"
    if rule.applicability_status:
        return f"{rule.rule_id} has canonical applicability {rule.applicability_status}"
    return f"{rule.rule_id} has no confirmed canonical applicability"


def authority_replacement(rules: Sequence[RuleAuthority]) -> str:
    """Status-faithful replacement for a sentence that asserted an unconfirmed obligation.

    Restates the canonical lifecycle/applicability values verbatim and asserts nothing else: no new
    legal obligation and no invented workflow requirement (the earlier "human review or monitoring
    still applies" closer is deliberately gone, because it implied a process duty some statuses such
    as SUPERSEDED do not support).
    """
    clauses = "; ".join(_rule_clause(rule) for rule in rules)
    return (f"{clauses}; not confirmed as a current applicable obligation for this product.")


def lifecycle_replacement(pairs: Sequence[tuple[str, str]]) -> str:
    """Backwards-compatible wrapper: pairs are treated as lifecycle-only (never confirmed)."""
    return authority_replacement(rule_authorities([(rule_id, status) for rule_id, status in pairs]))


def rule_status_information(analysis_result: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    """Canonical ``(rule_id, rule_status)`` pairs, read-only, for callers that want to log the basis."""
    findings = ((analysis_result or {}).get("verified") or {}).get("compliance_information") or []
    return tuple((str(item.get("rule_id")), str(item.get("rule_status")))
                 for item in findings if isinstance(item, Mapping))


# Narrow, deterministic adjacent-reference handling. Deliberately NOT coreference resolution: only
# "it" / "this rule" / "the rule" / "that rule", only from the immediately preceding sentence, and only
# when that sentence established EXACTLY ONE canonical rule.
_REFERENCE_MARKERS = ("it", "this rule", "the rule", "that rule")
_REFERENCE_RES = tuple(
    re.compile(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])") for marker in _REFERENCE_MARKERS
)


def uses_direct_reference(sentence: str) -> bool:
    """True when the sentence contains a simple direct reference to a previously named rule."""
    lowered = str(sentence or "").lower()
    return any(pattern.search(lowered) for pattern in _REFERENCE_RES)


def mentions_rule(sentence: str, rule_id: str) -> bool:
    """True when the sentence names this exact rule id (never a prefix of a longer id)."""
    if not rule_id:
        return False
    pattern = rf"(?<![a-z0-9\-]){re.escape(str(rule_id).lower())}(?![a-z0-9\-])"
    return re.search(pattern, str(sentence or "").lower()) is not None


def split_sentences(text: Any) -> tuple[str, ...]:
    """Sentence segmentation, identical to the Gate 8 check: punctuation or a line break."""
    return tuple(part for part in _SENTENCE_SPLIT.split("" if text is None else str(text)) if part.strip())


def associated_rules(
    text: Any, authorities: Sequence[RuleAuthority]
) -> tuple[tuple[RuleAuthority, ...], ...]:
    """Per sentence, the canonical rules that sentence is about.

    A sentence is associated with the rules it names itself. A sentence that names none is associated
    with the *immediately preceding* sentence's rule **only** when it uses a simple direct reference
    and that sentence established exactly one rule. References are never carried further than one
    sentence, and an ambiguous antecedent (two or more rules in the previous sentence) resolves to
    nothing - the guard never guesses.
    """
    sentences = split_sentences(text)
    associations: list[tuple[RuleAuthority, ...]] = []
    previous_mentions: tuple[RuleAuthority, ...] = ()
    for sentence in sentences:
        lowered = sentence.lower()
        mentioned = tuple(rule for rule in authorities if mentions_rule(lowered, rule.rule_id))
        if mentioned:
            association = mentioned
        elif len(previous_mentions) == 1 and uses_direct_reference(lowered):
            association = previous_mentions
        else:
            association = ()
        associations.append(association)
        previous_mentions = mentioned
    return tuple(associations)


def unconfirmed_rules(rules: Sequence[RuleAuthority]) -> tuple[RuleAuthority, ...]:
    """The subset of rules that canonical metadata does not confirm as current obligations."""
    return tuple(rule for rule in rules if not rule.is_confirmed_current_obligation)


def guard_response_with_findings(
    final_text: Any,
    canonical_status: Any,
    rule_status_information: Iterable[Any] = (),
) -> GuardResult:
    """Apply the guard and report what was rewritten.

    Two independent safety rules, both deterministic:

    * a compliance conclusion while the canonical review status is unresolved, and
    * definite current-obligation wording about a rule that the canonical result does **not** confirm
      as a current applicable obligation - i.e. any rule other than
      ``rule_status == EFFECTIVE AND applicability_status == APPLICABLE``, which includes a
      non-current lifecycle (PROPOSED / WATCHLIST / SUPERSEDED / UNKNOWN) and an EFFECTIVE rule whose
      canonical applicability is ``NOT_APPLICABLE`` / ``NEEDS_INFO`` / ``REVIEW_REQUIRED`` or absent.

    ``rule_status_information`` carries the canonical authority entries (see
    :func:`rule_authority_information`). The second rule does not depend on the review status: an
    unconfirmed obligation claim is unsafe regardless of how the review status reads.
    """
    text = "" if final_text is None else str(final_text)
    authorities = rule_authorities(rule_status_information)
    guard_conclusions = should_guard(canonical_status)
    if not text.strip() or (not guard_conclusions and not authorities):
        return GuardResult(text, (), False)
    sentences = split_sentences(text)
    if not sentences:                                   # whitespace-only text: nothing to rewrite
        return GuardResult(text, (), False)
    associations = associated_rules(text, authorities)
    findings: list[GuardFinding] = []
    rewritten: list[str] = []
    for index, sentence in enumerate(sentences):
        unconfirmed = unconfirmed_rules(associations[index])
        if unconfirmed and has_definite_current_obligation(sentence):
            label = ", ".join(
                f"{rule.rule_id}/{rule.rule_status or 'UNKNOWN'}"
                f"{'/' + rule.applicability_status if rule.applicability_status else ''}"
                for rule in unconfirmed
            )
            findings.append(GuardFinding(index, f"unconfirmed current obligation ({label})", sentence))
            rewritten.append(authority_replacement(unconfirmed))
            continue
        pattern = matched_conclusion(sentence) if guard_conclusions else None
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
    """Convenience wrapper for the adapter: derive the canonical authority, then guard the prose."""
    status = ((analysis_result or {}).get("review") or {}).get("status")
    result = guard_response_with_findings(final_text, status,
                                         rule_authority_information(analysis_result))
    return result.text, result
