"""Offline tests for the deterministic response guard (`src/agent/response_guard.py`).

Pure, offline, no provider, no credential. These tests pin the guard's contract: it activates only for an
unresolved canonical status, rewrites only the sentence that asserts a compliance conclusion, preserves
uncertainty wording and every other sentence, and never touches the canonical result.
"""

from __future__ import annotations

import inspect
import json

import pytest

from src.agent import response_guard as guard

REPLACEMENT = ("Evidence is insufficient to confirm compliance at this stage. "
               "Further human review is required.")


# =========================================================================== #
# 1 — required cases from the task
# =========================================================================== #
def test_uncertainty_statement_is_returned_unchanged() -> None:
    text = "Evidence is insufficient to confirm compliance. Further review is required."
    assert guard.guard_response(text, "REVIEW_REQUIRED") == text


def test_compliance_claim_with_all_requirements_is_rewritten() -> None:
    guarded = guard.guard_response("The product is compliant with all requirements.", "REVIEW_REQUIRED")
    assert guarded == REPLACEMENT
    assert "is compliant" not in guarded


def test_safe_to_import_claim_is_rewritten() -> None:
    guarded = guard.guard_response("The product is safe to import.", "REVIEW_REQUIRED")
    assert guarded == REPLACEMENT


def test_resolved_status_is_never_rewritten() -> None:
    text = "The product is compliant."
    assert guard.guard_response(text, "RESOLVED") == text
    assert guard.guard_response(text, "UNSUPPORTED") == text
    assert guard.guard_response(text, None) == text
    assert guard.guard_response(text, "") == text


# =========================================================================== #
# 2 — activation rule
# =========================================================================== #
def test_guard_activates_only_for_an_unresolved_status() -> None:
    assert guard.should_guard("REVIEW_REQUIRED") is True
    assert guard.should_guard("review_required") is True          # case-insensitive status
    assert guard.should_guard("NEEDS_INFO") is True               # same A5 rule in the core
    assert guard.should_guard("RESOLVED") is False
    assert guard.should_guard("UNSUPPORTED") is False
    assert guard.should_guard(None) is False


def test_guard_accepts_an_enum_status_value() -> None:
    class _Status:
        value = "REVIEW_REQUIRED"

    assert guard.guard_response("The product is compliant.", _Status()) == REPLACEMENT


# =========================================================================== #
# 3 — detection vocabulary
# =========================================================================== #
@pytest.mark.parametrize("claim", [
    "The product is compliant.",
    "The product is fully compliant.",
    "These products are compliant with the rule.",
    "The product is compliant with all requirements.",
    "The product is safe to import.",
    "The item is approved for import.",
    "The item is certified for import.",
    "All requirements are satisfied.",
    "The product meets all requirements.",
    "There are no compliance obligations for this product.",
    "The product is compliant overall.",
])
def test_definitive_conclusions_are_detected(claim: str) -> None:
    assert guard.matched_conclusion(claim) is not None, claim


@pytest.mark.parametrize("safe", [
    "Evidence is insufficient to confirm compliance.",
    "Further review is required.",
    "Compliance cannot be determined at this stage.",
    "Compliance cannot be confirmed at this stage.",
    "R-ELEC-019 is proposed and not yet in force.",
    "R-ELEC-002 is applicable per the canonical engine.",
    "The product may require an FCC SDoC; human review remains required.",
    "Additional documentation must be provided before this can be assessed.",
])
def test_uncertainty_and_status_statements_are_not_detected(safe: str) -> None:
    assert guard.matched_conclusion(safe) is None, safe
    assert guard.guard_response(safe, "REVIEW_REQUIRED") == safe


# =========================================================================== #
# 4 — surgical rewrite: only the offending sentence changes
# =========================================================================== #
def test_only_the_claim_sentence_is_replaced() -> None:
    text = ("The product category is small_consumer_electronics. "
            "R-ELEC-002 is applicable per the canonical engine. "
            "The product is compliant with all requirements. "
            "R-ELEC-019 is proposed and not yet in force.")
    guarded = guard.guard_response(text, "REVIEW_REQUIRED")
    lines = guarded.split(". ")
    assert lines[0] == "The product category is small_consumer_electronics"
    assert "R-ELEC-002 is applicable per the canonical engine" in guarded
    assert "R-ELEC-019 is proposed and not yet in force" in guarded
    assert REPLACEMENT in guarded
    assert "is compliant" not in guarded


def test_multiple_claims_are_each_replaced() -> None:
    result = guard.guard_response_with_findings(
        "The product is compliant. It is safe to import. Further review is required.", "REVIEW_REQUIRED")
    assert result.activated is True
    assert len(result.findings) == 2
    assert result.text.count(REPLACEMENT) == 2
    assert result.text.endswith("Further review is required.")
    assert [finding.pattern for finding in result.findings] == ["is compliant", "safe to import"]
    # Findings describe the rewrite without leaking the unsafe sentence back out.
    assert all(finding.sentence.startswith(("The product", "It is")) for finding in result.findings)


def test_findings_are_empty_and_activated_false_when_nothing_is_rewritten() -> None:
    result = guard.guard_response_with_findings("Evidence is insufficient to confirm compliance.",
                                                "REVIEW_REQUIRED")
    assert result.activated is False
    assert result.findings == ()
    assert result.text == "Evidence is insufficient to confirm compliance."
    assert guard.guard_summary(result) == "response guard: not activated"


def test_guard_summary_lists_the_matched_patterns() -> None:
    result = guard.guard_response_with_findings("The product is compliant.", "REVIEW_REQUIRED")
    assert guard.guard_summary(result) == ("response guard: 1 compliance conclusion(s) rewritten "
                                           "(is compliant)")


# =========================================================================== #
# 5 — rewrite output is itself Gate-8-clean
# =========================================================================== #
def test_replacement_is_a5_safe() -> None:
    from src.certification import certification as cert

    assert cert.check_final_response(REPLACEMENT, non_effective_rule_ids=("R-ELEC-018", "R-ELEC-019"),
                                     review_status="REVIEW_REQUIRED") == []
    # No compliance conclusion and no "absent requirement" phrasing.
    assert not any(pattern in REPLACEMENT.lower() for pattern in cert.COMPLIANCE_CLAIMS)
    assert not any(pattern in REPLACEMENT.lower() for pattern in cert.ABSENCE_PHRASES)
    # It does contain an obligation phrase ("is required"), which is A5-safe ONLY because the replacement
    # names no rule: A5 pairs a rule_id with obligation wording, and this text has none.
    assert "is required" in REPLACEMENT.lower()
    assert not any(rule_id in REPLACEMENT for rule_id in ("R-ELEC-018", "R-ELEC-019", "R-ELEC-002"))


def test_guard_removes_the_a5_compliance_finding_from_a_guarded_case() -> None:
    """The guard's purpose: A5 must not fire on what Gate 8 finally reads."""
    from src.certification import certification as cert

    text = "The product is compliant with all requirements."
    assert any(p.startswith("A5") for p in cert.check_final_response(text, review_status="REVIEW_REQUIRED"))
    guarded = guard.guard_response(text, "REVIEW_REQUIRED")
    assert cert.check_final_response(guarded, review_status="REVIEW_REQUIRED") == []


# =========================================================================== #
# 6 — contract: pure, no canonical mutation, no invented facts
# =========================================================================== #
def test_guard_module_is_pure_and_offline() -> None:
    source = inspect.getsource(guard)
    for forbidden in ("os.environ", "getenv", "socket", "requests", "print(", "input("):
        assert forbidden not in source, forbidden


def test_rule_status_information_is_read_only_and_carried() -> None:
    result = {
        "review": {"status": "REVIEW_REQUIRED"},
        "verified": {"compliance_information": [
            {"rule_id": "R-ELEC-018", "rule_status": "WATCHLIST"},
            {"rule_id": "R-ELEC-019", "rule_status": "PROPOSED"},
        ]},
    }
    info = guard.rule_status_information(result)
    assert info == (("R-ELEC-018", "WATCHLIST"), ("R-ELEC-019", "PROPOSED"))
    assert result["verified"]["compliance_information"][1]["rule_status"] == "PROPOSED"   # unmutated
    assert guard.guard_response("The product is compliant.", "REVIEW_REQUIRED", info) == REPLACEMENT


def test_guard_observation_text_derives_the_status_from_the_canonical_result() -> None:
    result = {"review": {"status": "REVIEW_REQUIRED"},
              "verified": {"compliance_information": [{"rule_id": "R-ELEC-019", "rule_status": "PROPOSED"}]}}
    text, outcome = guard.guard_observation_text(result, "The product is safe to import.")
    assert text == REPLACEMENT
    assert outcome.activated is True
    resolved = {"review": {"status": "RESOLVED"}}
    text2, outcome2 = guard.guard_observation_text(resolved, "The product is safe to import.")
    assert text2 == "The product is safe to import."
    assert outcome2.activated is False


@pytest.mark.parametrize("empty", [None, "", "   ", "\n"])
def test_empty_text_is_returned_unchanged(empty) -> None:
    assert guard.guard_response(empty, "REVIEW_REQUIRED") == ("" if empty is None else empty)


# =========================================================================== #
# 5 — canonical-authority guarding (live Case B regression)
# =========================================================================== #
# Only EFFECTIVE + APPLICABLE is a confirmed current obligation; everything else is guarded.
AUTHORITIES = (
    guard.RuleAuthority("R-ELEC-002", "EFFECTIVE", "APPLICABLE"),
    guard.RuleAuthority("R-ELEC-018", "WATCHLIST", "APPLICABLE"),
    guard.RuleAuthority("R-ELEC-019", "PROPOSED", "APPLICABLE"),
)
LIFECYCLE = AUTHORITIES
NOT_CONFIRMED_MARKER = "not confirmed as a current applicable obligation for this product"


def test_non_current_rules_keeps_only_non_current_lifecycles() -> None:
    assert guard.non_current_rules(LIFECYCLE) == (("R-ELEC-018", "WATCHLIST"), ("R-ELEC-019", "PROPOSED"))
    # Mappings are accepted too, and UNKNOWN counts as not-established-as-effective.
    assert guard.non_current_rules([
        {"rule_id": "R-X", "rule_status": "UNKNOWN"},
        {"rule_id": "R-Y", "rule_status": "EFFECTIVE"},
        {"rule_id": "R-Z", "rule_status": "superseded"},
    ]) == (("R-X", "UNKNOWN"), ("R-Z", "SUPERSEDED"))


def test_confirmed_current_obligation_requires_effective_and_applicable() -> None:
    """The canonical authority definition shared by the guard and the Gate 8 checker."""
    confirmed = guard.RuleAuthority("R-X", "EFFECTIVE", "APPLICABLE")
    assert guard.confirmed_current_obligation(confirmed) is True
    for lifecycle, applicability in (
        ("EFFECTIVE", "NEEDS_INFO"),
        ("EFFECTIVE", "REVIEW_REQUIRED"),
        ("EFFECTIVE", "NOT_APPLICABLE"),
        ("EFFECTIVE", ""),
        ("PROPOSED", "APPLICABLE"),
        ("WATCHLIST", "APPLICABLE"),
        ("SUPERSEDED", "APPLICABLE"),
        ("UNKNOWN", "APPLICABLE"),
        ("", "APPLICABLE"),
    ):
        rule = guard.RuleAuthority("R-X", lifecycle, applicability)
        assert guard.confirmed_current_obligation(rule) is False, (lifecycle, applicability)


def test_rule_authority_information_joins_lifecycle_and_applicability() -> None:
    result = {
        "verified": {"compliance_information": [
            {"rule_id": "R-ELEC-002", "rule_status": "EFFECTIVE"},
            {"rule_id": "R-ELEC-019", "rule_status": "PROPOSED"},
        ]},
        "applicability": {"rules": [
            {"rule_id": "R-ELEC-002", "applicability_status": "APPLICABLE"},
            {"rule_id": "R-ELEC-019", "applicability_status": "REVIEW_REQUIRED"},
            {"rule_id": "R-EXTRA", "applicability_status": "NOT_APPLICABLE"},
        ]},
    }
    authorities = {rule.rule_id: rule for rule in guard.rule_authority_information(result)}
    assert authorities["R-ELEC-002"] == guard.RuleAuthority("R-ELEC-002", "EFFECTIVE", "APPLICABLE")
    assert authorities["R-ELEC-019"] == guard.RuleAuthority("R-ELEC-019", "PROPOSED", "REVIEW_REQUIRED")
    # A rule known only from the applicability verdicts is never a confirmed obligation.
    assert authorities["R-EXTRA"].is_confirmed_current_obligation is False


@pytest.mark.parametrize("lifecycle,applicability,claim", [
    ("PROPOSED", "APPLICABLE", "R-ELEC-018 is currently required for this product."),
    ("WATCHLIST", "APPLICABLE", "R-ELEC-018 is mandatory."),
    ("SUPERSEDED", "APPLICABLE", "R-ELEC-018 is in force."),
    ("UNKNOWN", "APPLICABLE", "R-ELEC-018 is required."),
    ("EFFECTIVE", "NEEDS_INFO", "R-ELEC-018 is required now."),
    ("EFFECTIVE", "NOT_APPLICABLE", "This product must comply with R-ELEC-018."),
    ("EFFECTIVE", "REVIEW_REQUIRED", "R-ELEC-018 is mandatory for this product."),
    ("EFFECTIVE", "", "R-ELEC-018 has to meet the requirement."),
])
def test_every_unconfirmed_authority_is_guarded(lifecycle: str, applicability: str, claim: str) -> None:
    """Only EFFECTIVE + APPLICABLE may be described as a current obligation."""
    authority = guard.RuleAuthority("R-ELEC-018", lifecycle, applicability)
    result = guard.guard_response_with_findings(claim, "REVIEW_REQUIRED", [authority])
    assert result.activated is True, (lifecycle, applicability, claim)
    assert NOT_CONFIRMED_MARKER in result.text
    assert lifecycle in result.text  # the canonical lifecycle is restated verbatim
    assert any(finding.pattern.startswith("unconfirmed current obligation")
               for finding in result.findings)


def test_confirmed_effective_applicable_rule_is_never_guarded() -> None:
    """Do not over-guard: a real current obligation stays explainable."""
    text = ("R-ELEC-002 is a current applicable requirement for this product. "
            "The transmitter must comply with R-ELEC-002 and it is required.")
    result = guard.guard_response_with_findings(text, "REVIEW_REQUIRED", AUTHORITIES)
    assert result.activated is False
    assert result.text == text


@pytest.mark.parametrize("claim", [
    "R-ELEC-019 is currently required for this product.",
    "The importer must comply with R-ELEC-019 now.",
    "This product has to meet R-ELEC-019.",
    "R-ELEC-018 is mandatory.",
    "R-ELEC-018 is in force.",
    # A distant negation in the same sentence must not mask a real claim.
    "R-ELEC-019 is not optional and is required.",
    "R-ELEC-019 is not a suggestion; it is mandatory.",
])
def test_promoted_non_current_rule_is_rewritten(claim: str) -> None:
    """PROPOSED / WATCHLIST material must never be presented as a current obligation."""
    result = guard.guard_response_with_findings(claim, "REVIEW_REQUIRED", LIFECYCLE)
    assert result.activated is True, claim
    assert NOT_CONFIRMED_MARKER in result.text
    assert any(finding.pattern.startswith("unconfirmed current obligation")
               for finding in result.findings)


@pytest.mark.parametrize("safe", [
    "R-ELEC-019 is proposed and is not currently effective.",
    "Monitor R-ELEC-019; it is not a current obligation.",
    "R-ELEC-018 remains on the watchlist; human review is required.",
    "R-ELEC-019 is proposed, so monitoring is required.",
    "Human review of R-ELEC-019 is required.",
])
def test_safe_lifecycle_wording_is_never_rewritten(safe: str) -> None:
    result = guard.guard_response_with_findings(safe, "REVIEW_REQUIRED", LIFECYCLE)
    assert result.activated is False, safe
    assert result.text == safe


def test_immediate_negation_is_honoured_but_a_distant_one_is_not() -> None:
    """The negation must modify the phrase; unrelated negation cannot hide a real claim."""
    assert guard.definite_current_obligation_phrases("R-ELEC-019 is not currently effective.") == ()
    assert guard.definite_current_obligation_phrases(
        "R-ELEC-019 is not optional and is required."
    ) == ("is required",)


def test_effective_rule_may_still_be_explained_as_current() -> None:
    """An EFFECTIVE + APPLICABLE canonical rule is a real current requirement: never rewritten."""
    text = ("R-ELEC-002 is required for this product and the transmitter must comply with it. "
            "R-ELEC-019 is currently required.")
    result = guard.guard_response_with_findings(text, "REVIEW_REQUIRED", LIFECYCLE)
    assert result.activated is True
    assert "R-ELEC-002 is required for this product and the transmitter must comply with it." in result.text
    assert "R-ELEC-019 is currently required." not in result.text
    assert "R-ELEC-019 is PROPOSED" in result.text


def test_effective_but_unconfirmed_applicability_is_guarded() -> None:
    """EFFECTIVE alone is not enough: without canonical APPLICABLE it is not a confirmed obligation."""
    authority = guard.RuleAuthority("R-ELEC-002", "EFFECTIVE", "NEEDS_INFO")
    result = guard.guard_response_with_findings(
        "R-ELEC-002 is required for this product.", "REVIEW_REQUIRED", [authority])
    assert result.activated is True
    assert "R-ELEC-002 is EFFECTIVE with canonical applicability NEEDS_INFO" in result.text
    assert NOT_CONFIRMED_MARKER in result.text


def test_lifecycle_guarding_is_independent_of_the_review_status() -> None:
    """A promotion is unsafe even if the review status were resolved."""
    result = guard.guard_response_with_findings("R-ELEC-019 is mandatory.", "RESOLVED", LIFECYCLE)
    assert result.activated is True
    assert NOT_CONFIRMED_MARKER in result.text


def test_lifecycle_guarding_never_mutates_the_canonical_findings() -> None:
    findings = [{"rule_id": "R-ELEC-019", "rule_status": "PROPOSED", "applicability_status": "APPLICABLE"}]
    before = [dict(item) for item in findings]
    guard.guard_response_with_findings("R-ELEC-019 is required.", "REVIEW_REQUIRED", findings)
    assert findings == before


def test_guarding_never_mutates_any_canonical_block() -> None:
    """Mutation test: the applicability-aware guard reads every canonical block and changes none."""
    canonical = {
        "classification": {"category": "small_consumer_electronics", "category_status": "CONFIRMED"},
        "applicability": {"rules": [
            {"rule_id": "R-ELEC-002", "applicability_status": "APPLICABLE", "reason_codes": ["FACT_PRESENT"]},
            {"rule_id": "R-ELEC-019", "applicability_status": "REVIEW_REQUIRED", "reason_codes": []},
        ]},
        "verified": {"compliance_information": [
            {"rule_id": "R-ELEC-002", "rule_status": "EFFECTIVE", "evidence_status": "VERIFIED"},
            {"rule_id": "R-ELEC-019", "rule_status": "PROPOSED", "evidence_status": "NOT_FOUND"},
        ]},
        "risk": {"level": "MEDIUM", "findings": []},
        "cost": {"currency": "USD", "line_items": []},
        "actions": {"recommended_actions": [], "triggers": []},
        "product_facts": {"facts": [{"attribute_id": "A-ELEC-001", "origin": "USER"}]},
        "what_if": {"deltas": []},
        "review": {"status": "REVIEW_REQUIRED", "triggers": [], "reviewer_actions": []},
    }
    before = json.dumps(canonical, sort_keys=True)
    guarded, result = guard.guard_observation_text(
        canonical, "R-ELEC-019 is currently required. It must be complied with.")
    assert result.activated is True
    guard.guard_response_with_findings("R-ELEC-019 is required.", "REVIEW_REQUIRED",
                                       guard.rule_authority_information(canonical))
    assert json.dumps(canonical, sort_keys=True) == before      # nothing written back
    assert canonical["applicability"]["rules"][1] == {
        "rule_id": "R-ELEC-019", "applicability_status": "REVIEW_REQUIRED", "reason_codes": []}
    assert canonical["verified"]["compliance_information"][1]["rule_status"] == "PROPOSED"
    assert canonical["review"]["status"] == "REVIEW_REQUIRED"
    # Only the advisory sentence changed; the confirmed rule's canonical values are restated verbatim.
    assert "R-ELEC-002" not in guarded
    assert "R-ELEC-019 is PROPOSED with canonical applicability REVIEW_REQUIRED" in guarded


def test_lifecycle_guard_keeps_the_existing_compliance_claim_protection() -> None:
    text = "The product is compliant. R-ELEC-019 is currently required."
    result = guard.guard_response_with_findings(text, "REVIEW_REQUIRED", LIFECYCLE)
    assert result.activated is True
    assert "is compliant" not in result.text.lower()
    assert "currently required" not in result.text.lower()
    assert {finding.pattern for finding in result.findings} == {
        "is compliant",
        "unconfirmed current obligation (R-ELEC-019/PROPOSED/APPLICABLE)",
    }
    assert guard.guard_summary(result).startswith("response guard: 2 ")
    assert "unconfirmed current obligation (R-ELEC-019/PROPOSED/APPLICABLE)" in guard.guard_summary(result)


def test_guarded_lifecycle_text_is_a5_safe() -> None:
    """The replacement the guard emits must itself pass the certification A5 vocabulary."""
    for claim in ("R-ELEC-019 is currently required.", "R-ELEC-018 is mandatory."):
        guarded = guard.guard_response(claim, "REVIEW_REQUIRED", LIFECYCLE)
        assert guard.has_definite_current_obligation(guarded) is False, guarded
        # The replacement states only canonical metadata; no workflow or legal duty is invented.
        assert NOT_CONFIRMED_MARKER in guarded
        for invented in ("review", "monitoring", "must apply", "still applies"):
            assert invented not in guarded.lower()


# =========================================================================== #
# 6 — narrow adjacent-reference safety
# =========================================================================== #
def test_adjacent_reference_is_guarded_within_one_sentence() -> None:
    """Regression: "R-X is proposed. It is currently required." must be caught without an LLM."""
    text = "R-ELEC-019 is proposed. It is currently required."
    result = guard.guard_response_with_findings(text, "REVIEW_REQUIRED", LIFECYCLE)
    assert result.activated is True
    assert "It is currently required." not in result.text
    assert "R-ELEC-019 is PROPOSED" in result.text
    assert result.findings[0].index == 1


def test_adjacent_reference_with_safe_wording_is_allowed() -> None:
    text = "R-ELEC-019 is proposed. It is not currently effective."
    result = guard.guard_response_with_findings(text, "REVIEW_REQUIRED", LIFECYCLE)
    assert result.activated is False
    assert result.text == text


def test_the_rule_phrasing_is_also_recognised() -> None:
    for reference in ("This rule is currently required.", "The rule is mandatory."):
        text = f"R-ELEC-019 is proposed. {reference}"
        result = guard.guard_response_with_findings(text, "REVIEW_REQUIRED", LIFECYCLE)
        assert result.activated is True, reference


def test_ambiguous_antecedent_is_never_guessed() -> None:
    """Two rules in the previous sentence: the guard must not decide what "it" refers to."""
    text = "R-ELEC-018 is on the watchlist and R-ELEC-019 is proposed. It is currently required."
    result = guard.guard_response_with_findings(text, "REVIEW_REQUIRED", LIFECYCLE)
    assert result.activated is False
    assert result.text == text


def test_references_are_not_carried_beyond_one_sentence() -> None:
    text = ("R-ELEC-019 is proposed. Risks were reviewed. "
            "It is currently required.")
    result = guard.guard_response_with_findings(text, "REVIEW_REQUIRED", LIFECYCLE)
    assert result.activated is False  # the middle sentence breaks the adjacency chain


def test_a_confirmed_rule_reference_is_not_guarded() -> None:
    text = "R-ELEC-002 applies. It is a current applicable requirement."
    result = guard.guard_response_with_findings(text, "REVIEW_REQUIRED", LIFECYCLE)
    assert result.activated is False


def test_rule_id_matching_never_uses_a_prefix() -> None:
    assert guard.mentions_rule("R-ELEC-019 applies.", "R-ELEC-02") is False
    assert guard.mentions_rule("R-ELEC-02 applies.", "R-ELEC-02") is True
