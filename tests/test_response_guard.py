"""Offline tests for the deterministic response guard (`src/agent/response_guard.py`).

Pure, offline, no provider, no credential. These tests pin the guard's contract: it activates only for an
unresolved canonical status, rewrites only the sentence that asserts a compliance conclusion, preserves
uncertainty wording and every other sentence, and never touches the canonical result.
"""

from __future__ import annotations

import inspect

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
