"""Focused tests for the consumer Web UI helper/glue layer.

These tests are pure Python: they exercise the translation dictionary, session
state helpers, presentation mapping and the engine adapter. They never render
Streamlit widgets, never touch the network and never call a provider.

They also act as objective guardrails for the UI contract:

* canonical engine results are never mutated or re-decided by the UI,
* no credential can reach a presentation payload,
* no canonical engine is duplicated or instantiated in the UI layers.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.agent.model_factory import CompatibilityStatus, resolve_combination
from src.repositories.compliance_repository import JsonComplianceRepository
from src.services.actions import ActionType
from src.services.analysis import AnalysisService
from src.services.applicability import TRIGGER_SPECS, valid_spec_rule_ids
from src.services.classification import CategorySource, CategoryStatus, allowed_category_values
from src.services.cost import CostCalculationStatus
from src.services.risk import RiskLevel
from src.state import FactOrigin, ProductFact
from src.ui import pipeline, presenters, state as ui_state
from src.ui.i18n import (
    DEFAULT_LANGUAGE,
    LANGUAGE_LABELS,
    available_languages,
    keys_for,
    normalize_language,
    t,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = PROJECT_ROOT / "src" / "ui"
FAKE_KEY = "sk-FAKE-UI-TEST-KEY-0001"
FORBIDDEN_APPROVAL_WORDING = (
    "fully compliant",
    "approved for import",
    "safe to import",
    "no compliance obligations",
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def repository() -> JsonComplianceRepository:
    return JsonComplianceRepository()


@pytest.fixture(scope="module")
def analysis_service(repository: JsonComplianceRepository) -> AnalysisService:
    return AnalysisService(repository)


@pytest.fixture(scope="module")
def electronics_category(repository: JsonComplianceRepository):
    return pipeline.category_result_for(repository, "small_consumer_electronics")


@pytest.fixture(scope="module")
def demo_facts() -> list[ProductFact]:
    """The primary demo scenario: a Bluetooth (intentional RF) device.

    All required facts of ``R-ELEC-002`` are supplied, which is what the missing
    information flow collects from the user, so the canonical applicability
    verdict (and therefore the What-if delta) is reachable.
    """
    return [
        ProductFact(attribute_id="A-ELEC-002", value=True, origin=FactOrigin.USER),
        ProductFact(
            attribute_id="A-ELEC-003",
            value=["Bluetooth 2.4 GHz"],
            origin=FactOrigin.USER,
        ),
        ProductFact(
            attribute_id="A-ELEC-007",
            value="fully certified module",
            origin=FactOrigin.USER,
        ),
        ProductFact(
            attribute_id="A-ELEC-008",
            value="FCC ID ABC123; PCB antenna; 10 mW",
            origin=FactOrigin.USER,
        ),
        ProductFact(attribute_id="A-ELEC-009", value=0, origin=FactOrigin.USER),
        ProductFact(attribute_id="A-ELEC-010", value=1, origin=FactOrigin.USER),
        ProductFact(attribute_id="A-ELEC-021", value="clear", origin=FactOrigin.USER),
    ]


@pytest.fixture(scope="module")
def demo_analysis(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
    demo_facts,
):
    outcome = pipeline.run_analysis(
        repository,
        analysis_service,
        electronics_category,
        facts=demo_facts,
        model=None,
        description="Bluetooth wireless earbuds with a rechargeable battery",
    )
    assert outcome.ok, outcome.error_code
    assert outcome.analysis is not None
    return outcome.analysis


# --------------------------------------------------------------------------- #
# i18n
# --------------------------------------------------------------------------- #
def test_translation_key_parity_between_languages() -> None:
    assert keys_for("en") == keys_for("zh")
    assert keys_for("en")


def test_language_metadata_is_fixed() -> None:
    assert normalize_language("ZH") == "zh"
    assert normalize_language("fr") == DEFAULT_LANGUAGE
    assert normalize_language(None) == DEFAULT_LANGUAGE
    assert available_languages()[0] == DEFAULT_LANGUAGE
    assert set(available_languages()) == set(LANGUAGE_LABELS)


def test_missing_translation_falls_back_safely() -> None:
    assert t("this_key_does_not_exist", "zh") == "this_key_does_not_exist"
    assert t("app_title", "zh") == "ImportReady AI"
    # An unknown language falls back to English rather than raising.
    assert t("product_description", "de") == t("product_description", "en")
    # A template with missing kwargs stays a template instead of raising.
    assert "{count}" in t("status_current_requirements", "en")


def test_canonical_identifiers_are_never_translated() -> None:
    """Fixed UI labels are translated; canonical status values pass through."""
    assert presenters.review_status_label("SOME_FUTURE_STATUS", "zh") == "SOME_FUTURE_STATUS"
    assert presenters.applicability_status_label("SOME_FUTURE_STATUS", "zh") == (
        "SOME_FUTURE_STATUS"
    )
    assert presenters.priority_label("SOME_FUTURE_PRIORITY", "zh") == "SOME_FUTURE_PRIORITY"


# --------------------------------------------------------------------------- #
# Category presentation / confirmation flow
# --------------------------------------------------------------------------- #
def test_every_canonical_category_has_a_localized_label(
    repository: JsonComplianceRepository,
) -> None:
    for category in allowed_category_values(repository):
        for lang in available_languages():
            label = presenters.category_label(category, lang)
            assert label and label != f"cat_{category}"


def test_consumer_choices_are_a_canonical_subset(
    repository: JsonComplianceRepository,
) -> None:
    allowed = set(allowed_category_values(repository))
    choices = pipeline.consumer_category_choices(repository)
    assert set(choices) <= allowed
    assert "uncertain" not in choices
    assert {"small_consumer_electronics", "childrens_toys", "dual", "unsupported"} <= set(
        choices
    )


def test_human_confirmation_is_the_only_resolved_path(
    repository: JsonComplianceRepository,
) -> None:
    resolved = pipeline.category_result_for(repository, "small_consumer_electronics")
    assert resolved.category_source is CategorySource.HUMAN_CONFIRMED
    assert resolved.category_status is CategoryStatus.RESOLVED

    unsupported = pipeline.category_result_for(repository, "unsupported")
    assert unsupported.category_status is CategoryStatus.UNSUPPORTED

    uncertain = pipeline.category_result_for(repository, "uncertain")
    assert uncertain.category_status is CategoryStatus.NEEDS_INFO


def test_no_ai_suggestion_is_fabricated_without_a_model(
    repository: JsonComplianceRepository,
) -> None:
    outcome = pipeline.suggest_category(repository, "Bluetooth earbuds", model=None)
    assert outcome.suggestion is None
    assert outcome.error_code is None
    assert outcome.ai_used is False


def test_sensitive_input_is_rejected_before_classification(
    repository: JsonComplianceRepository,
) -> None:
    outcome = pipeline.suggest_category(repository, f"earbuds {FAKE_KEY}", model=None)
    assert outcome.suggestion is None
    assert outcome.error_code == "sensitive_input"


# --------------------------------------------------------------------------- #
# Missing-information flow
# --------------------------------------------------------------------------- #
def test_explicit_answers_become_user_facts_only() -> None:
    facts = pipeline.build_facts(
        {"A-ELEC-002": True, "A-ELEC-003": None, "A-ELEC-011": "", "A-ELEC-012": []}
    )
    assert [fact.attribute_id for fact in facts] == ["A-ELEC-002"]
    assert facts[0].origin is FactOrigin.USER
    assert facts[0].value is True
    # No agent/classifier origin is ever constructed by the UI adapter.
    assert "FactOrigin.CLASSIFIER" not in (UI_DIR / "pipeline.py").read_text(encoding="utf-8")


def test_missing_information_grouping_is_complete_and_disjoint(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    """key ∪ additional == canonical missing set, and the groups never overlap."""
    partial = [ProductFact(attribute_id="A-ELEC-002", value=True, origin=FactOrigin.USER)]
    outcome = pipeline.run_analysis(
        repository, analysis_service, electronics_category, facts=partial, model=None
    )
    analysis = outcome.analysis
    assert analysis is not None

    key, additional = pipeline.missing_information_split(repository, analysis)
    canonical = [item.attribute_id for item in analysis.unknown.missing_information]

    key_ids = [item.attribute_id for item in key]
    additional_ids = [item.attribute_id for item in additional]

    assert key_ids and additional_ids
    assert set(key_ids) | set(additional_ids) == set(canonical)
    assert len(key_ids) + len(additional_ids) == len(canonical)
    assert not (set(key_ids) & set(additional_ids))
    # Both groups hold only real canonical items; "additional" keeps canonical order.
    assert all(item.attribute_id in canonical for item in key + additional)
    assert additional_ids == [aid for aid in canonical if aid in set(additional_ids)]
    assert set(additional_ids) == set(canonical) - set(key_ids)


def test_key_information_prioritizes_trigger_spec_facts_generically(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    """Derived from approved TriggerSpec/rule metadata - never from prose or a hard-code."""
    partial = [ProductFact(attribute_id="A-ELEC-002", value=True, origin=FactOrigin.USER)]
    outcome = pipeline.run_analysis(
        repository, analysis_service, electronics_category, facts=partial, model=None
    )
    analysis = outcome.analysis
    assert analysis is not None
    key, _ = pipeline.missing_information_split(repository, analysis)
    key_ids = {item.attribute_id for item in key}

    # The facts required to obtain the deterministic R-ELEC-002 verdict are in the
    # key group because R-ELEC-002 is a valid automated TriggerSpec rule here.
    r_elec_002 = repository.get_rule("R-ELEC-002")
    assert r_elec_002 is not None
    expected = set(r_elec_002.required_attribute_ids) - {"A-ELEC-002"}
    assert expected <= key_ids

    # The grouping is exactly the spec-derived rule/attribute relationship.
    valid = valid_spec_rule_ids(repository)
    derived: set[str] = set()
    for result in analysis.applicability.rules:
        if result.rule_id in valid:
            derived.update(result.required_attribute_ids)
    assert key_ids <= derived


def test_unresolved_category_puts_all_missing_items_in_the_additional_group(
    repository: JsonComplianceRepository, analysis_service: AnalysisService
) -> None:
    outcome = pipeline.run_analysis(
        repository, analysis_service, pipeline.category_result_for(repository, "uncertain")
    )
    assert outcome.analysis is not None
    key, additional = pipeline.missing_information_split(repository, outcome.analysis)
    assert key == []
    assert additional == list(outcome.analysis.unknown.missing_information)


# --------------------------------------------------------------------------- #
# Summary / risk presentation
# --------------------------------------------------------------------------- #
def test_summary_never_asserts_approval(demo_analysis) -> None:
    view = presenters.summary(demo_analysis, "REVIEW_REQUIRED", "en")
    blob = " ".join(view.lines).lower()
    for wording in FORBIDDEN_APPROVAL_WORDING:
        assert wording not in blob
    assert any("analysis completed" in line.lower() for line in view.lines)
    assert any("human review required" in line.lower() for line in view.lines)
    assert view.current_requirement_count >= 1


def test_summary_is_localized_and_mirrors_canonical_status(demo_analysis) -> None:
    en = presenters.summary(demo_analysis, "NEEDS_INFO", "en")
    zh = presenters.summary(demo_analysis, "NEEDS_INFO", "zh")
    assert en.review_status == zh.review_status == "NEEDS_INFO"
    assert en.lines != zh.lines
    assert en.category == demo_analysis.classification["category"]


@pytest.mark.parametrize("level", [level.value for level in RiskLevel])
def test_risk_level_presentation_is_fixed_and_non_alarmist(level: str) -> None:
    for lang in available_languages():
        label = presenters.risk_level_label(level, True, lang)
        assert label
        lowered = label.lower()
        for forbidden in ("danger", "illegal", "severe", "violation"):
            assert forbidden not in lowered
    assert presenters.risk_level_label(None, False, "en") == t("risk_not_assessed", "en")


def test_risk_view_uses_canonical_items_in_canonical_order(demo_analysis) -> None:
    view = presenters.risk_view(demo_analysis.risk, "en", limit=None)
    canonical = demo_analysis.risk.items
    assert [item.rule_id for item in view.items] == [item.rule_id for item in canonical]
    assert [item.level for item in view.items] == [
        item.risk_level.value for item in canonical
    ]
    assert view.level == demo_analysis.risk.level.value
    assert view.remaining == []
    # No numeric risk score is invented anywhere in the view.
    assert not hasattr(view, "score")


def test_hidden_risk_items_remain_reachable_and_complete(demo_analysis) -> None:
    """First 5 inline, every other canonical item in the renderable remainder."""
    canonical = demo_analysis.risk.items
    assert len(canonical) > 5  # the demo genuinely has a tail
    view = presenters.risk_view(demo_analysis.risk, "en", limit=5)
    assert len(view.items) == 5
    assert len(view.remaining) == len(canonical) - 5
    assert [item.rule_id for item in view.items + view.remaining] == [
        item.rule_id for item in canonical
    ]
    assert [item.level for item in view.items + view.remaining] == [
        item.risk_level.value for item in canonical
    ]
    assert t("view_remaining", "en", count=len(view.remaining))


def test_unassessed_risk_is_not_presented_as_no_risk(
    repository: JsonComplianceRepository, analysis_service: AnalysisService
) -> None:
    unresolved = pipeline.category_result_for(repository, "uncertain")
    outcome = pipeline.run_analysis(repository, analysis_service, unresolved, facts=[])
    assert outcome.analysis is not None
    view = presenters.risk_view(outcome.analysis.risk, "en")
    assert view.assessed is False
    assert view.level is None
    assert view.level_label == t("risk_not_assessed", "en")


# --------------------------------------------------------------------------- #
# Requirements
# --------------------------------------------------------------------------- #
def test_requirement_cards_quote_canonical_fields_verbatim(demo_analysis) -> None:
    cards = presenters.requirement_cards(demo_analysis, "en")
    findings = {finding.rule_id: finding for finding in demo_analysis.verified.compliance_information}
    assert cards
    for card in cards:
        finding = findings[card.rule_id]
        assert card.requirement == finding.requirement
        assert card.authority == finding.authority
        assert card.requirement_type == finding.requirement_type
        assert card.source_ids == list(finding.source_ids)
        assert card.contextual_label
    statuses = {
        result.rule_id: result for result in demo_analysis.applicability.rules
    }
    for card in cards:
        expected = statuses[card.rule_id].applicability_status.value
        assert card.applicability_status == expected


def test_applicable_requirements_contain_only_applicable_rules(demo_analysis) -> None:
    """The view is a filtered projection of canonical APPLICABLE verdicts only."""
    cards = presenters.requirement_cards(demo_analysis, "en")
    assert cards
    assert {card.applicability_status for card in cards} == {"APPLICABLE"}
    # Candidate rules of other canonical states are present in the analysis data
    # but must never be presented as applicable requirements.
    other_states = {
        result.rule_id
        for result in demo_analysis.applicability.rules
        if result.applicability_status.value != "APPLICABLE"
    }
    assert other_states
    assert not ({card.rule_id for card in cards} & other_states)
    assert "R-ELEC-002" in {card.rule_id for card in cards}


def test_review_required_rule_is_not_shown_as_an_applicable_requirement(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    """With only the RF flag answered, R-ELEC-002 is REVIEW_REQUIRED, not applicable."""
    partial = [ProductFact(attribute_id="A-ELEC-002", value=True, origin=FactOrigin.USER)]
    outcome = pipeline.run_analysis(
        repository, analysis_service, electronics_category, facts=partial, model=None
    )
    analysis = outcome.analysis
    assert analysis is not None
    statuses = {
        result.rule_id: result.applicability_status.value
        for result in analysis.applicability.rules
    }
    assert statuses["R-ELEC-002"] == "REVIEW_REQUIRED"
    assert analysis.verified.compliance_information  # candidates are retrieved
    cards = presenters.requirement_cards(analysis, "en")
    assert "R-ELEC-002" not in {card.rule_id for card in cards}
    assert all(card.applicability_status == "APPLICABLE" for card in cards)


def test_unresolved_category_has_no_applicable_requirements(
    repository: JsonComplianceRepository, analysis_service: AnalysisService
) -> None:
    outcome = pipeline.run_analysis(
        repository,
        analysis_service,
        pipeline.category_result_for(repository, "uncertain"),
        facts=[],
    )
    assert outcome.analysis is not None
    assert presenters.requirement_cards(outcome.analysis, "en") == []
    message = t("no_requirements", "en").lower()
    for forbidden in ("no requirements", "fully compliant", "safe to import"):
        assert forbidden not in message


# --------------------------------------------------------------------------- #
# Action plan presentation
# --------------------------------------------------------------------------- #
def test_action_grouping_does_not_mutate_the_canonical_plan(demo_analysis) -> None:
    plan = demo_analysis.actions
    before = plan.model_dump(mode="json")

    view = presenters.actions_view(plan, "en")
    after = plan.model_dump(mode="json")

    assert before == after
    assert {group.action_type for group in view.groups} <= {
        member.value for member in ActionType
    }
    assert sum(group.total for group in view.groups) == len(plan.items)
    shown = {item.action_id for group in view.groups for item in group.items}
    assert shown == {item.action_id for item in plan.items}
    for group in view.groups:
        assert [item.priority for item in group.items] == [
            item.priority.value
            for item in plan.items
            if item.action_type.value == group.action_type
        ]


def test_every_canonical_action_is_rendered_in_its_group(demo_analysis) -> None:
    """No permanent top-N: the union of rendered group items is the whole plan."""
    plan = demo_analysis.actions
    view = presenters.actions_view(plan, "en")
    assert sum(group.total for group in view.groups) == len(plan.items)
    assert sum(len(group.items) for group in view.groups) == len(plan.items)
    assert {
        item.action_id for group in view.groups for item in group.items
    } == {item.action_id for item in plan.items}
    # Only CURRENT_OBLIGATION opens by default; the others stay collapsed but complete.
    for group in view.groups:
        assert group.expanded_by_default is (group.action_type == "CURRENT_OBLIGATION")


# --------------------------------------------------------------------------- #
# Cost presentation
# --------------------------------------------------------------------------- #
def test_cost_presentation_preserves_statuses_and_never_totals(demo_analysis) -> None:
    view = presenters.cost_view(demo_analysis.cost, "en")
    assert view.intro
    assert not hasattr(view, "total")
    for item in view.items:
        assert item.status in {member.value for member in CostCalculationStatus}
        assert item.status_label
        if item.amount_text:
            assert item.status != CostCalculationStatus.QUOTE_REQUIRED.value
        else:
            assert item.status == CostCalculationStatus.QUOTE_REQUIRED.value
    assert view.total_unavailable_reason == demo_analysis.cost.total_unavailable_reason


def test_cost_amount_formatting_is_passthrough(repository: JsonComplianceRepository) -> None:
    """Each approved amount is formatted on its own; nothing is aggregated."""
    from src.services.cost import CostEngine

    assessment = CostEngine(repository).assess("small_consumer_electronics")
    view = presenters.cost_view(assessment, "en")
    direct = [item for item in view.items if item.status == "DIRECT"]
    assert len(direct) == sum(
        1 for record in assessment.items if record.calculation_status.value == "DIRECT"
    )
    for item in direct:
        canonical = [record for record in assessment.items if record.cost_id == item.cost_id][0]
        if canonical.exact_amount is not None:
            assert f"{canonical.exact_amount:,.2f}" in item.amount_text
    # The presenter layer never aggregates monetary values at all.
    source = (UI_DIR / "presenters.py").read_text(encoding="utf-8")
    assert "sum(" not in source


# --------------------------------------------------------------------------- #
# What-if presentation
# --------------------------------------------------------------------------- #
def test_what_if_control_attributes_come_from_approved_specs(
    repository: JsonComplianceRepository,
) -> None:
    attributes = pipeline.what_if_control_attributes(repository, "small_consumer_electronics")
    ids = [attribute.attribute_id for attribute in attributes]
    assert "A-ELEC-002" in ids  # the primary demo control
    assert "A-CMN-001" not in ids  # the category is fixed
    assert len(ids) == len(set(ids))
    valid = valid_spec_rule_ids(repository)
    declared = {
        attribute_id
        for rule_id in valid
        for attribute_id in TRIGGER_SPECS[rule_id].deciding_attribute_ids
    }
    assert set(ids) <= declared
    for attribute_id in ids:
        assert repository.get_attribute(attribute_id) is not None


def test_what_if_presentation_uses_canonical_delta(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
    demo_facts,
) -> None:
    outcome = pipeline.run_what_if(
        analysis_service,
        repository,
        electronics_category,
        demo_facts,
        {"A-ELEC-002": False},
    )
    assert outcome.ok and outcome.result is not None
    result = outcome.result
    assert result.hypothetical is True
    assert result.assessed is True

    view = presenters.what_if_view(result, "en", lambda attribute_id: attribute_id)
    assert len(view.overrides) == len(result.overrides) == 1
    assert view.overrides[0].attribute_id == "A-ELEC-002"
    assert view.overrides[0].before_text == t("answer_yes", "en")
    assert view.overrides[0].after_text == t("answer_no", "en")

    # Every displayed change is a canonical delta entry, nothing added or dropped.
    assert [change.identifier for change in view.rule_changes] == [
        change.rule_id for change in result.delta.rule_changes
    ]
    assert len(view.risk_changes) == len(result.delta.risk_changes)
    assert len(view.cost_changes) == len(result.delta.cost_changes)
    assert len(view.action_changes) == len(result.delta.action_changes)
    assert view.counts == result.delta.counts
    changed_rules = {change.identifier for change in view.rule_changes}
    assert "R-ELEC-002" in changed_rules
    assert view.no_changes is False


def test_what_if_never_mixes_hypothetical_facts_into_the_baseline(
    analysis_service: AnalysisService,
    repository: JsonComplianceRepository,
    electronics_category,
    demo_facts,
) -> None:
    original = [fact.model_dump(mode="json") for fact in demo_facts]
    outcome = pipeline.run_what_if(
        analysis_service, repository, electronics_category, demo_facts, {"A-ELEC-002": False}
    )
    assert outcome.ok
    assert [fact.model_dump(mode="json") for fact in demo_facts] == original
    assert outcome.result.after is not None
    assert outcome.result.hypothetical is True


def test_what_if_input_error_is_reported_safely(
    analysis_service: AnalysisService,
    repository: JsonComplianceRepository,
    electronics_category,
    demo_facts,
) -> None:
    outcome = pipeline.run_what_if(
        analysis_service,
        repository,
        electronics_category,
        demo_facts,
        {"A-CMN-001": "childrens_toys"},
    )
    assert outcome.ok is False
    assert outcome.result is None
    assert outcome.error_code == ui_state.ERROR_WHAT_IF_INPUT
    assert presenters.error_message(outcome.error_code, "en")


def test_what_if_requires_a_resolved_category(
    analysis_service: AnalysisService, repository: JsonComplianceRepository
) -> None:
    outcome = pipeline.run_what_if(
        analysis_service,
        repository,
        pipeline.category_result_for(repository, "uncertain"),
        [],
        {"A-ELEC-002": True},
    )
    assert outcome.ok is True
    assert outcome.result.assessed is False


# --------------------------------------------------------------------------- #
# Credential handling
# --------------------------------------------------------------------------- #
def test_credential_is_session_only_and_clearable() -> None:
    session: dict[str, Any] = {}
    ui_state.initialize_state(session)

    ui_state.set_byok_credential(session, FAKE_KEY)
    assert ui_state.has_byok_credential(session) is True
    assert ui_state.get_byok_credential(session) == FAKE_KEY

    # The key lives in exactly one session entry and nowhere else.
    holders = [key for key, value in session.items() if FAKE_KEY in str(value)]
    assert holders == [ui_state.KEY_BYOK_CREDENTIAL]

    ui_state.clear_byok_credential(session)
    assert ui_state.has_byok_credential(session) is False
    assert ui_state.get_byok_credential(session) is None
    assert not any(FAKE_KEY in str(value) for value in session.values())


def test_empty_typed_key_does_not_store_a_credential() -> None:
    session: dict[str, Any] = {}
    ui_state.initialize_state(session)
    ui_state.set_byok_credential(session, "   ")
    assert ui_state.get_byok_credential(session) is None


def test_credential_mask_is_constant_and_leaks_nothing() -> None:
    mask = ui_state.credential_mask()
    assert mask == ui_state.CREDENTIAL_MASK
    assert "sk" not in mask.lower()
    assert "0001" not in mask


def test_start_over_clears_analysis_but_keeps_language_and_key() -> None:
    session: dict[str, Any] = {}
    ui_state.initialize_state(session)
    ui_state.set_byok_credential(session, FAKE_KEY)
    session[ui_state.KEY_LANGUAGE] = "zh"
    session[ui_state.KEY_CONFIRMED_CATEGORY] = "small_consumer_electronics"
    session[ui_state.KEY_ANALYSIS] = {"anything": True}
    session[ui_state.KEY_WHAT_IF] = {"anything": True}

    ui_state.start_over(session)

    assert session[ui_state.KEY_LANGUAGE] == "zh"
    assert ui_state.get_byok_credential(session) == FAKE_KEY
    assert session[ui_state.KEY_CONFIRMED_CATEGORY] is None
    assert session[ui_state.KEY_ANALYSIS] is None
    assert session[ui_state.KEY_WHAT_IF] is None


def test_api_key_never_appears_in_presentation_serialization(
    demo_analysis,
) -> None:
    """Every presentation payload is built without session state at all."""
    session: dict[str, Any] = {}
    ui_state.initialize_state(session)
    ui_state.set_byok_credential(session, FAKE_KEY)

    payload = {
        "summary": presenters.summary(demo_analysis, "REVIEW_REQUIRED", "en").__dict__,
        "risk": presenters.risk_view(demo_analysis.risk, "en").__dict__,
        "actions": [
            group.__dict__
            for group in presenters.actions_view(demo_analysis.actions, "en").groups
        ],
        "cost": presenters.cost_view(demo_analysis.cost, "en").__dict__,
        "evidence": presenters.evidence_view(demo_analysis, "en").__dict__,
        "technical": presenters.technical_view(
            demo_analysis,
            {
                "review_status": "REVIEW_REQUIRED",
                "agent_runtime": {"mode": "offline", "status": "NOT_USED"},
                "classification_runtime": {"mode": "offline", "status": "NOT_USED"},
            },
            "en",
        ),
    }
    blob = json.dumps(payload, default=str, ensure_ascii=False)
    assert FAKE_KEY not in blob
    assert "api_key" not in blob.lower()


def test_technical_view_is_an_explicit_allowlist(demo_analysis) -> None:
    view = presenters.technical_view(demo_analysis, {"review_status": "REVIEW_REQUIRED"}, "en")
    assert set(view) == {
        "category",
        "review_status",
        "request",
        "classification_runtime",
        "ai_runtime",
        "knowledge",
        "triggers",
        "reviewer_actions",
        "applicability_counts",
        "applicability_missing_attribute_ids",
        "input_issues",
        "risk_counts",
        "cost_counts",
        "action_counts",
        "not_evaluated",
    }
    assert view["review_status"] == demo_analysis.review.status


# --------------------------------------------------------------------------- #
# Provider state
# --------------------------------------------------------------------------- #
def test_unconfigured_competition_demo_provider_is_a_safe_state(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("MODEL_ID", raising=False)
    status = ui_state.demo_provider_status()
    assert status.available is False
    assert status.reason == "not_configured"
    resolution = ui_state.resolve_competition_demo_model()
    assert resolution.ok is False
    assert resolution.model is None
    assert resolution.error_code == ui_state.ERROR_PROVIDER_NOT_CONFIGURED
    message = presenters.error_message(resolution.error_code, "en")
    assert message and "Traceback" not in message


def test_competition_demo_refuses_an_experimental_combination(monkeypatch) -> None:
    """A configured EXPERIMENTAL model must never be consumer-ready in the demo.

    ``deepseek/deepseek-flash`` is registered EXPERIMENTAL and is deliberately not
    promoted by this task; the UI boundary must refuse it on its own.
    """
    combination = resolve_combination("deepseek", "deepseek-flash")
    assert combination is not None
    assert combination.compatibility_status is CompatibilityStatus.EXPERIMENTAL
    assert combination.ui_exposed is False

    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_ID", "deepseek-flash")
    monkeypatch.setenv("API_KEY", "sk-FAKE-NOT-A-REAL-KEY-000000")

    status = ui_state.demo_provider_status()
    assert status.available is False
    assert status.reason == "not_verified"
    assert status.display_name is None

    resolution = ui_state.resolve_competition_demo_model()
    assert resolution.ok is False
    assert resolution.model is None
    assert resolution.error_code == ui_state.ERROR_PROVIDER_NOT_VERIFIED
    assert presenters.error_message(resolution.error_code, "en")
    # The registry itself is untouched by the UI gate.
    assert resolve_combination("deepseek", "deepseek-flash").compatibility_status is (
        CompatibilityStatus.EXPERIMENTAL
    )


def test_competition_demo_accepts_only_verified_combinations(monkeypatch) -> None:
    """The gate is exactly ``VERIFIED``: flipping the status in memory is enough."""
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_ID", "deepseek-flash")
    monkeypatch.setenv("API_KEY", "sk-FAKE-NOT-A-REAL-KEY-000000")
    assert ui_state.demo_provider_status().available is False

    combination = resolve_combination("deepseek", "deepseek-flash")
    assert combination is not None
    verified = replace(combination, compatibility_status=CompatibilityStatus.VERIFIED)
    index = {(verified.provider_id, verified.model_id): verified}
    monkeypatch.setattr(
        "src.ui.state.resolve_combination",
        lambda provider_id, model_id: index.get((provider_id, model_id)),
    )
    status = ui_state.demo_provider_status()
    assert status.available is True
    assert status.provider_id == "deepseek"
    assert status.reason is None


def test_unknown_demo_provider_is_not_available(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "definitely-not-a-provider")
    monkeypatch.setenv("MODEL_ID", "whatever")
    status = ui_state.demo_provider_status()
    assert status.available is False


def test_byok_resolution_respects_registry_status_semantics() -> None:
    session: dict[str, Any] = {}
    ui_state.initialize_state(session)
    options = ui_state.consumer_provider_options()
    if not options:
        resolution = ui_state.resolve_byok_model(session)
        assert resolution.ok is False
        assert resolution.error_code == ui_state.ERROR_PROVIDER_NONE_AVAILABLE
        return
    # A verified, consumer-exposed provider exists: a missing key must be
    # reported rather than silently substituted.
    resolution = ui_state.resolve_byok_model(session)
    assert resolution.ok is False
    assert resolution.error_code == ui_state.ERROR_PROVIDER_MISSING_KEY


def test_demo_and_byok_model_objects_are_never_cached() -> None:
    cache_call = re.compile(r"cache_(data|resource)\s*\(")
    for name in ("state.py", "pipeline.py", "presenters.py"):
        source = (UI_DIR / name).read_text(encoding="utf-8")
        assert cache_call.search(source) is None
    # The only cached resources are the credential-free repository/service.
    resources = (UI_DIR / "resources.py").read_text(encoding="utf-8")
    assert resources.count("st.cache_resource") == 2
    assert "AnalysisService" in resources
    for forbidden in ("KEY_BYOK", "api_key", "resolve_byok", "credential="):
        assert forbidden not in resources


# --------------------------------------------------------------------------- #
# Error presentation + UI contract scans
# --------------------------------------------------------------------------- #
def test_all_ui_error_codes_have_safe_localized_messages() -> None:
    codes = [
        ui_state.ERROR_PROVIDER_NOT_CONFIGURED,
        ui_state.ERROR_PROVIDER_NONE_AVAILABLE,
        ui_state.ERROR_PROVIDER_MISSING_KEY,
        ui_state.ERROR_PROVIDER_REJECTED,
        ui_state.ERROR_PROVIDER_UNAVAILABLE,
        ui_state.ERROR_ANALYSIS_FAILED,
        ui_state.ERROR_WHAT_IF_INPUT,
        ui_state.ERROR_WHAT_IF_CONFIGURATION,
        ui_state.ERROR_UNSUPPORTED_CATEGORY,
        "sensitive_input",
        "classification_failed",
    ]
    for code in codes:
        for lang in available_languages():
            message = presenters.error_message(code, lang)
            assert message
            for forbidden in ("Traceback", "Error:", "http", "Exception", "None"):
                assert forbidden not in message


def test_error_message_returns_none_without_a_code() -> None:
    assert presenters.error_message(None, "en") is None


def test_ui_helpers_do_not_duplicate_or_instantiate_canonical_engines() -> None:
    forbidden = (
        "RiskEngine(",
        "CostEngine(",
        "ActionEngine(",
        "ApplicabilityEngine(",
        "compare_analyses(",
        "RiskItem(",
        "CostItem(",
        "ActionItem(",
        "RiskAssessment(",
        "CostAssessment(",
        "ActionPlan(",
    )
    for name in ("pipeline.py", "presenters.py", "components.py", "state.py"):
        source = (UI_DIR / name).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{name} must not build canonical objects ({token})"


def test_ui_source_contains_no_secrets_or_dangerous_calls() -> None:
    key_shape = re.compile(r"sk-[A-Za-z0-9_\-]{12,}")
    files = sorted(UI_DIR.glob("*.py")) + [PROJECT_ROOT / "app.py"]
    for path in files:
        source = path.read_text(encoding="utf-8")
        # No .env access and no dynamic code execution anywhere in the UI.
        for forbidden in ('".env"', "'.env'", "(.env)", "load_dotenv", "dotenv", "eval(", "exec("):
            assert forbidden not in source, f"{path.name}: {forbidden}"
        # No traceback/exception plumbing is ever reachable from the UI.
        for forbidden in (
            "import traceback",
            "traceback.format_exc",
            "traceback.print_exc",
            "st.exception",
        ):
            assert forbidden not in source, f"{path.name}: {forbidden}"
        for forbidden in (
            "import logging",
            "logging.getLogger",
            "logger.",
            "logging.basicConfig",
        ):
            assert forbidden not in source, f"{path.name}: {forbidden}"
        # No hard-coded credential-shaped literal.
        assert key_shape.search(source) is None
        assert "AKIA" not in source


def test_ui_never_writes_environment_or_files() -> None:
    for path in sorted(UI_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "os.environ[" not in source
        assert "setenv" not in source
        assert "write_text" not in source
        assert "open(" not in source


def test_ui_modules_import_cleanly() -> None:
    import src.ui.app as ui_app
    import src.ui.components as ui_components
    import src.ui.resources as ui_resources

    assert callable(ui_app.main)
    assert callable(ui_components.render_results)
    assert callable(ui_resources.get_repository)


def test_every_translation_key_used_by_the_ui_exists() -> None:
    """No UI module may reference a translation key that is not in the dictionary."""
    pattern = re.compile(r"""(?<![\w.])t\(\s*["']([A-Za-z0-9_]+)["']""")
    used: set[str] = set()
    for path in sorted(UI_DIR.glob("*.py")):
        used.update(pattern.findall(path.read_text(encoding="utf-8")))
    assert used
    missing = sorted(key for key in used if key not in keys_for("en"))
    assert missing == []
    missing_zh = sorted(key for key in used if key not in keys_for("zh"))
    assert missing_zh == []


# --------------------------------------------------------------------------- #
# Local app smoke test (simulated Streamlit runtime, no network, no provider)
# --------------------------------------------------------------------------- #
def _smoke_button(app, label):
    return next(button for button in app.button if button.label == label)


def _smoke_widget(app, key):
    for collection in (
        app.radio,
        app.text_input,
        app.number_input,
        app.text_area,
        app.selectbox,
    ):
        for candidate in collection:
            if candidate.key == key:
                return candidate
    raise AssertionError(f"widget not found: {key}")


def test_consumer_app_smoke_and_primary_demo_flow() -> None:
    """Render the page and walk the primary demo scenario without a provider.

    Provider calls are not required: the deterministic engines produce the
    canonical result, and the missing-information answers come from the user.
    """
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=90)
    app.run()
    assert list(app.exception) == []

    # Required surface (section 37 checklist).
    assert any(w.label == t("product_description", "en") for w in app.text_area)
    assert _smoke_button(app, t("analyze_product", "en"))
    assert any(r.label == t("language_label", "en") for r in app.sidebar.radio)
    assert any(r.label == t("mode_label", "en") for r in app.sidebar.radio)
    assert _smoke_button(app, t("clear_key", "en"))

    # Bilingual switch changes UI-owned labels only.
    next(r for r in app.sidebar.radio if r.label == t("language_label", "en")).set_value(
        "zh"
    ).run()
    assert list(app.exception) == []
    assert _smoke_button(app, t("analyze_product", "zh"))
    next(r for r in app.sidebar.radio if r.label == t("language_label", "zh")).set_value(
        "en"
    ).run()
    assert list(app.exception) == []

    # Section 1 -> 2: description, safe no-provider state, human confirmation.
    app.text_area[0].set_value("Bluetooth wireless earbuds with a rechargeable battery").run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert list(app.exception) == []
    assert any(t("no_suggestion", "en") in element.value for element in app.info)

    next(s for s in app.selectbox if s.label == t("category_choice", "en")).set_value(
        "small_consumer_electronics"
    ).run()
    _smoke_button(app, t("confirm_and_analyze", "en")).click().run()
    assert list(app.exception) == []
    assert any(t("category_confirmed", "en") in element.value for element in app.success)

    headings = [
        element.value for element in app.markdown if isinstance(element.value, str)
    ]
    for key in (
        "summary_section",
        "top_risks",
        "applicable_requirements",
        "recommended_actions",
        "cost_references",
        "what_if_section",
    ):
        assert any(t(key, "en") in heading for heading in headings), key

    # Applicable Requirements shows confirmed-applicable rules only: with the RF
    # flag alone no verdict is APPLICABLE, so the safe wording is shown instead.
    assert any(t("no_requirements", "en") in element.value for element in app.info)
    # Risk tail is reachable behind a real control.
    expander_labels = [expander.label for expander in app.expander]
    assert any("View remaining" in label for label in expander_labels)
    assert not any("No applicable requirement" in str(element.value) for element in app.error)

    # Section 3 is progressively disclosed: key information first, the rest behind a
    # collapsed "Additional information (N)" control - both groups answerable.
    assert any("Key information" in heading for heading in headings)
    assert any("Additional information (" in label for label in expander_labels)
    key_group_widgets = [
        widget
        for widget in list(app.radio) + list(app.selectbox) + list(app.text_input)
        if str(widget.key).startswith("ir_fact_")
    ]
    assert key_group_widgets

    # Section 3: the user supplies the canonical missing facts explicitly.
    for key, value in (
        ("ir_fact_A-ELEC-002", "yes"),
        ("ir_fact_A-ELEC-003", "Bluetooth 2.4 GHz"),
        ("ir_fact_A-ELEC-007", "fully certified module"),
        ("ir_fact_A-ELEC-008", "FCC ID ABC123; PCB antenna; 10 mW"),
        ("ir_fact_A-ELEC-009", 0),
        ("ir_fact_A-ELEC-010", 1),
        ("ir_fact_A-ELEC-021", "clear"),
    ):
        _smoke_widget(app, key).set_value(value)
    app.run()
    assert list(app.exception) == []
    _smoke_button(app, t("update_analysis", "en")).click().run()
    assert list(app.exception) == []

    blob = " ".join(
        [value.value for value in app.markdown if isinstance(value.value, str)]
        + [element.value for element in app.caption]
    )
    assert "R-ELEC-002" in blob
    assert t("risk_high", "en") in blob

    # After the canonical facts are supplied, R-ELEC-002 is a confirmed applicable
    # requirement and its canonical requirement text appears in that section.
    rule = JsonComplianceRepository().get_rule("R-ELEC-002")
    assert rule is not None
    applicable_heading_index = next(
        index
        for index, value in enumerate(app.markdown)
        if isinstance(value.value, str) and t("applicable_requirements", "en") in value.value
    )
    applicable_blob = " ".join(
        value.value
        for value in app.markdown[applicable_heading_index:]
        if isinstance(value.value, str)
    )
    assert rule.requirement in applicable_blob
    assert not any(t("no_requirements", "en") in element.value for element in app.info)

    # Section 4: the approved What-if engine produces the R-ELEC-002 delta.
    _smoke_widget(app, "ir_wi_A-ELEC-002").set_value(False).run()
    assert list(app.exception) == []
    _smoke_button(app, t("run_what_if", "en")).click().run()
    assert list(app.exception) == []
    blob = " ".join(
        [value.value for value in app.markdown if isinstance(value.value, str)]
        + [element.value for element in app.caption]
    )
    assert t("hypothetical_scenario", "en") in blob
    assert t("changed_requirements", "en") in blob
    assert "R-ELEC-002" in blob
    assert "APPLICABLE → NOT_APPLICABLE" in blob

    # Reset controls keep the page consistent.
    _smoke_button(app, t("reset_what_if", "en")).click().run()
    assert list(app.exception) == []
    _smoke_button(app, t("start_over", "en")).click().run()
    assert list(app.exception) == []
    assert _smoke_button(app, t("analyze_product", "en"))
