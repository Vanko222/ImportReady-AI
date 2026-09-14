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
from src.ui import i18n, narrative, pipeline, presenters, state as ui_state
from src.ui import intake as intake_flow
from src.ui import theme as theme
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
# Missing-information mapping + localization repair
# --------------------------------------------------------------------------- #
def _rows_for(repository, analysis, lang="en"):
    return pipeline.missing_information_rows(repository, analysis, lang)


#: Approved control for each canonical data type that renders an input widget.
#: ``multi_select`` is line-based like ``structured_list`` (one explicit customer
#: value per line); no allowed value is invented or pre-selected.
_WIDGET_FOR_DATA_TYPE = {
    "boolean": "segmented_control",
    "enum": "selectbox",
    "integer": "number_input",
    "decimal": "number_input",
    "number": "number_input",
    "text": "text_input",
    "structured_text": "text_input",
    "structured_list": "text_area",
    "multi_select": "text_area",
    "date": "text_input",
}


def test_every_attribute_has_approved_english_and_chinese_question_copy(
    repository: JsonComplianceRepository,
) -> None:
    """The catalog is the resolution authority, so it must cover every attribute."""
    for attribute in repository.attributes:
        en = i18n.attribute_question_copy(attribute.attribute_id, "en")
        zh = i18n.attribute_question_copy(attribute.attribute_id, "zh")
        assert en and en.strip(), attribute.attribute_id
        assert zh and zh.strip(), attribute.attribute_id
        assert en != zh, attribute.attribute_id  # genuinely localized, never copied


def test_each_missing_attribute_gets_exactly_one_localized_row(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    """One attribute -> one row -> one widget, with that attribute's own question."""
    outcome = pipeline.run_analysis(repository, analysis_service, electronics_category, facts=[])
    analysis = outcome.analysis
    rows = _rows_for(repository, analysis)

    ids = [row.attribute_id for row in rows]
    assert len(ids) == len(set(ids)) == len(analysis.unknown.missing_information)
    # Each row carries its own attribute metadata, straight from the repository.
    for row in rows:
        attribute = repository.get_attribute(row.attribute_id)
        assert row.attribute is attribute
        assert row.data_type == attribute.data_type
        assert row.allowed_values == list(attribute.allowed_values)
        assert row.question == i18n.attribute_question_copy(row.attribute_id, "en")


def test_one_attribute_required_by_several_rules_is_asked_once(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    """Deduplication is by attribute_id, so a shared attribute never renders twice."""
    from src.services.analysis import MissingInformation

    outcome = pipeline.run_analysis(repository, analysis_service, electronics_category, facts=[])
    analysis = outcome.analysis

    # A-ELEC-002 is required by R-ELEC-002 and R-ELEC-005 in this category, and the
    # canonical layer can legitimately repeat an attribute across contributors.
    duplicating = analysis.model_copy(
        update={
            "unknown": analysis.unknown.model_copy(
                update={
                    "missing_information": list(analysis.unknown.missing_information)
                    + [MissingInformation(attribute_id="A-ELEC-002", attribute_name="intentional_rf_transmitter")]
                }
            )
        }
    )
    rows = _rows_for(repository, duplicating)
    ids = [row.attribute_id for row in rows]
    assert ids.count("A-ELEC-002") == 1

    key, additional = pipeline.missing_information_split(repository, duplicating)
    key_ids = [item.attribute_id for item in key]
    additional_ids = [item.attribute_id for item in additional]
    assert len(key_ids) == len(set(key_ids))
    assert len(additional_ids) == len(set(additional_ids))
    assert not (set(key_ids) & set(additional_ids))


def test_two_attributes_of_the_same_rule_get_their_own_questions(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    """R-ELEC-002 requires both attributes; each must be asked with its own question."""
    outcome = pipeline.run_analysis(repository, analysis_service, electronics_category, facts=[])
    rows = {row.attribute_id: row for row in _rows_for(repository, outcome.analysis)}

    rule = repository.get_rule("R-ELEC-002")
    assert {"A-ELEC-002", "A-ELEC-003"} <= set(rule.required_attribute_ids)
    assert rows["A-ELEC-002"].question != rows["A-ELEC-003"].question
    assert rows["A-ELEC-002"].question == "Does the device intentionally transmit RF?"
    assert "radio protocols and frequency bands" in rows["A-ELEC-003"].question


def test_no_rule_level_clarification_question_is_reused_as_an_attribute_prompt(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    """Regression for BUG B: a shared rule-level question is never a widget label."""
    outcome = pipeline.run_analysis(repository, analysis_service, electronics_category, facts=[])
    analysis = outcome.analysis

    shared_rule_questions = {
        (item.canonical_text or "").strip()
        for item in analysis.actions.items
        if item.action_type.value == "MISSING_INFORMATION" and (item.canonical_text or "").strip()
    }
    assert len(shared_rule_questions) > 1

    for lang in ("en", "zh"):
        rows = _rows_for(repository, analysis, lang)
        questions = [row.question for row in rows]
        assert len(questions) == len(set(questions)), "one question was reused across attributes"
        for question in questions:
            assert question not in shared_rule_questions

    # The canonical rule prose itself is intact and still available verbatim.
    rule = repository.get_rule("R-ELEC-002")
    assert rule.clarification_question in shared_rule_questions


def test_question_lookup_is_by_attribute_id_not_by_rule_id(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    """Swapping attribute ids must swap the questions, proving attribute-keyed lookup."""
    outcome = pipeline.run_analysis(repository, analysis_service, electronics_category, facts=[])
    analysis = outcome.analysis
    rf = repository.get_attribute("A-ELEC-002")
    protocols = repository.get_attribute("A-ELEC-003")
    assert pipeline.attribute_question(analysis, rf, "en") == i18n.attribute_question_copy(
        "A-ELEC-002", "en"
    )
    assert pipeline.attribute_question(analysis, protocols, "en") == i18n.attribute_question_copy(
        "A-ELEC-003", "en"
    )
    # An unknown attribute id falls back to approved metadata, never to another question.
    unknown = protocols.model_copy(update={"attribute_id": "A-XXX-999", "attribute_name": "unknown_thing"})
    fallback = pipeline.attribute_question(analysis, unknown, "en")
    assert "unknown_thing" in fallback
    assert fallback != i18n.attribute_question_copy("A-ELEC-003", "en")


def test_shared_canonical_question_is_only_used_when_it_belongs_to_one_attribute(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    """The attribute-exclusive fallback can never borrow a question another attribute owns."""
    outcome = pipeline.run_analysis(repository, analysis_service, electronics_category, facts=[])
    analysis = outcome.analysis
    for attribute in repository.attributes:
        exclusive = pipeline.canonical_attribute_question(analysis.actions, attribute.attribute_id)
        if exclusive is None:
            continue
        carriers = {
            (item.attribute_id, (item.canonical_text or "").strip())
            for item in analysis.actions.items
            if item.action_type.value == "MISSING_INFORMATION"
        }
        assert sum(1 for _, text in carriers if text == exclusive) == 1


def test_question_data_type_matches_the_attribute_and_the_widget(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    """question / attribute_id / data_type / allowed_values always describe one attribute.

    ``_WIDGET_FOR_DATA_TYPE`` is the approved control mapping; the AppTest E2E fixture
    additionally checks the real rendered widget against it.
    """
    outcome = pipeline.run_analysis(repository, analysis_service, electronics_category, facts=[])
    rows = _rows_for(repository, outcome.analysis, "zh")

    seen_types = set()
    for row in rows:
        attribute = repository.get_attribute(row.attribute_id)
        seen_types.add(row.data_type)
        assert row.data_type == attribute.data_type
        assert row.allowed_values == list(attribute.allowed_values)
        assert row.question == i18n.attribute_question_copy(row.attribute_id, "zh")
        if row.data_type == "enum":
            assert row.allowed_values, row.attribute_id

    # The scenario covers several different widget types, all under one attribute each.
    assert {"boolean", "enum", "number", "integer", "text", "structured_text"} <= seen_types
    # Every data type in the scenario now has a real input control (multi_select included).
    assert seen_types <= set(_WIDGET_FOR_DATA_TYPE)


def test_english_and_chinese_copy_are_both_rendered_from_the_same_attribute(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    outcome = pipeline.run_analysis(repository, analysis_service, electronics_category, facts=[])
    analysis = outcome.analysis
    en = {row.attribute_id: row.question for row in _rows_for(repository, analysis, "en")}
    zh = {row.attribute_id: row.question for row in _rows_for(repository, analysis, "zh")}

    assert set(en) == set(zh)
    assert en != zh
    assert en["A-ELEC-002"] == "Does the device intentionally transmit RF?"
    assert zh["A-ELEC-002"] == "该设备是否会主动发射射频（RF）信号？"
    assert en["A-ELEC-001"] == "Is this electronic device designed or marketed for children?"
    assert zh["A-ELEC-001"] == "该电子产品是否面向儿童设计或销售？"
    # Chinese questions are Chinese; canonical ids are untouched in both modes.
    for attribute_id, question in zh.items():
        assert CJK.search(question), attribute_id
        assert attribute_id not in question


def test_chinese_surface_contains_no_english_clarification_question(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    outcome = pipeline.run_analysis(repository, analysis_service, electronics_category, facts=[])
    analysis = outcome.analysis

    english_prompts = {
        i18n.attribute_question_copy(attribute.attribute_id, "en")
        for attribute in repository.attributes
    }
    rule_questions = {
        (item.canonical_text or "").strip()
        for item in analysis.actions.items
        if item.action_type.value == "MISSING_INFORMATION"
    }
    for row in _rows_for(repository, analysis, "zh"):
        assert CJK.search(row.question), row.attribute_id
        assert row.question not in english_prompts
        assert row.question not in rule_questions


def test_key_and_additional_groups_cover_every_unique_attribute_exactly_once(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    outcome = pipeline.run_analysis(repository, analysis_service, electronics_category, facts=[])
    analysis = outcome.analysis
    canonical = {item.attribute_id for item in analysis.unknown.missing_information}

    rows = _rows_for(repository, analysis, "en")
    key_ids = [row.attribute_id for row in rows if row.group == "key"]
    additional_ids = [row.attribute_id for row in rows if row.group == "additional"]

    assert len(key_ids) == len(set(key_ids))
    assert len(additional_ids) == len(set(additional_ids))
    assert not (set(key_ids) & set(additional_ids))
    assert set(key_ids) | set(additional_ids) == canonical
    assert len(rows) == len(canonical)  # nothing hidden, nothing duplicated


def test_answering_one_deduplicated_widget_creates_exactly_one_fact(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    outcome = pipeline.run_analysis(repository, analysis_service, electronics_category, facts=[])
    rows = _rows_for(repository, outcome.analysis, "en")
    answers = {row.attribute_id: None for row in rows}
    answers["A-ELEC-002"] = True

    facts = pipeline.build_facts(answers)
    assert [(fact.attribute_id, fact.value, fact.origin) for fact in facts] == [
        ("A-ELEC-002", True, FactOrigin.USER)
    ]


def test_i_dont_know_and_empty_answers_still_create_no_fact(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    outcome = pipeline.run_analysis(repository, analysis_service, electronics_category, facts=[])
    rows = _rows_for(repository, outcome.analysis, "en")
    answers = {row.attribute_id: None for row in rows}  # "I don't know" / untouched
    answers["A-ELEC-003"] = ""
    answers["A-ELEC-009"] = []
    assert pipeline.build_facts(answers) == []


def test_bluetooth_missing_facts_still_answer_r_elec_002_through_the_new_rows(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
    demo_facts,
) -> None:
    """Regression: the canonical R-ELEC-002 flow is unchanged by the mapping repair."""
    outcome = pipeline.run_analysis(
        repository, analysis_service, electronics_category, facts=demo_facts
    )
    analysis = outcome.analysis
    verified = {result.rule_id: result for result in analysis.applicability.rules}
    assert verified["R-ELEC-002"].applicability_status.value == "APPLICABLE"
    # The answered attributes therefore no longer appear as missing information.
    rows = {row.attribute_id for row in _rows_for(repository, analysis, "en")}
    assert not (set(rule.attribute_id for rule in demo_facts) & rows)


def test_toy_missing_attributes_get_their_own_localized_questions(
    repository: JsonComplianceRepository, analysis_service: AnalysisService
) -> None:
    outcome = pipeline.run_analysis(
        repository, analysis_service, pipeline.category_result_for(repository, "childrens_toys")
    )
    analysis = outcome.analysis
    assert analysis is not None and analysis.unknown.missing_information

    zh_rows = {row.attribute_id: row for row in _rows_for(repository, analysis, "zh")}
    toy_rows = {aid: row for aid, row in zh_rows.items() if aid.startswith("A-TOY-")}
    assert toy_rows
    for attribute_id, row in toy_rows.items():
        assert row.question == i18n.attribute_question_copy(attribute_id, "zh")
        assert CJK.search(row.question), attribute_id
    # Two toy attributes required by one rule are still asked separately.
    assert zh_rows["A-TOY-004"].question != zh_rows["A-TOY-005"].question
    assert zh_rows["A-TOY-004"].question == "在收到状态下，是否有部件可完全放入小零件测试筒？"


# --------------------------------------------------------------------------- #
# multi_select must be answerable
# --------------------------------------------------------------------------- #
_MULTI_SELECT_ATTRIBUTE_IDS = (
    "A-CMN-003",
    "A-CMN-004",
    "A-ELEC-006",
    "A-TOY-003",
    "A-TOY-006",
    "A-TOY-020",
    "A-TOY-024",
)


def test_the_canonical_multi_select_set_is_unchanged(
    repository: JsonComplianceRepository,
) -> None:
    """The fix is UI-only: the approved multi_select definitions are untouched."""
    actual = tuple(
        attribute.attribute_id
        for attribute in repository.attributes
        if attribute.data_type == "multi_select"
    )
    assert sorted(actual) == sorted(_MULTI_SELECT_ATTRIBUTE_IDS)
    # Closed vocabularies and the deliberately vocabulary-free ones are unchanged.
    assert repository.get_attribute("A-ELEC-006").allowed_values == [
        "battery only",
        "USB",
        "AC adapter",
        "mains",
        "operates while charging",
    ]
    assert repository.get_attribute("A-CMN-003").allowed_values == [
        "US states and territories; California flag"
    ]
    assert repository.get_attribute("A-TOY-024").allowed_values == []


def test_every_multi_select_missing_attribute_maps_to_its_own_answerable_row(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    """Each approved multi_select attribute gets its own row, question and data type."""
    for attribute_id in _MULTI_SELECT_ATTRIBUTE_IDS:
        attribute = repository.get_attribute(attribute_id)
        for lang in ("en", "zh"):
            question = pipeline.attribute_question(None, attribute, lang)
            assert question == i18n.attribute_question_copy(attribute_id, lang)
            assert "multi_select" not in question
            if lang == "zh":
                assert CJK.search(question), attribute_id

    # In the reported electronics scenario A-ELEC-006 is a real missing attribute.
    outcome = pipeline.run_analysis(repository, analysis_service, electronics_category, facts=[])
    rows = {row.attribute_id: row for row in _rows_for(repository, outcome.analysis, "en")}
    assert "A-ELEC-006" in rows
    assert rows["A-ELEC-006"].data_type == "multi_select"
    assert rows["A-ELEC-006"].question == i18n.attribute_question_copy("A-ELEC-006", "en")
    assert _WIDGET_FOR_DATA_TYPE["multi_select"] == "text_area"


def test_multi_select_answers_use_the_existing_user_fact_path(
    repository: JsonComplianceRepository,
) -> None:
    """One item -> one-member list[str]; several lines -> the explicit customer list."""
    single = pipeline.build_facts({"A-ELEC-006": ["battery only"]})
    assert [(fact.attribute_id, fact.value, fact.origin) for fact in single] == [
        ("A-ELEC-006", ["battery only"], FactOrigin.USER)
    ]
    assert isinstance(single[0].value, list)
    assert all(isinstance(member, str) for member in single[0].value)

    several = pipeline.build_facts({"A-ELEC-006": ["battery only", "operates while charging"]})
    assert several[0].value == ["battery only", "operates while charging"]
    # Explicit customer values are preserved verbatim: nothing is inferred or corrected.
    assert several[0].origin is FactOrigin.USER


def test_multi_select_empty_input_creates_no_fact(
    repository: JsonComplianceRepository,
) -> None:
    """Untouched / empty input -> None / [] -> no ProductFact, never False."""
    for value in (None, []):
        assert pipeline.build_facts({"A-ELEC-006": value}) == []
    # A row that was never rendered contributes no key at all.
    assert pipeline.build_facts({"A-ELEC-006": None, "A-CMN-004": None}) == []
    # Absence is never converted to False (the boolean rule is untouched).
    assert pipeline.build_facts({}) == []


def test_multi_select_closed_vocabulary_is_still_validated_by_the_engine(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    """The Applicability Engine remains the single validation authority."""
    invalid = ProductFact(
        attribute_id="A-ELEC-006", value=["not-an-approved-value"], origin=FactOrigin.USER
    )
    outcome = pipeline.run_analysis(
        repository, analysis_service, electronics_category, facts=[invalid]
    )
    issues = outcome.analysis.applicability.input_issues
    assert [issue.attribute_id for issue in issues] == ["A-ELEC-006"]
    assert issues[0].issue_code.value == "INVALID_VALUE"
    assert "not-an-approved-value" not in issues[0].detail  # the value is never echoed

    valid = ProductFact(
        attribute_id="A-ELEC-006", value=["battery only"], origin=FactOrigin.USER
    )
    clean = pipeline.run_analysis(
        repository, analysis_service, electronics_category, facts=[valid]
    )
    assert [issue.attribute_id for issue in clean.analysis.applicability.input_issues] == []


def test_vocabulary_free_multi_select_remains_usable(
    repository: JsonComplianceRepository, analysis_service: AnalysisService
) -> None:
    """A-CMN-003 / A-TOY-024 accept any explicit list[str], per the approved engine."""
    category = pipeline.category_result_for(repository, "childrens_toys")
    facts = [
        ProductFact(attribute_id="A-CMN-003", value=["California"], origin=FactOrigin.USER),
        ProductFact(
            attribute_id="A-TOY-024",
            value=["16 CFR part 1501", "ASTM F963"],
            origin=FactOrigin.USER,
        ),
    ]
    outcome = pipeline.run_analysis(repository, analysis_service, category, facts=facts)
    assert outcome.analysis.applicability.input_issues == []


def test_multi_select_rows_never_repeat_another_attribute_question(
    repository: JsonComplianceRepository, analysis_service: AnalysisService
) -> None:
    """No regression in the question mapping: every row still owns its question."""
    category = pipeline.category_result_for(repository, "childrens_toys")
    outcome = pipeline.run_analysis(repository, analysis_service, category, facts=[])
    for lang in ("en", "zh"):
        rows = _rows_for(repository, outcome.analysis, lang)
        questions = [row.question for row in rows]
        assert len(questions) == len(set(questions))
    multi = {
        row.attribute_id: row
        for row in _rows_for(repository, outcome.analysis, "en")
        if row.data_type == "multi_select"
    }
    assert multi
    for attribute_id, row in multi.items():
        assert row.question == i18n.attribute_question_copy(attribute_id, "en")


def test_multi_select_flow_never_mutates_the_canonical_result(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    fact = ProductFact(attribute_id="A-ELEC-006", value=["battery only"], origin=FactOrigin.USER)
    outcome = pipeline.run_analysis(
        repository, analysis_service, electronics_category, facts=[fact]
    )
    analysis = outcome.analysis
    before = json.dumps(analysis.model_dump(mode="json"), sort_keys=True, default=str)

    _rows_for(repository, analysis, "zh")
    pipeline.build_facts({"A-ELEC-006": ["battery only", "USB"]})

    assert json.dumps(analysis.model_dump(mode="json"), sort_keys=True, default=str) == before
    # The answered attribute is no longer missing information.
    assert "A-ELEC-006" not in {
        item.attribute_id for item in analysis.unknown.missing_information
    }


def test_resolving_questions_never_mutates_the_canonical_result(
    repository: JsonComplianceRepository,
    analysis_service: AnalysisService,
    electronics_category,
) -> None:
    """Presentation-only guarantee: the engines' result is byte-identical afterwards."""
    outcome = pipeline.run_analysis(repository, analysis_service, electronics_category, facts=[])
    analysis = outcome.analysis
    before = json.dumps(analysis.model_dump(mode="json"), sort_keys=True, default=str)

    pipeline.missing_information_rows(repository, analysis, "en")
    pipeline.missing_information_rows(repository, analysis, "zh")
    pipeline.missing_information_split(repository, analysis)

    assert json.dumps(analysis.model_dump(mode="json"), sort_keys=True, default=str) == before
    # The canonical partitions are derived, never written back.
    key, additional = pipeline.missing_information_split(repository, analysis)
    assert {item.attribute_id for item in key} | {
        item.attribute_id for item in additional
    } == {item.attribute_id for item in analysis.unknown.missing_information}
    assert not (
        {item.attribute_id for item in key} & {item.attribute_id for item in additional}
    )


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


def test_competition_demo_accepts_the_promoted_deepseek_v4_target(monkeypatch) -> None:
    """After the human-approved promotion the demo gate resolves exactly the certified pair."""
    combination = resolve_combination("deepseek", "deepseek-v4-flash")
    assert combination is not None
    assert combination.compatibility_status is CompatibilityStatus.VERIFIED
    assert combination.ui_exposed is True

    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_ID", "deepseek-v4-flash")
    monkeypatch.setenv("API_KEY", "sk-FAKE-NOT-A-REAL-KEY-000000")
    monkeypatch.setattr("src.ui.state.build_model", lambda: object())

    status = ui_state.demo_provider_status()
    assert status.available is True
    assert status.reason is None
    assert status.provider_id == "deepseek"

    resolution = ui_state.resolve_competition_demo_model()
    assert resolution.ok is True
    assert resolution.provider_id == "deepseek"
    assert resolution.credential_present is True
    # The registry is only read: the promoted values are what it already holds.
    assert resolve_combination("deepseek", "deepseek-v4-flash").compatibility_status is (
        CompatibilityStatus.VERIFIED
    )


def test_competition_demo_still_refuses_an_unregistered_deepseek_v4_pro(monkeypatch) -> None:
    """No sibling DeepSeek id inherits the promotion: ``deepseek-v4-pro`` stays unregistered."""
    assert resolve_combination("deepseek", "deepseek-v4-pro") is None

    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_ID", "deepseek-v4-pro")
    monkeypatch.setenv("API_KEY", "sk-FAKE-NOT-A-REAL-KEY-000000")

    status = ui_state.demo_provider_status()
    assert status.available is False
    assert status.reason == "not_configured"
    resolution = ui_state.resolve_competition_demo_model()
    assert resolution.ok is False
    assert resolution.model is None
    assert resolution.error_code == ui_state.ERROR_PROVIDER_NOT_CONFIGURED


def test_competition_demo_resolves_the_v4_target_once_verified(monkeypatch) -> None:
    """Simulated promotion: the demo path resolves exactly the certified pair."""
    combination = resolve_combination("deepseek", "deepseek-v4-flash")
    assert combination is not None
    promoted = replace(combination, compatibility_status=CompatibilityStatus.VERIFIED)
    index = {(promoted.provider_id, promoted.model_id): promoted}
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_ID", "deepseek-v4-flash")
    monkeypatch.setenv("API_KEY", "sk-FAKE-NOT-A-REAL-KEY-000000")
    monkeypatch.setattr("src.ui.state.resolve_combination", lambda p, m: index.get((p, m)))
    monkeypatch.setattr("src.ui.state.build_model", lambda: object())

    status = ui_state.demo_provider_status()
    assert status.available is True
    assert status.provider_id == "deepseek"
    resolution = ui_state.resolve_competition_demo_model()
    assert resolution.ok is True
    assert resolution.provider_id == "deepseek"


def test_byok_presents_only_the_promoted_verified_combination(monkeypatch) -> None:
    """The BYOK picker offers the certified pair and nothing freer-form."""
    promoted = {
        "provider_id": "deepseek",
        "display_name": "DeepSeek (OpenAI-compatible)",
        "model_id": "deepseek-v4-flash",
        "certification_ref": None,
    }
    monkeypatch.setattr("src.ui.state.verified_combinations", lambda: [promoted])
    captured: dict[str, object] = {}

    def fake_build(config):
        captured["config"] = config
        return object()

    monkeypatch.setattr("src.ui.state.build_model_from_runtime_config", fake_build)

    options = ui_state.consumer_provider_options()
    assert [(option.provider_id, option.model_id) for option in options] == [
        ("deepseek", "deepseek-v4-flash")
    ]

    session: dict[str, Any] = {}
    ui_state.initialize_state(session)
    ui_state.set_byok_credential(session, FAKE_KEY)
    session[ui_state.KEY_PROVIDER] = "deepseek"
    resolution = ui_state.resolve_byok_model(session)
    assert resolution.ok is True
    config = captured["config"]
    assert (config.provider_id, config.model_id) == ("deepseek", "deepseek-v4-flash")
    assert config.credential == FAKE_KEY  # request-scoped only
    assert FAKE_KEY not in repr(config)  # never serialized
    # The consumer surface offers no base URL and no free-form model id.
    assert not hasattr(config, "base_url")
    assert "base_url" not in ui_state.ProviderOption.__dataclass_fields__


def test_byok_provider_picker_uses_the_registry_verified_set(monkeypatch) -> None:
    """Rendered picker: the promoted pair is the only selectable option."""
    promoted = {
        "provider_id": "deepseek",
        "display_name": "DeepSeek (OpenAI-compatible)",
        "model_id": "deepseek-v4-flash",
        "certification_ref": None,
    }
    monkeypatch.setattr("src.ui.state.verified_combinations", lambda: [promoted])
    app = _run_app()
    next(r for r in app.radio if r.label == t("mode_label", "en")).set_value("byok").run()
    assert list(app.exception) == []
    picker = next(s for s in app.selectbox if s.label == t("provider_label", "en"))
    # The consumer sees the provider display name; the raw value stays the exact
    # registry provider id, and there is no model-id or base-URL field at all.
    assert picker.options == ["DeepSeek (OpenAI-compatible)"]
    assert picker.value == "deepseek"
    assert [option.model_id for option in ui_state.consumer_provider_options()] == [
        "deepseek-v4-flash"
    ]
    assert any(w.label == t("api_key_label", "en") for w in app.text_input)


def test_byok_picker_exposes_the_certified_pair_from_the_real_registry() -> None:
    """Post-promotion: the real registry (no stubs) drives the BYOK picker."""
    options = ui_state.consumer_provider_options()
    assert [(option.provider_id, option.model_id) for option in options] == [
        ("deepseek", "deepseek-v4-flash")
    ]
    assert options[0].display_name == "DeepSeek (OpenAI-compatible)"
    # No endpoint and no free-form model id are part of the consumer-facing option.
    assert set(ui_state.ProviderOption.__dataclass_fields__) == {
        "provider_id",
        "display_name",
        "model_id",
        "certification_ref",
    }


def test_byok_never_accepts_a_free_form_provider_or_model(monkeypatch) -> None:
    """A session-provided provider/model id can never widen the registry's exact pair."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "src.ui.state.build_model_from_runtime_config",
        lambda config: captured.setdefault("config", config) or object(),
    )
    session: dict[str, Any] = {}
    ui_state.initialize_state(session)
    ui_state.set_byok_credential(session, FAKE_KEY)

    for selected in ("acme", "deepseek-v4-pro", "", "bedrock"):
        session[ui_state.KEY_PROVIDER] = selected
        resolution = ui_state.resolve_byok_model(session)
        assert resolution.ok is True
        config = captured["config"]
        # The unknown selection falls back to the single certified pair; nothing free-form survives.
        assert (config.provider_id, config.model_id) == ("deepseek", "deepseek-v4-flash")
        assert config.credential == FAKE_KEY
        assert FAKE_KEY not in repr(config)


def test_byok_credential_field_is_masked_and_the_raw_key_never_renders() -> None:
    """BYOK guarantee: the key is entered through a password field and is never echoed."""
    components = (UI_DIR / "components.py").read_text(encoding="utf-8")
    credential_controls = components[components.index("def _render_credential_controls"):]
    assert 'type="password"' in credential_controls

    app = _run_app()
    next(r for r in app.radio if r.label == t("mode_label", "en")).set_value("byok").run()
    assert list(app.exception) == []
    app.session_state[ui_state.KEY_BYOK_CREDENTIAL] = FAKE_KEY
    app.run()
    assert list(app.exception) == []
    rendered = _all_text(app)
    assert ui_state.credential_mask() in rendered    # only the constant mask is shown
    assert FAKE_KEY not in rendered                  # the raw key is never rendered
    assert ui_state.get_byok_credential(app.session_state) == FAKE_KEY  # still session-only


def test_competition_demo_resolves_only_through_the_verified_registry_gate() -> None:
    """The real demo gate: the certified pair is accepted, an uncertified one is not."""
    import os as _os

    variables = ("MODEL_PROVIDER", "MODEL_ID", "API_KEY")
    before = {name: _os.environ.get(name) for name in variables}
    try:
        _os.environ["MODEL_PROVIDER"] = "deepseek"
        _os.environ["MODEL_ID"] = "deepseek-v4-flash"
        _os.environ["API_KEY"] = FAKE_KEY
        assert ui_state.demo_provider_status().available is True

        _os.environ["MODEL_ID"] = "deepseek-flash"          # EXPERIMENTAL sibling
        status = ui_state.demo_provider_status()
        assert status.available is False and status.reason == "not_verified"

        _os.environ["MODEL_ID"] = "deepseek-v4-pro"         # unregistered sibling
        status = ui_state.demo_provider_status()
        assert status.available is False and status.reason == "not_configured"
    finally:
        for name, value in before.items():
            if value is None:
                _os.environ.pop(name, None)
            else:
                _os.environ[name] = value


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
    for name in ("state.py", "pipeline.py", "presenters.py", "intake.py", "narrative.py"):
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
    for name in ("pipeline.py", "presenters.py", "components.py", "state.py", "intake.py",
                 "narrative.py"):
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
        app.segmented_control,
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


def _select_category(app, category: str, lang: str = "en"):
    next(s for s in app.selectbox if s.label == t("category_choice", lang)).set_value(
        category
    ).run()
    return app


def _confirm_bundle(app, lang: str = "en", category: str | None = None):
    """Walk the two human boundaries: category binding, then bundle confirmation."""
    if category is not None:
        _select_category(app, category, lang)
    assert list(app.exception) == []
    if any(b.label == t("use_selected_category", lang) for b in app.button):
        # The selected category differs from the category the bundle was read with.
        _smoke_button(app, t("use_selected_category", lang)).click().run()
        assert list(app.exception) == []
    _smoke_button(app, t("confirm_bundle", lang)).click().run()
    assert list(app.exception) == []
    return app


def test_consumer_app_smoke_and_primary_demo_flow() -> None:
    """Render the page and walk the Consumer UX v2 flow without a provider.

    Provider calls are not required: the deterministic engines produce the
    canonical result, the narrative report is composed deterministically, and the
    advanced (technical) view still exposes the canonical missing-information
    widgets - all of which live behind collapsed expanders now.
    """
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=90)
    app.run()
    assert list(app.exception) == []

    # Required surface: ONE intake box, no questionnaire.
    assert any(w.label == t("product_description", "en") for w in app.text_area)
    assert _smoke_button(app, t("analyze_product", "en"))
    assert t("intake_section", "en") in _all_text(app)
    assert any(
        c.label == t("language_label", "en") for c in app.sidebar.segmented_control
    )
    assert any(
        c.label == t("appearance_label", "en") for c in app.sidebar.segmented_control
    )
    assert any(r.label == t("mode_label", "en") for r in app.sidebar.radio)
    assert _smoke_button(app, t("clear_key", "en"))
    # Default page shows no per-attribute questionnaire.
    assert not [w for w in app.text_area if str(w.key or "").startswith("ir_fact_")]
    assert t("ai_understood_title", "en") not in _all_text(app)

    # Bilingual switch changes UI-owned labels only.
    _smoke_widget(app, ui_state.KEY_LANGUAGE_CONTROL).set_value("zh").run()
    assert list(app.exception) == []
    assert _smoke_button(app, t("analyze_product", "zh"))
    _smoke_widget(app, ui_state.KEY_LANGUAGE_CONTROL).set_value("en").run()
    assert list(app.exception) == []

    # Step 1 -> 2: description, then the AI-understood review without a provider.
    app.text_area[0].set_value("Bluetooth wireless earbuds with a rechargeable battery").run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert list(app.exception) == []
    assert t("ai_understood_title", "en") in _all_text(app)
    assert t("no_suggestion", "en") in _all_text(app)
    assert t("intake_ai_unavailable", "en") not in _all_text(app)  # no provider is not an error
    assert t("ai_understood_unknown_note", "en") in _all_text(app)

    # Step 2: explicit category + bundle confirmation (the only path to facts).
    _select_category(app, "small_consumer_electronics")
    _confirm_bundle(app)
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] == "small_consumer_electronics"

    # Step 3: the customer report is natural language, with no rule ids by default.
    headings = [
        element.value for element in app.markdown if isinstance(element.value, str)
    ]
    for key in (
        "report_section",
        "report_overall",
        "report_attention",
        "report_verify",
        "report_cost",
        "report_next_step",
        "report_limitations",
    ):
        assert any(t(key, "en") in heading for heading in headings), key
    # The default report region (report -> scenario controls) shows no canonical rule
    # id; identifiers live in Technical details, which renders after it.
    report_start = next(
        index
        for index, value in enumerate(app.markdown)
        if isinstance(value.value, str) and t("report_section", "en") in value.value
    )
    technical_start = next(
        index
        for index, value in enumerate(app.markdown)
        if index > report_start
        and isinstance(value.value, str)
        and t("summary_section", "en") in value.value
    )
    default_region = " ".join(
        value.value
        for value in app.markdown[report_start:technical_start]
        if isinstance(value.value, str)
    )
    assert not re.search(r"\bR-[A-Z]{3,4}-\d{3}\b", default_region)

    # Step 4 + 5: What-if and the technical interface are collapsed expanders.
    expander_labels = [expander.label for expander in app.expander]
    assert any(t("scenario_section", "en") in label for label in expander_labels)
    assert any(t("technical_details_label", "en") in label for label in expander_labels)

    # Technical details still exposes the canonical missing-information widgets.
    assert any(t("key_information", "en") in heading for heading in headings)
    assert any(
        label.startswith(t("additional_information", "en").split("{")[0].strip())
        for label in expander_labels
    )
    key_group_widgets = [
        widget
        for widget in list(app.segmented_control)
        + list(app.selectbox)
        + list(app.text_input)
        if str(widget.key).startswith("ir_fact_")
    ]
    assert key_group_widgets

    # The advanced view still answers the canonical missing facts explicitly.
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

    blob = _all_text(app)
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
    assert t("no_requirements", "en") not in blob
    # The narrative report now states a current applicable requirement in the
    # bounded wording (and still never claims approval).
    narrative_blob = _all_text(app)
    assert t("report_overall_attention", "en") in narrative_blob
    for forbidden in FORBIDDEN_APPROVAL_WORDING:
        assert forbidden not in narrative_blob.lower()

    # Scenario: the approved What-if engine produces the R-ELEC-002 delta.
    _smoke_widget(app, "ir_wi_A-ELEC-002").set_value(False).run()
    assert list(app.exception) == []
    _smoke_button(app, t("run_what_if", "en")).click().run()
    assert list(app.exception) == []
    blob = _all_text(app)
    assert t("scenario_summary_lead", "en") in blob
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
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] is None
    assert app.session_state[ui_state.KEY_INTAKE_CANDIDATES] == []


# --------------------------------------------------------------------------- #
# Bilingual audit (no Chinese UI copy in English mode and vice versa)
# --------------------------------------------------------------------------- #
CJK = re.compile(r"[\u4e00-\u9fff]")

#: The only CJK string allowed while English is selected: a language is always
#: shown in its own script.
ALLOWED_CJK_IN_ENGLISH = {"中文"}


def _visible_strings(app) -> list[str]:
    """Every user-visible string the Streamlit testing API exposes."""
    values: list[str] = []

    def add(element, attribute: str) -> None:
        value = getattr(element, attribute, None)
        if isinstance(value, str):
            values.append(value)

    for element in app.markdown:
        add(element, "value")
    for element in list(app.caption) + list(app.button) + list(app.expander):
        add(element, "label")
        add(element, "value")
    for collection in (
        app.radio,
        app.segmented_control,
        app.selectbox,
        app.text_area,
        app.text_input,
        app.number_input,
    ):
        for element in collection:
            add(element, "label")
            for option in getattr(element, "options", []) or []:
                values.append(str(option))
    return values


def _all_text(app) -> str:
    return " \n ".join(_visible_strings(app))


def _run_app():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=90)
    app.run()
    assert list(app.exception) == []
    return app


def test_default_language_and_theme_are_english_and_light() -> None:
    app = _run_app()
    assert app.session_state[ui_state.KEY_LANGUAGE] == "en"
    assert app.session_state[ui_state.KEY_THEME] == "light"
    language_control = _smoke_widget(app, ui_state.KEY_LANGUAGE_CONTROL)
    assert language_control.value == "en"
    assert _smoke_widget(app, ui_state.KEY_THEME_CONTROL).value == "light"
    assert t("app_title", "en") in _all_text(app)


def test_english_mode_shows_no_chinese_ui_copy() -> None:
    """The reported bug: English selected must not render Chinese UI copy."""
    app = _run_app()
    offenders = [
        value
        for value in _visible_strings(app)
        if CJK.search(value) and value.strip() not in ALLOWED_CJK_IN_ENGLISH
    ]
    assert offenders == []

    # ... and the same after the full product flow, including empty/error states.
    app.text_area[0].set_value("Bluetooth wireless earbuds").run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    _smoke_button(app, t("analyze_product", "en")).click().run()  # empty-description state
    assert list(app.exception) == []
    offenders = [
        value
        for value in _visible_strings(app)
        if CJK.search(value) and value.strip() not in ALLOWED_CJK_IN_ENGLISH
    ]
    assert offenders == []


def test_chinese_mode_translates_the_main_surface() -> None:
    app = _run_app()
    _smoke_widget(app, ui_state.KEY_LANGUAGE_CONTROL).set_value("zh").run()
    assert list(app.exception) == []
    assert app.session_state[ui_state.KEY_LANGUAGE] == "zh"

    text = _all_text(app)
    for key in (
        "app_subtitle",
        "intake_section",
        "product_description",
        "analyze_product",
        "language_label",
        "appearance_label",
        "settings_title",
        "start_over",
        "clear_key",
    ):
        assert t(key, "zh") in text, key
    # The English UI copy is gone (brand and canonical content stay untouched).
    assert t("product_description", "en") not in text
    assert t("analyze_product", "en") not in text
    assert "ImportReady AI" in text


def test_missing_information_e2e_fixture_chinese_bluetooth_earbuds() -> None:
    """E2E regression for the reported bug, in Chinese mode.

    Product: "Bluetooth wireless earbuds with a rechargeable lithium-ion battery and
    a USB-C charging case." Asserts: one widget per unique attribute, each with its
    own Chinese question matching its data type, no question reused across
    attributes, and canonical identifiers untouched.
    """
    app = _run_app()
    _smoke_widget(app, ui_state.KEY_LANGUAGE_CONTROL).set_value("zh").run()
    assert list(app.exception) == []

    app.text_area[0].set_value(
        "Bluetooth wireless earbuds with a rechargeable lithium-ion battery "
        "and a USB-C charging case."
    ).run()
    _smoke_button(app, t("analyze_product", "zh")).click().run()
    assert list(app.exception) == []
    _confirm_bundle(app, "zh", category="small_consumer_electronics")

    from src.services.analysis import AnalysisResult

    analysis = AnalysisResult.model_validate(app.session_state[ui_state.KEY_ANALYSIS])
    repo = JsonComplianceRepository()
    rows = pipeline.missing_information_rows(repo, analysis, "zh")
    rows_by_id = {row.attribute_id: row for row in rows}
    assert rows_by_id

    # One canonical attribute -> exactly one rendered widget, of the matching type.
    rendered: dict[str, tuple[str, str]] = {}
    for widget_type, collection in (
        ("segmented_control", app.segmented_control),
        ("selectbox", app.selectbox),
        ("number_input", app.number_input),
        ("text_input", app.text_input),
        ("text_area", app.text_area),
    ):
        for widget in collection:
            key = str(widget.key or "")
            if not key.startswith("ir_fact_"):
                continue
            attribute_id = key[len("ir_fact_"):]
            assert attribute_id not in rendered, f"{attribute_id} rendered twice"
            rendered[attribute_id] = (widget_type, str(widget.label))

    # Every widget rendered corresponds to a resolved row (and vice versa).
    assert set(rendered) <= set(rows_by_id)
    for attribute_id, (widget_type, label) in rendered.items():
        row = rows_by_id[attribute_id]
        assert label == row.question, attribute_id
        assert CJK.search(label), attribute_id
        expected = _WIDGET_FOR_DATA_TYPE.get(row.data_type)
        if expected is not None:
            assert widget_type == expected, (attribute_id, row.data_type, widget_type)

    # The reported RF and child-directed attributes each have their own Chinese prompt.
    assert rendered["A-ELEC-002"][1] == "该设备是否会主动发射射频（RF）信号？"
    assert rendered["A-ELEC-001"][1] == "该电子产品是否面向儿童设计或销售？"
    assert rendered["A-ELEC-002"][1] != rendered["A-ELEC-001"][1]

    # No duplicate question text is attached to different attributes or data types.
    labels: dict[str, set[str]] = {}
    for attribute_id, (_, label) in rendered.items():
        labels.setdefault(label, set()).add(attribute_id)
    reused = {label: ids for label, ids in labels.items() if len(ids) > 1}
    assert reused == {}

    # The reported rule-level RF question is never used as a widget prompt.
    rf_rule_question = repo.get_rule("R-ELEC-002").clarification_question
    assert rf_rule_question not in labels
    child_rule_question = repo.get_rule("R-ELEC-001").clarification_question
    assert child_rule_question not in labels

    # Canonical identifiers stay verbatim (widget keys, enum values, rule ids).
    assert "ir_fact_A-ELEC-002" in {f"ir_fact_{aid}" for aid in rendered}
    protocol_widget = next(
        widget for widget in app.text_area if widget.key == "ir_fact_A-ELEC-003"
    )
    assert protocol_widget.label == rows_by_id["A-ELEC-003"].question
    if "ir_fact_A-ELEC-021" in rendered:
        coverage = next(w for w in app.selectbox if w.key == "ir_fact_A-ELEC-021")
        assert "clear" in [str(option) for option in coverage.options]  # canonical enum value

    # --- multi_select is a real editable control, not a caption-only row ---------
    assert rows_by_id["A-ELEC-006"].data_type == "multi_select"
    assert rendered["A-ELEC-006"][0] == "text_area", rendered["A-ELEC-006"]
    assert rendered["A-ELEC-006"][1] == rows_by_id["A-ELEC-006"].question
    assert "multi_select" not in _all_text(app)          # no "— multi_select" leakage
    multi_widget = next(w for w in app.text_area if w.key == "ir_fact_A-ELEC-006")
    assert multi_widget.label == rows_by_id["A-ELEC-006"].question
    assert multi_widget.value in (None, "")

    # Untouched / empty input means "no answer": None and no ProductFact at all.
    _smoke_button(app, t("update_analysis", "zh")).click().run()
    assert list(app.exception) == []
    stored = app.session_state[ui_state.KEY_FACT_ANSWERS]
    assert stored["A-ELEC-006"] is None
    assert [value for value in stored.values() if value is not None] == []
    after_empty = AnalysisResult.model_validate(app.session_state[ui_state.KEY_ANALYSIS])
    assert "A-ELEC-006" in {
        item.attribute_id for item in after_empty.unknown.missing_information
    }

    # Blank lines and surrounding whitespace are ignored; explicit values are kept.
    _smoke_widget(app, "ir_fact_A-ELEC-006").set_value("\n  battery only  \n\n USB \n\n")
    app.run()
    assert list(app.exception) == []
    _smoke_button(app, t("update_analysis", "zh")).click().run()
    assert list(app.exception) == []
    stored = app.session_state[ui_state.KEY_FACT_ANSWERS]
    assert stored["A-ELEC-006"] == ["battery only", "USB"]
    assert isinstance(stored["A-ELEC-006"], list)
    assert all(isinstance(member, str) for member in stored["A-ELEC-006"])
    assert [key for key, value in stored.items() if value is not None] == ["A-ELEC-006"]

    # The canonical engine accepted exactly that list (no issue for this attribute),
    # and the answered attribute left the missing set.
    refreshed = AnalysisResult.model_validate(app.session_state[ui_state.KEY_ANALYSIS])
    assert [
        issue.attribute_id
        for issue in refreshed.applicability.input_issues
        if issue.attribute_id == "A-ELEC-006"
    ] == []
    assert "A-ELEC-006" not in {
        item.attribute_id for item in refreshed.unknown.missing_information
    }

    # Answering a second, unrelated widget adds exactly one more fact.
    _smoke_widget(app, "ir_fact_A-ELEC-002").set_value("yes")
    app.run()
    assert list(app.exception) == []
    _smoke_button(app, t("update_analysis", "zh")).click().run()
    assert list(app.exception) == []
    stored = app.session_state[ui_state.KEY_FACT_ANSWERS]
    assert stored["A-ELEC-002"] is True
    assert {key for key, value in stored.items() if value is not None} == {
        "A-ELEC-002",
        "A-ELEC-006",
    }

    # Switching back to English renders the English prompt for the same attributes.
    _smoke_widget(app, ui_state.KEY_LANGUAGE_CONTROL).set_value("en").run()
    assert list(app.exception) == []
    english_labels = {
        str(widget.key)[len("ir_fact_"):]: str(widget.label)
        for collection in (app.segmented_control, app.selectbox, app.number_input,
                           app.text_input, app.text_area)
        for widget in collection
        if str(widget.key or "").startswith("ir_fact_")
    }
    # The answered attribute is resolved and no longer asked; the others keep their
    # own English prompts, with no Chinese and no reused text.
    assert "A-ELEC-002" not in english_labels
    assert english_labels["A-ELEC-003"] == "Which radio protocols and frequency bands does the device use?"
    assert english_labels["A-ELEC-001"] == (
        "Is this electronic device designed or marketed for children?"
    )
    assert len(english_labels.values()) == len(set(english_labels.values()))
    for attribute_id, label in english_labels.items():
        assert not CJK.search(label), attribute_id


def test_language_switch_preserves_analysis_facts_and_credential() -> None:
    app = _run_app()
    app.session_state[ui_state.KEY_BYOK_CREDENTIAL] = FAKE_KEY
    app.text_area[0].set_value("Bluetooth wireless earbuds with a battery").run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    _confirm_bundle(app, "en", category="small_consumer_electronics")
    assert list(app.exception) == []

    analysis_before = json.dumps(app.session_state[ui_state.KEY_ANALYSIS], default=str)
    facts_before = dict(app.session_state[ui_state.KEY_FACT_ANSWERS])

    _smoke_widget(app, ui_state.KEY_LANGUAGE_CONTROL).set_value("zh").run()
    assert list(app.exception) == []
    assert json.dumps(app.session_state[ui_state.KEY_ANALYSIS], default=str) == analysis_before
    assert dict(app.session_state[ui_state.KEY_FACT_ANSWERS]) == facts_before
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] == "small_consumer_electronics"
    assert ui_state.get_byok_credential(app.session_state) == FAKE_KEY
    assert FAKE_KEY not in _all_text(app)
    assert ui_state.credential_mask() in _all_text(app)


def test_missing_information_e2e_fixture_toy_multi_select_answered() -> None:
    """Toy flow: every multi_select row that is asked is answerable with a list[str]."""
    app = _run_app()
    _smoke_widget(app, ui_state.KEY_LANGUAGE_CONTROL).set_value("zh").run()
    app.text_area[0].set_value("Wooden stacking toy for toddlers").run()
    _smoke_button(app, t("analyze_product", "zh")).click().run()
    assert list(app.exception) == []
    next(s for s in app.selectbox if s.label == t("category_choice", "zh")).set_value(
        "childrens_toys"
    ).run()
    _confirm_bundle(app, "zh")

    from src.services.analysis import AnalysisResult

    analysis = AnalysisResult.model_validate(app.session_state[ui_state.KEY_ANALYSIS])
    repo = JsonComplianceRepository()
    rows_by_id = {
        row.attribute_id: row
        for row in pipeline.missing_information_rows(repo, analysis, "zh")
    }
    asked_multi = {
        attribute_id
        for attribute_id in ("A-TOY-003", "A-TOY-006", "A-TOY-020", "A-TOY-024")
        if attribute_id in rows_by_id
    }
    assert asked_multi, "the toy scenario should ask at least one multi_select attribute"

    rendered = {
        str(widget.key)[len("ir_fact_"):]: widget
        for widget in app.text_area
        if str(widget.key or "").startswith("ir_fact_")
    }
    assert asked_multi <= set(rendered), (asked_multi, set(rendered))
    for attribute_id in sorted(asked_multi):
        row = rows_by_id[attribute_id]
        assert row.data_type == "multi_select"
        assert rendered[attribute_id].label == row.question
        assert CJK.search(row.question)

    # Enter one approved value per requested attribute (verbatim, never auto-selected).
    answers: dict[str, list[str]] = {}
    for attribute_id in sorted(asked_multi):
        allowed = rows_by_id[attribute_id].allowed_values
        member = allowed[0] if allowed else "customer value one"
        answers[attribute_id] = [member, f"{member} follow-up"]
        _smoke_widget(app, f"ir_fact_{attribute_id}").set_value(
            f"\n{member}\n\n{member} follow-up\n"
        )
    app.run()
    assert list(app.exception) == []
    _smoke_button(app, t("update_analysis", "zh")).click().run()
    assert list(app.exception) == []

    stored = app.session_state[ui_state.KEY_FACT_ANSWERS]
    assert {key for key, value in stored.items() if value is not None} == asked_multi
    for attribute_id, expected in answers.items():
        assert stored[attribute_id] == expected, attribute_id

    refreshed = AnalysisResult.model_validate(app.session_state[ui_state.KEY_ANALYSIS])
    # Vocabulary-free attributes are accepted as entered; closed ones are checked by the
    # engine only (the UI never corrects a value).
    assert "A-TOY-024" not in {
        issue.attribute_id for issue in refreshed.applicability.input_issues
    } or "A-TOY-024" not in asked_multi


# --------------------------------------------------------------------------- #
# Light / dark theme
# --------------------------------------------------------------------------- #
def test_theme_tokens_are_complete_and_distinct() -> None:
    names = theme.token_names()
    assert "background" in names and "sidebar_background" in names
    assert len(names) == len(set(names))
    for name in names:
        light = theme.TOKENS["light"][name]
        dark = theme.TOKENS["dark"][name]
        assert light and dark
    assert theme.TOKENS["light"]["background"] != theme.TOKENS["dark"]["background"]
    assert theme.TOKENS["light"]["text_primary"] != theme.TOKENS["dark"]["text_primary"]


def test_theme_css_is_token_driven_with_no_stray_literal_colors() -> None:
    for name in theme.THEMES:
        css = theme.theme_css(name)
        assert "--ir-background:" in css
        assert "var(--ir-background)" in css
        # Every hex literal is a token definition, never an inline component color.
        assert len(re.findall(r"#[0-9A-Fa-f]{6}", css)) == len(theme.token_names())
        # Both themes emit the same structural style sheet.
        assert css.count("ir-note--error") == 1


def test_theme_css_differs_between_light_and_dark() -> None:
    light = theme.theme_css("light")
    dark = theme.theme_css("dark")
    assert light != dark
    assert theme.TOKENS["dark"]["background"] in dark
    assert theme.TOKENS["dark"]["background"] not in light


def test_theme_normalization_defaults_to_light() -> None:
    assert theme.normalize_theme("DARK") == "dark"
    assert theme.normalize_theme("dark") == "dark"
    assert theme.normalize_theme("solarized") == theme.DEFAULT_THEME
    assert theme.normalize_theme(None) == "light"


# --------------------------------------------------------------------------- #
# Final minor visual polish (light primary-button contrast + light hierarchy)
# --------------------------------------------------------------------------- #
def _relative_luminance(color: str) -> float:
    """WCAG relative luminance for a ``#RRGGBB`` value."""
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]

    def linear(value: float) -> float:
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (linear(value) for value in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(foreground: str, background: str) -> float:
    light, dark = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)), reverse=True
    )
    return (light + 0.05) / (dark + 0.05)


def test_light_primary_button_uses_high_contrast_foreground() -> None:
    """Light primary buttons must be white-on-blue, including hover and focus."""
    light = theme.TOKENS["light"]
    ratio = _contrast_ratio(light["accent_text"], light["accent"])
    assert ratio >= 4.5, ratio
    assert _contrast_ratio(light["accent_text"], light["accent_hover"]) >= 4.5

    css = theme.theme_css("light")
    # Explicit foreground for primary buttons, on the button and on its label
    # (Streamlit renders the label inside a markdown container).
    assert 'button[kind="primary"] :is(p, div, span)' in css
    for key in (
        "ir_analyze",
        "ir_confirm_category",
        "ir_update_analysis",
        "ir_run_what_if",
    ):
        assert f".st-key-{key} button :is(p, div, span)" in css
        assert f".st-key-{key} button:hover :is(p, div, span)" in css
        assert f".st-key-{key} button:focus :is(p, div, span)" in css
    # The button label never inherits the body text colour.
    assert (
        '[data-testid="stButton"] button :is(p, div, span),\n'
        '[data-testid="stFormSubmitButton"] button :is(p, div, span) { color: inherit; }'
        in css
    )


def test_dark_primary_button_stays_readable() -> None:
    """Dark passed visual review: its resting primary button stays AA-readable.

    The dark hover shade measures 4.09:1 with white text (just under the 4.5 AA
    normal-text threshold, above the 3:1 large-text/UI-component threshold). Raising
    it would mean changing an approved dark token, which is outside this light-only
    polish task, so it is asserted at the UI-component threshold and reported to the
    human reviewer instead of being silently retuned.
    """
    dark = theme.TOKENS["dark"]
    assert _contrast_ratio(dark["accent_text"], dark["accent"]) >= 4.5
    assert _contrast_ratio(dark["accent_text"], dark["accent_hover"]) >= 3.0
    # Dark still uses the reviewed accent pair (no light value leaked in).
    assert dark["accent"] == "#2E6EA8"
    assert dark["accent_hover"] == "#3C82BF"


def test_light_theme_hierarchy_tokens_are_more_defined() -> None:
    """Light surfaces/borders are more defined but still subtle and unsaturating."""
    light = theme.TOKENS["light"]

    # Borders are visible against the page, yet stay light (no dark borders).
    border_ratio = _contrast_ratio(light["border"], light["background"])
    assert 1.1 <= border_ratio <= 2.0, border_ratio
    assert _contrast_ratio(light["border_strong"], light["background"]) > border_ratio
    assert _relative_luminance(light["border"]) > 0.5

    # Selected control surface is clearly distinct from page and unselected control.
    assert _contrast_ratio(light["accent_selected"], light["background"]) > 1.1
    assert light["accent_selected"] != light["surface"]
    assert light["accent_selected"] != light["accent_soft"]
    assert _relative_luminance(light["accent_selected"]) > 0.5  # still a light tint

    # Section-level text keeps a very high contrast ratio.
    assert _contrast_ratio(light["text_primary"], light["background"]) >= 7.0
    assert _contrast_ratio(light["text_muted"], light["background"]) >= 4.5


def test_selected_segmented_control_styling_exists() -> None:
    for name in theme.THEMES:
        css = theme.theme_css(name)
        assert '[data-testid="stButtonGroup"] button[aria-checked="true"]' in css
        assert "background: var(--ir-accent-selected)" in css
        assert "border-color: var(--ir-accent)" in css
    # The definition ring is a light-only refinement: dark renders as reviewed.
    assert "inset 0 0 0 1px var(--ir-accent)" in theme.theme_css("light")
    assert "box-shadow" not in theme.theme_css("dark")
    assert "box-shadow" not in theme._STYLESHEET


def test_four_approved_visual_states_render_cleanly() -> None:
    """English/中文 × Light/Dark: every reviewed combination still renders."""
    for language in ("en", "zh"):
        for appearance in ("light", "dark"):
            app = _run_app()
            _smoke_widget(app, ui_state.KEY_LANGUAGE_CONTROL).set_value(language).run()
            _smoke_widget(app, ui_state.KEY_THEME_CONTROL).set_value(appearance).run()
            assert list(app.exception) == [], (language, appearance)
            assert app.session_state[ui_state.KEY_LANGUAGE] == language
            assert app.session_state[ui_state.KEY_THEME] == appearance
            assert _smoke_button(app, t("analyze_product", language))
            assert t("product_description", language) in _all_text(app)
            if language == "en":
                offenders = [
                    value
                    for value in _visible_strings(app)
                    if CJK.search(value) and value.strip() not in ALLOWED_CJK_IN_ENGLISH
                ]
                assert offenders == []
            # The selected theme's own palette is the one that would be injected.
            assert theme.TOKENS[appearance]["background"] in theme.theme_css(appearance)
            other = "dark" if appearance == "light" else "light"
            assert theme.TOKENS[other]["background"] not in theme.theme_css(appearance)


def test_dark_theme_tokens_are_unchanged_by_the_light_polish() -> None:
    """Dark passed visual review; the light polish must not touch its palette."""
    expected_dark = {
        "background": "#101722",
        "surface": "#17202D",
        "surface_alt": "#1D2634",
        "surface_elevated": "#1B2533",
        "text_primary": "#E8EDF4",
        "text_secondary": "#BAC5D3",
        "text_muted": "#97A5B7",
        "border": "#2B3646",
        "border_strong": "#3A475A",
        "accent": "#2E6EA8",
        "accent_hover": "#3C82BF",
        "accent_soft": "#1B2B3D",
        "accent_text": "#FFFFFF",
        "input_background": "#131C28",
        "sidebar_background": "#0C131C",
    }
    for token, value in expected_dark.items():
        assert theme.TOKENS["dark"][token] == value, token
    # The parity-only selected token keeps the reviewed dark selected surface.
    assert theme.TOKENS["dark"]["accent_selected"] == theme.TOKENS["dark"]["accent_soft"]
    # No light value leaked into the dark palette (white-on-accent is shared by design).
    for token in theme.token_names():
        if token == "accent_text":
            assert theme.TOKENS["dark"][token] == "#FFFFFF"
            continue
        assert theme.TOKENS["dark"][token] != theme.TOKENS["light"][token], token


def test_main_content_width_targets_a_wide_desktop() -> None:
    width = int(theme.LAYOUT["content_width"].replace("px", ""))
    assert 1050 <= width <= 1200
    assert "max-width: var(--ir-content-width)" in theme.theme_css("light")


def test_theme_switch_dark_and_back() -> None:
    app = _run_app()
    _smoke_widget(app, ui_state.KEY_THEME_CONTROL).set_value("dark").run()
    assert list(app.exception) == []
    assert app.session_state[ui_state.KEY_THEME] == "dark"
    assert _smoke_widget(app, ui_state.KEY_THEME_CONTROL).value == "dark"

    _smoke_widget(app, ui_state.KEY_THEME_CONTROL).set_value("light").run()
    assert list(app.exception) == []
    assert app.session_state[ui_state.KEY_THEME] == "light"


def test_theme_switch_preserves_product_state_and_credential() -> None:
    app = _run_app()
    app.session_state[ui_state.KEY_BYOK_CREDENTIAL] = FAKE_KEY
    app.text_area[0].set_value("Bluetooth wireless earbuds with a battery").run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    _confirm_bundle(app, "en", category="small_consumer_electronics")
    analysis_before = json.dumps(app.session_state[ui_state.KEY_ANALYSIS], default=str)

    _smoke_widget(app, ui_state.KEY_THEME_CONTROL).set_value("dark").run()
    assert list(app.exception) == []
    assert json.dumps(app.session_state[ui_state.KEY_ANALYSIS], default=str) == analysis_before
    assert app.session_state[ui_state.KEY_DESCRIPTION]
    assert ui_state.get_byok_credential(app.session_state) == FAKE_KEY
    assert FAKE_KEY not in _all_text(app)


def test_start_over_preserves_theme_and_language() -> None:
    app = _run_app()
    _smoke_widget(app, ui_state.KEY_THEME_CONTROL).set_value("dark").run()
    _smoke_widget(app, ui_state.KEY_LANGUAGE_CONTROL).set_value("zh").run()
    app.session_state[ui_state.KEY_BYOK_CREDENTIAL] = FAKE_KEY
    app.text_area[0].set_value("Bluetooth wireless earbuds").run()
    _smoke_button(app, t("analyze_product", "zh")).click().run()
    assert list(app.exception) == []

    app.session_state[ui_state.KEY_BYOK_CREDENTIAL] = FAKE_KEY
    _smoke_button(app, t("start_over", "zh")).click().run()
    assert list(app.exception) == []
    assert app.session_state[ui_state.KEY_THEME] == "dark"
    assert app.session_state[ui_state.KEY_LANGUAGE] == "zh"
    assert app.session_state[ui_state.KEY_ANALYSIS] is None
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] is None
    assert ui_state.get_byok_credential(app.session_state) == FAKE_KEY


def test_header_brand_copy_is_translated_in_both_languages() -> None:
    for key in ("app_subtitle", "app_tagline", "disclaimer"):
        assert t(key, "en") and t(key, "zh")
        assert t(key, "en") != t(key, "zh")
    assert "ImportReady AI" in t("app_title", "en") == t("app_title", "zh")


def test_generic_and_provider_error_messages_are_localized() -> None:
    # An unknown value is not an error code: callers fall back to their own label.
    assert presenters.error_message("unknown_error_code", "en") is None
    assert presenters.error_message(ui_state.ERROR_ANALYSIS_FAILED, "en") == t(
        "err_analysis_failed", "en"
    )
    assert t("err_generic", "en") == "Something went wrong. Please try again."
    assert t("err_generic", "zh") == "操作失败，请重试。"
    assert CJK.search(presenters.error_message("analysis_failed", "zh") or "")
    assert not CJK.search(presenters.error_message("analysis_failed", "en") or "")


def test_unavailable_provider_and_empty_results_are_neutral_not_errors() -> None:
    app = _run_app()
    # No native error surface before the user has done anything.
    assert list(app.error) == []
    assert list(app.warning) == []
    text = _all_text(app)
    # The demo provider state is always a calm note: unconfigured, or configured
    # but not certified for consumer use (never an error banner).
    assert (
        t("demo_provider_not_configured", "en") in text
        or t("demo_provider_not_verified", "en") in text
    )
    # Before any intake there is no report and no questionnaire: just the intake box.
    assert t("report_section", "en") not in text
    assert not [w for w in app.text_area if str(w.key or "").startswith("ir_fact_")]

    # An empty submission is an informational note, never a blocking error page.
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert list(app.exception) == []
    assert list(app.error) == []
    assert t("intake_no_text", "en") in _all_text(app)


def test_ui_modules_other_than_i18n_contain_no_hard_coded_chinese() -> None:
    """UI-owned Chinese lives only in the central translation dictionary."""
    for path in sorted(UI_DIR.glob("*.py")):
        if path.name == "i18n.py":
            continue
        source = path.read_text(encoding="utf-8")
        assert not CJK.search(source), f"{path.name} contains hard-coded Chinese copy"


# =========================================================================== #
# Consumer UX v2 - intake authority
# =========================================================================== #
INTAKE_DESCRIPTION = (
    "Bluetooth wireless earbuds with a rechargeable lithium-ion battery and a "
    "USB-C charging case. Sold on Amazon US. The supplier says FCC testing was "
    "performed but I do not currently have the test report."
)


def _intake_session() -> dict[str, Any]:
    session: dict[str, Any] = {}
    ui_state.initialize_state(session)
    intake_flow.initialize_intake_state(session)
    session[ui_state.KEY_DESCRIPTION] = INTAKE_DESCRIPTION
    return session


def _accept(repository, description, category, raw):
    from src.agent.intake import validate_candidates

    accepted, dropped = validate_candidates(repository, description, category, raw)
    return accepted, dropped


def test_default_page_has_one_primary_natural_language_intake_area() -> None:
    app = _run_app()
    intake_boxes = [
        widget for widget in app.text_area if widget.key == ui_state.KEY_DESCRIPTION
    ]
    assert len(intake_boxes) == 1
    assert t("intake_section", "en") in _all_text(app)
    assert t("intake_placeholder", "en") in intake_boxes[0].placeholder


def test_no_giant_missing_information_questionnaire_in_the_default_flow() -> None:
    """Neither before nor after Analyze does the default page render 40+ widgets."""
    app = _run_app()
    app.text_area[0].set_value(INTAKE_DESCRIPTION).run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert list(app.exception) == []
    assert t("ai_understood_title", "en") in _all_text(app)
    attribute_widgets = [
        widget
        for collection in (
            app.text_area,
            app.text_input,
            app.number_input,
            app.selectbox,
            app.segmented_control,
        )
        for widget in collection
        if str(widget.key or "").startswith("ir_fact_")
    ]
    assert attribute_widgets == []
    assert t("key_information", "en") not in _all_text(app)


def test_fact_candidate_must_map_to_a_known_canonical_attribute_id(
    repository: JsonComplianceRepository,
) -> None:
    accepted, dropped = _accept(
        repository,
        INTAKE_DESCRIPTION,
        "small_consumer_electronics",
        [
            {"attribute_id": "R-ELEC-002", "value": True, "supporting_text": "Bluetooth"},
            {"attribute_id": "A-NOPE-999", "value": True, "supporting_text": "Bluetooth"},
            {"attribute_id": "A-TOY-012", "value": True, "supporting_text": "earbuds"},
            {"attribute_id": "A-ELEC-002", "value": True, "supporting_text": "Bluetooth wireless earbuds"},
        ],
    )
    assert [c.attribute_id for c in accepted] == ["A-ELEC-002"]
    assert any("unknown_attribute_id" in reason for reason in dropped)
    assert any("attribute_outside_category_vocabulary" in reason for reason in dropped)


def test_extracted_fact_requires_supporting_user_text(
    repository: JsonComplianceRepository,
) -> None:
    accepted, dropped = _accept(
        repository,
        INTAKE_DESCRIPTION,
        "small_consumer_electronics",
        [
            {"attribute_id": "A-ELEC-002", "value": True, "supporting_text": ""},
            {"attribute_id": "A-ELEC-011", "value": "lithium ion", "supporting_text": "graphene battery"},
            {"attribute_id": "A-ELEC-011", "value": "lithium ion", "supporting_text": "rechargeable lithium-ion battery"},
        ],
    )
    assert [c.attribute_id for c in accepted] == ["A-ELEC-011"]
    assert dropped.count("supporting_text_not_traceable:A-ELEC-002") == 1
    assert "supporting_text_not_traceable:A-ELEC-011" in dropped
    assert accepted[0].supporting_text in INTAKE_DESCRIPTION


def test_unstated_fact_is_never_converted_to_false(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    """Silence about magnets/batteries never becomes a negative fact."""
    accepted, _ = _accept(
        repository,
        "Rechargeable lithium-ion battery pack with a USB-C charging case.",
        "small_consumer_electronics",
        [
            {"attribute_id": "A-TOY-012", "value": False, "supporting_text": "not mentioned"},
        ],
    )
    assert accepted == []

    outcome = pipeline.run_analysis(repository, analysis_service, electronics_category, facts=[])
    missing = {item.attribute_id for item in outcome.analysis.unknown.missing_information}
    assert "A-ELEC-015" in missing  # never stated, still unknown, never False
    assert "A-ELEC-011" in missing


def test_candidate_stays_non_canonical_before_human_confirmation(
    repository: JsonComplianceRepository,
) -> None:
    session = _intake_session()
    accepted, _ = _accept(
        repository,
        INTAKE_DESCRIPTION,
        "small_consumer_electronics",
        [
            {"attribute_id": "A-ELEC-002", "value": True, "supporting_text": "Bluetooth wireless earbuds"},
            {"attribute_id": "A-ELEC-011", "value": "lithium ion", "supporting_text": "rechargeable lithium-ion battery"},
        ],
    )
    intake_flow.run_extraction(
        session, repository, session[ui_state.KEY_DESCRIPTION], "small_consumer_electronics", None
    )
    # The stub above produced nothing (no model); store the validated bundle directly.
    session[intake_flow.KEY_INTAKE_CANDIDATES] = [c.model_dump(mode="json") for c in accepted]
    session[intake_flow.KEY_INTAKE_TEXT_USED] = session[ui_state.KEY_DESCRIPTION]

    assert intake_flow.is_confirmed(session) is False
    assert session[ui_state.KEY_FACT_ANSWERS] == {}
    assert session[ui_state.KEY_ANALYSIS] is None
    assert pipeline.build_facts(session[ui_state.KEY_FACT_ANSWERS]) == []
    # The candidates are visible to the customer, but no fact exists yet.
    views = intake_flow.candidate_views(session, repository, "en")
    assert {view.attribute_id for view in views} == {"A-ELEC-002", "A-ELEC-011"}
    assert all(view.supporting_text for view in views)


def test_one_confirmation_converts_the_bundle_through_the_user_fact_path(
    repository: JsonComplianceRepository,
) -> None:
    session = _intake_session()
    accepted, _ = _accept(
        repository,
        INTAKE_DESCRIPTION,
        "small_consumer_electronics",
        [
            {"attribute_id": "A-ELEC-002", "value": True, "supporting_text": "Bluetooth wireless earbuds"},
            {"attribute_id": "A-ELEC-011", "value": "lithium ion", "supporting_text": "rechargeable lithium-ion battery"},
        ],
    )
    session[intake_flow.KEY_INTAKE_CANDIDATES] = [c.model_dump(mode="json") for c in accepted]
    session[intake_flow.KEY_INTAKE_TEXT_USED] = session[ui_state.KEY_DESCRIPTION]
    session[intake_flow.KEY_INTAKE_CATEGORY_USED] = "small_consumer_electronics"

    answers = intake_flow.confirm_bundle(session, repository, "small_consumer_electronics")
    assert answers == {"A-ELEC-002": True, "A-ELEC-011": "lithium ion"}
    assert intake_flow.is_confirmed(session) is True
    facts = pipeline.build_facts(session[ui_state.KEY_FACT_ANSWERS])
    assert [(fact.attribute_id, fact.value, fact.origin) for fact in facts] == [
        ("A-ELEC-002", True, FactOrigin.USER),
        ("A-ELEC-011", "lithium ion", FactOrigin.USER),
    ]
    # No privileged origin exists anywhere on this path.
    assert "FactOrigin.AGENT" not in (UI_DIR / "intake.py").read_text(encoding="utf-8")


def test_editing_intake_invalidates_stale_unconfirmed_extraction(
    repository: JsonComplianceRepository,
) -> None:
    session = _intake_session()
    session[intake_flow.KEY_INTAKE_CANDIDATES] = [
        {"attribute_id": "A-ELEC-002", "value": True, "supporting_text": "Bluetooth wireless earbuds"}
    ]
    session[intake_flow.KEY_INTAKE_TEXT_USED] = session[ui_state.KEY_DESCRIPTION]
    session[ui_state.KEY_ANALYSIS] = {"anything": True}
    session[ui_state.KEY_CONFIRMED_CATEGORY] = "small_consumer_electronics"

    # Unchanged text: nothing is invalidated.
    assert intake_flow.invalidate_if_stale(session) is False
    assert intake_flow.load_candidates(session)

    # Edited text: the unconfirmed extraction and its analysis are dropped.
    session[ui_state.KEY_DESCRIPTION] = INTAKE_DESCRIPTION + " Now with a charging case."
    assert intake_flow.invalidate_if_stale(session) is True
    assert intake_flow.load_candidates(session) == ()
    assert session[ui_state.KEY_ANALYSIS] is None
    assert session[ui_state.KEY_CONFIRMED_CATEGORY] is None
    assert session[ui_state.KEY_FACT_ANSWERS] == {}
    assert session[intake_flow.KEY_INTAKE_TEXT_USED] == ""


def test_unknown_or_omitted_details_remain_missing_after_confirmation(
    repository: JsonComplianceRepository, analysis_service: AnalysisService
) -> None:
    session = _intake_session()
    accepted, _ = _accept(
        repository,
        INTAKE_DESCRIPTION,
        "small_consumer_electronics",
        [{"attribute_id": "A-ELEC-002", "value": True, "supporting_text": "Bluetooth wireless earbuds"}],
    )
    session[intake_flow.KEY_INTAKE_CANDIDATES] = [c.model_dump(mode="json") for c in accepted]
    session[intake_flow.KEY_INTAKE_TEXT_USED] = session[ui_state.KEY_DESCRIPTION]
    session[intake_flow.KEY_INTAKE_CATEGORY_USED] = "small_consumer_electronics"
    intake_flow.confirm_bundle(session, repository, "small_consumer_electronics")

    outcome = pipeline.run_analysis(
        repository,
        analysis_service,
        pipeline.category_result_for(repository, "small_consumer_electronics"),
        facts=pipeline.build_facts(session[ui_state.KEY_FACT_ANSWERS]),
    )
    missing = {item.attribute_id for item in outcome.analysis.unknown.missing_information}
    assert "A-ELEC-002" not in missing          # explicitly stated and confirmed
    assert {"A-ELEC-012", "A-ELEC-015"}.issubset(missing)  # never stated -> still unknown
    assert "A-ELEC-016" in missing              # button/coin cell never mentioned


def test_sensitive_input_is_blocked_before_model_use(
    repository: JsonComplianceRepository,
) -> None:
    from src.agent import intake as agent_intake

    calls: list[str] = []

    class _SpyModel:  # never used: the guard must fire first
        def __getattr__(self, name):  # pragma: no cover - would mean the guard failed
            calls.append(name)
            raise AssertionError("the model must not be touched")

    extraction = agent_intake.extract_candidates(
        _SpyModel(),
        repository,
        "Bluetooth earbuds, my key is sk-FAKE-TEST-KEY-0001ABCDEF",
        "small_consumer_electronics",
    )
    assert extraction.error_code == agent_intake.SENSITIVE_INPUT
    assert extraction.candidates == ()
    assert calls == []

    # The app surfaces it safely and stores no text-derived bundle.
    app = _run_app()
    app.text_area[0].set_value("My API key is sk-FAKE-TEST-KEY-0001ABCDEF").run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert list(app.exception) == []
    assert t("intake_sensitive", "en") in _all_text(app)
    assert app.session_state[ui_state.KEY_INTAKE_CANDIDATES] == []
    assert app.session_state[ui_state.KEY_INTAKE_NOTES] == []
    assert app.session_state[ui_state.KEY_SUGGESTION] is None
    assert app.session_state[ui_state.KEY_ANALYSIS] is None
    assert "sk-FAKE-TEST-KEY-0001ABCDEF" not in _all_text(app)


def test_category_still_requires_human_confirmation() -> None:
    from src.agent.tools import build_tools
    from src.services.classification import AgentClassifier, CategorySource, CategoryStatus

    repository = JsonComplianceRepository()
    service = AnalysisService(repository)
    allowed = pipeline.allowed_categories(repository)
    classifier = AgentClassifier(lambda _: "small_consumer_electronics", allowed)
    suggestion = classifier.classify(INTAKE_DESCRIPTION)
    assert suggestion.category == "small_consumer_electronics"
    assert suggestion.category_source is CategorySource.AGENT_GENERATED
    assert suggestion.category_status is not CategoryStatus.RESOLVED

    # Only the human confirmation creates the RESOLVED canonical category.
    confirmed = pipeline.category_result_for(repository, "small_consumer_electronics")
    assert confirmed.category_source is CategorySource.HUMAN_CONFIRMED
    assert confirmed.category_status is CategoryStatus.RESOLVED

    # The page requires the two explicit human boundaries and never confirms alone.
    app = _run_app()
    app.text_area[0].set_value(INTAKE_DESCRIPTION).run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] is None
    _select_category(app, "small_consumer_electronics")
    assert _smoke_button(app, t("use_selected_category", "en"))  # category boundary
    assert not any(b.label == t("confirm_bundle", "en") for b in app.button)
    _smoke_button(app, t("use_selected_category", "en")).click().run()
    assert _smoke_button(app, t("confirm_bundle", "en"))  # bundle boundary
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] is None
    assert len(build_tools(service, confirmed)[0]) == 2  # exactly two Agent tools


def test_no_third_agent_tool_exists() -> None:
    """The intake extractor is a boundary, not a tool: the Agent keeps exactly two."""
    import src.agent.intake as agent_intake
    from src.agent.tools import build_tools

    repository = JsonComplianceRepository()
    service = AnalysisService(repository)
    category = pipeline.category_result_for(repository, "small_consumer_electronics")
    tools, _ = build_tools(service, category)
    names = {getattr(tool, "tool_name", None) or getattr(tool, "__name__", "") for tool in tools}
    assert len(tools) == 2
    assert names == {"analyze_product", "get_compliance_evidence"}

    for module in (agent_intake,):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "@tool" not in source
        assert "build_tools" not in source
    # The intake boundary never runs the compliance pipeline.
    assert "analysis_service" not in Path(agent_intake.__file__).read_text(encoding="utf-8")


# =========================================================================== #
# Consumer UX v2 - narrative report
# =========================================================================== #
FORBIDDEN_REPORT_CLAIMS = (
    "fully compliant",
    "approved for import",
    "safe to import",
    "guaranteed compliant",
    "legal approval",
    "90%",
    "95%",
    "compliance probability",
    "probability",
    "score",
)


def _report_for(repository, analysis, lang="en", category_status="RESOLVED", notes=()):
    from src.ui import narrative

    return narrative.build_report(
        repository, analysis, lang, category_status=category_status, commercial_notes=notes
    )


def _analysis_with(repository, analysis_service, category, facts):
    return pipeline.run_analysis(
        repository, analysis_service, category, facts=facts
    ).analysis


def test_customer_report_is_natural_language_oriented(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    from src.ui import narrative

    analysis = _analysis_with(
        repository,
        analysis_service,
        electronics_category,
        [ProductFact(attribute_id="A-ELEC-002", value=True, origin=FactOrigin.USER)],
    )
    report = _report_for(repository, analysis)
    text = narrative.bounded_report_text(report)
    assert len(report.overall_assessment.split()) >= 8
    assert report.overall_assessment.endswith(".")
    assert report.recommended_next_step.endswith(".")
    assert report.disclaimer
    # Paragraph-style prose, not a table dump: no code fences, pipes or enum noise.
    for token in ("|", "```", "APPLICABLE", "REVIEW_REQUIRED", "NEEDS_INFO", "NOT_APPLICABLE"):
        assert token not in text


def test_default_report_shows_no_rule_ids(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    from src.ui import narrative

    analysis = _analysis_with(
        repository,
        analysis_service,
        electronics_category,
        [ProductFact(attribute_id="A-ELEC-002", value=True, origin=FactOrigin.USER)],
    )
    text = narrative.bounded_report_text(_report_for(repository, analysis))
    assert not re.search(r"\bR-[A-Z]{3,4}-\d{3}\b", text)
    assert "ACT-" not in text
    assert "S00" not in text


def test_rule_ids_and_evidence_remain_available_in_technical_details(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    analysis = _analysis_with(
        repository,
        analysis_service,
        electronics_category,
        [ProductFact(attribute_id="A-ELEC-002", value=True, origin=FactOrigin.USER)],
    )
    cards = presenters.requirement_cards(analysis, "en")
    assert all(card.rule_id.startswith("R-") for card in cards)
    evidence = presenters.evidence_view(analysis, "en")
    assert evidence.items
    assert all(item.source_id for item in evidence.items)
    technical = presenters.technical_view(analysis, {"review_status": "REVIEW_REQUIRED"}, "en")
    assert technical["category"]["value"] == "small_consumer_electronics"
    assert technical["review_status"]
    assert "applicability_counts" in technical
    # Every canonical per-rule result keeps its identifier and statuses available.
    rows = presenters.rule_result_rows(analysis, "en")
    assert len(rows) == len(analysis.applicability.rules)
    assert all(row["rule_id"].startswith("R-") for row in rows)
    assert {row["rule_id"] for row in rows} >= {"R-ELEC-002", "R-ELEC-018", "R-ELEC-019"}
    for row in rows:
        assert row["applicability_status"] and row["rule_status"] and row["evidence_status"]


def test_report_never_claims_compliance_approval_or_probability(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    from src.ui import narrative

    for facts in ([], [ProductFact(attribute_id="A-ELEC-002", value=False, origin=FactOrigin.USER)]):
        analysis = _analysis_with(repository, analysis_service, electronics_category, facts)
        for lang in ("en", "zh"):
            text = narrative.bounded_report_text(_report_for(repository, analysis, lang)).lower()
            for phrase in FORBIDDEN_REPORT_CLAIMS:
                assert phrase not in text, (lang, phrase)
            assert not re.search(r"\b\d{1,3}\s?%", text)


def test_non_current_rules_are_never_described_as_current_obligations(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    """WATCHLIST / PROPOSED stay monitoring-only in the customer wording."""
    from src.ui import narrative

    analysis = _analysis_with(repository, analysis_service, electronics_category, [])
    lifecycle = {
        finding.rule_id: finding.rule_status
        for finding in analysis.verified.compliance_information
    }
    assert "R-ELEC-018" in lifecycle and lifecycle["R-ELEC-018"] == "WATCHLIST"
    assert "R-ELEC-019" in lifecycle and lifecycle["R-ELEC-019"] == "PROPOSED"

    report = _report_for(repository, analysis)
    assert narrative._current_applicable_findings(analysis) == []
    # The non-current lifecycle items are reported as monitoring, never as a
    # current confirmed obligation.
    monitoring = {finding.rule_id for finding in narrative._monitoring_findings(analysis)}
    assert {"R-ELEC-018", "R-ELEC-019"} <= monitoring
    assert report.overall_assessment == t("report_overall_verify", "en")
    assert report.recommended_next_step == t("report_next_verify", "en")
    assert t("report_overall_attention", "en") not in report.overall_assessment
    # Monitoring is a real computed area candidate for this state.
    assert "area_monitoring" in narrative._AREA_ORDER


def test_effective_applicable_may_be_described_as_a_current_requirement(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    from src.ui import narrative

    facts = [
        ProductFact(attribute_id="A-ELEC-002", value=True, origin=FactOrigin.USER),
        ProductFact(attribute_id="A-ELEC-003", value=["Bluetooth 2.4 GHz"], origin=FactOrigin.USER),
        ProductFact(attribute_id="A-ELEC-007", value="fully certified module", origin=FactOrigin.USER),
        ProductFact(attribute_id="A-ELEC-008", value="FCC ID ABC123", origin=FactOrigin.USER),
        ProductFact(attribute_id="A-ELEC-009", value=0, origin=FactOrigin.USER),
        ProductFact(attribute_id="A-ELEC-010", value=1, origin=FactOrigin.USER),
        ProductFact(attribute_id="A-ELEC-021", value="clear", origin=FactOrigin.USER),
    ]
    analysis = _analysis_with(repository, analysis_service, electronics_category, facts)
    current = narrative._current_applicable_findings(analysis)
    assert "R-ELEC-002" in {finding.rule_id for finding in current}
    report = _report_for(repository, analysis)
    assert report.overall_assessment == t("report_overall_attention", "en")
    assert report.recommended_next_step == t("report_next_attention", "en")


def test_effective_but_unconfirmed_applicability_is_not_a_confirmed_obligation(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    """EFFECTIVE + NEEDS_INFO/REVIEW_REQUIRED must never read as a current obligation."""
    from src.ui import narrative

    analysis = _analysis_with(repository, analysis_service, electronics_category, [])
    unconfirmed = {finding.rule_id for finding in narrative._unconfirmed_findings(analysis)}
    assert {"R-ELEC-002", "R-ELEC-012"} <= unconfirmed
    report = _report_for(repository, analysis)
    assert report.overall_assessment == t("report_overall_verify", "en")
    assert narrative._current_applicable_findings(analysis) == []


def test_missing_information_is_summarized_and_bounded(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    from src.ui import narrative

    analysis = _analysis_with(repository, analysis_service, electronics_category, [])
    report = _report_for(repository, analysis)
    assert 1 <= len(report.verification_items) <= narrative.MAX_VERIFICATION_ITEMS
    assert report.verification_more  # the rest is pointed at Technical details
    # No attribute id is exposed in the customer summary.
    for item in report.verification_items:
        assert item.question and item.attribute_id not in item.question
    # The bounded list is far smaller than the canonical missing set.
    assert len(analysis.unknown.missing_information) > narrative.MAX_VERIFICATION_ITEMS
    assert report.attention_lead and len(report.attention_areas) <= narrative.MAX_ATTENTION_AREAS


def test_unsupported_product_gets_an_unsupported_scope_report(
    repository: JsonComplianceRepository,
) -> None:
    from src.ui import narrative

    report = narrative.build_report(
        repository, None, "en", category_status="UNSUPPORTED"
    )
    assert report.overall_assessment == t("report_overall_unsupported", "en")
    assert report.recommended_next_step == t("report_next_unsupported", "en")
    assert report.attention_areas == ()
    assert report.verification_items == ()
    assert report.disclaimer
    assert "compliant" not in report.overall_assessment.lower()


def test_english_and_chinese_reports_both_render(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    from src.ui import narrative

    analysis = _analysis_with(repository, analysis_service, electronics_category, [])
    en = narrative.bounded_report_text(_report_for(repository, analysis, "en"))
    zh = narrative.bounded_report_text(_report_for(repository, analysis, "zh"))
    assert en and zh and en != zh
    assert not CJK.search(en)
    assert CJK.search(zh)
    for key in (
        "report_overall",
        "report_attention",
        "report_verify",
        "report_cost",
        "report_next_step",
        "report_limitations",
    ):
        assert t(key, "en") and t(key, "zh") and t(key, "en") != t(key, "zh")


# =========================================================================== #
# Consumer UX v2 - cost outlook safety
# =========================================================================== #
def test_cost_assessment_remains_the_canonical_authority(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    from src.services.cost import CostAssessment

    analysis = _analysis_with(repository, analysis_service, electronics_category, [])
    assert isinstance(analysis.cost, CostAssessment)
    assert analysis.cost.assessed is True
    assert analysis.cost.total_available is False
    view = presenters.cost_view(analysis.cost, "en")
    assert view.items  # unchanged structured view, still available


def test_default_report_shows_no_cost_table_only_a_narrative(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    from src.ui import narrative

    analysis = _analysis_with(repository, analysis_service, electronics_category, [])
    report = _report_for(repository, analysis)
    text = "\n".join(report.cost_outlook)
    item_ids = {item.cost_id for item in analysis.cost.items}
    assert item_ids
    for cost_id in item_ids:
        assert cost_id not in text
    assert "|" not in text


def test_cost_outlook_uses_approved_reference_amounts_verbatim(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    from src.ui import narrative

    analysis = _analysis_with(repository, analysis_service, electronics_category, [])
    report = _report_for(repository, analysis)
    direct = [
        item
        for item in analysis.cost.items
        if item.exact_amount is not None or (item.low_amount is not None and item.high_amount is not None)
    ]
    amount_text = narrative._amount_text(direct[0], "en")
    assert amount_text
    assert any(amount_text in line for line in report.cost_outlook)


def test_no_cost_total_is_ever_invented(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    from src.ui import narrative

    analysis = _analysis_with(repository, analysis_service, electronics_category, [])
    assert analysis.cost.total_available is False
    report = _report_for(repository, analysis)
    text = " ".join(report.cost_outlook)
    assert t("report_cost_no_total", "en") in text
    # No summed/invented total and no arithmetic: the only transaction-cost mention
    # is the canonical "not modelled" limitation (covered by its own test).
    for token in ("total of", "total:", "sum of", "subtotal", "grand total"):
        assert token not in text.lower()
    assert not re.search(r"\bUSD\s?[\d,]+\.\d{2}\s*[-–]\s*USD", text)


def test_user_commercial_text_is_acknowledged_only_when_explicitly_present(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    from src.ui import narrative

    analysis = _analysis_with(repository, analysis_service, electronics_category, [])
    without = _report_for(repository, analysis)
    assert without.commercial_notes == ()
    assert without.commercial_notice == ""

    with_quote = _report_for(
        repository, analysis, notes=["The supplier quoted USD 4,200 for tooling."]
    )
    assert [note.text for note in with_quote.commercial_notes] == [
        "The supplier quoted USD 4,200 for tooling."
    ]
    assert with_quote.commercial_notice == t("report_commercial_not_additive", "en")
    # The user's own amount never enters the canonical cost outlook.
    assert "4,200" not in " ".join(with_quote.cost_outlook)
    assert narrative._amount_text(analysis.cost.items[0], "en") not in with_quote.commercial_notice


def test_quote_required_and_planning_only_keep_their_canonical_meaning(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    from src.ui import narrative

    analysis = _analysis_with(repository, analysis_service, electronics_category, [])
    report = _report_for(repository, analysis)
    text = " ".join(report.cost_outlook)
    quote_required = analysis.cost.counts.get("QUOTE_REQUIRED", 0)
    assert quote_required > 0
    assert t("report_cost_quote_required", "en", count=quote_required) in text
    assert "planning reference" in text
    # A quote-required record never contributes an amount: the only monetary text
    # comes from the canonical amount of a DIRECT / PLANNING_ONLY record.
    amounts = [
        narrative._amount_text(item, "en")
        for item in analysis.cost.items
        if str(getattr(item.calculation_status, "value", "")) == "QUOTE_REQUIRED"
    ]
    assert amounts and all(amount is None for amount in amounts)


def test_no_tariff_or_duty_estimation_appears_anywhere(
    repository: JsonComplianceRepository, analysis_service: AnalysisService, electronics_category
) -> None:
    from src.ui import narrative

    analysis = _analysis_with(repository, analysis_service, electronics_category, [])
    for lang in ("en", "zh"):
        text = " ".join(_report_for(repository, analysis, lang).cost_outlook)
        for forbidden in ("tariff estimate", "duty rate", "landed cost of", "customs value"):
            assert forbidden not in text.lower()
    # The only mention of tariff/duty is the explicit "not modelled" limitation.
    en = " ".join(_report_for(repository, analysis).cost_outlook).lower()
    assert "not modelled" in en and "tariff/duty" in en


# =========================================================================== #
# Consumer UX v2 - E2E scenarios (offline, injected extraction)
# =========================================================================== #
def _install_ai_stubs(
    monkeypatch,
    *,
    suggestion: str | None,
    raw_candidates,
    notes=(),
    error=None,
    warnings_prose=(),
    candidates_by_category=None,
):
    """Inject a deterministic stand-in for the two AI boundaries.

    Category suggestion and extraction are the only model touches in the flow, so
    stubbing them keeps the E2E fully offline while every other step (validation,
    category binding, confirmation, canonical analysis, report) runs for real.

    ``candidates_by_category`` lets a test return a different raw bundle per
    category, which is how the category-binding regressions are exercised.
    """
    from src.agent.intake import validate_candidates
    from src.services.classification import (
        AgentClassifier,
        CategoryResult,
        CategorySource,
        CategoryStatus,
    )
    from src.ui.pipeline import SuggestionOutcome

    repository = JsonComplianceRepository()
    extraction_calls: list[str | None] = []

    def fake_suggest(repo, description, model=None):
        if suggestion is None:
            return SuggestionOutcome(suggestion=None, error_code=None, ai_used=False)
        return SuggestionOutcome(
            suggestion=CategoryResult(
                category=suggestion,
                category_source=CategorySource.AGENT_GENERATED,
                category_status=CategoryStatus.REVIEW_REQUIRED,
            ),
            error_code=None,
            ai_used=True,
        )

    def fake_run_extraction(session_state, repo, description, category, model):
        extraction_calls.append(category)
        raw = raw_candidates
        if candidates_by_category is not None:
            raw = candidates_by_category.get(str(category or ""), [])
        accepted, _dropped = validate_candidates(repository, description, category, raw)
        session_state[ui_state.KEY_INTAKE_CANDIDATES] = [
            candidate.model_dump(mode="json") for candidate in accepted
        ]
        session_state[ui_state.KEY_INTAKE_NOTES] = list(notes)
        session_state[ui_state.KEY_INTAKE_WARNINGS] = list(warnings_prose)
        session_state[ui_state.KEY_INTAKE_ERROR] = error
        session_state[ui_state.KEY_INTAKE_AI_USED] = True
        session_state[ui_state.KEY_INTAKE_TEXT_USED] = description
        session_state[ui_state.KEY_INTAKE_CONFIRMED] = False
        # The bundle is bound to the exact category whose vocabulary produced it.
        session_state[ui_state.KEY_INTAKE_CATEGORY_USED] = (
            str(category).strip() or None if category else None
        )
        return None

    monkeypatch.setattr("src.ui.pipeline.suggest_category", fake_suggest)
    monkeypatch.setattr("src.ui.intake.run_extraction", fake_run_extraction)
    # No provider is resolved: the deterministic analysis path runs (no network).
    monkeypatch.setattr(
        "src.ui.state.resolve_model",
        lambda session_state, mode=None: ui_state.ModelResolution(model=None, error_code=None),
    )
    assert AgentClassifier  # imported for parity with the real classifier wiring
    return extraction_calls


def _run_intake_flow(app, text: str, *, confirm: bool = True):
    app.text_area[0].set_value(text).run()
    app.run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    if confirm:
        _confirm_bundle(app, "en")
    return app


def test_e2e_scenario_a_bluetooth_earbuds(monkeypatch) -> None:
    from src.services.analysis import AnalysisResult

    _install_ai_stubs(
        monkeypatch,
        suggestion="small_consumer_electronics",
        raw_candidates=[
            {"attribute_id": "A-ELEC-002", "value": True, "supporting_text": "Bluetooth wireless earbuds"},
            {"attribute_id": "A-ELEC-011", "value": "lithium ion", "supporting_text": "rechargeable lithium-ion battery"},
            {"attribute_id": "A-ELEC-019", "value": True, "supporting_text": "USB-C charging case"},
            {"attribute_id": "A-CMN-004", "value": ["Amazon"], "supporting_text": "Sold on Amazon US"},
            # Never stated -> must be rejected by the authority rules:
            {"attribute_id": "A-ELEC-001", "value": True, "supporting_text": "earbuds are for children"},
            {"attribute_id": "A-TOY-012", "value": False, "supporting_text": ""},
            {"attribute_id": "A-ELEC-015", "value": False, "supporting_text": "no report available"},
        ],
    )
    app = _run_app()
    _run_intake_flow(
        app,
        "Bluetooth wireless earbuds with a rechargeable lithium-ion battery and a "
        "USB-C charging case. Sold on Amazon US.",
        confirm=False,
    )
    assert list(app.exception) == []
    # Category suggestion is visible, human confirmation is still required.
    assert t("ai_understood_title", "en") in _all_text(app)
    assert t("cat_small_consumer_electronics", "en") in _all_text(app)
    assert t("confirm_category_bundle_note", "en") in _all_text(app)
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] is None

    stored = {c["attribute_id"] for c in app.session_state[ui_state.KEY_INTAKE_CANDIDATES]}
    assert stored == {"A-ELEC-002", "A-ELEC-011", "A-ELEC-019", "A-CMN-004"}
    assert "A-ELEC-001" not in stored   # child-directed claim not traceable -> dropped
    assert "A-TOY-012" not in stored    # magnets never mentioned (never False)
    assert "A-ELEC-015" not in stored   # absent document never asserted

    _smoke_button(app, t("confirm_bundle", "en")).click().run()
    assert list(app.exception) == []
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] == "small_consumer_electronics"
    facts = {fact.attribute_id for fact in pipeline.build_facts(app.session_state[ui_state.KEY_FACT_ANSWERS])}
    assert facts == {"A-ELEC-002", "A-ELEC-011", "A-ELEC-019", "A-CMN-004"}

    text = _all_text(app)
    assert t("report_section", "en") in text
    assert t("technical_details_label", "en") in [e.label for e in app.expander]
    missing = {
        item.attribute_id
        for item in AnalysisResult.model_validate(
            app.session_state[ui_state.KEY_ANALYSIS]
        ).unknown.missing_information
    }
    assert "A-ELEC-015" in missing and "A-ELEC-016" in missing


def test_e2e_scenario_b_toy_only_stated_facts(monkeypatch) -> None:
    from src.services.analysis import AnalysisResult

    _install_ai_stubs(
        monkeypatch,
        suggestion="childrens_toys",
        raw_candidates=[
            {"attribute_id": "A-TOY-001", "value": 36, "supporting_text": "ages 3 and up"},
            {"attribute_id": "A-TOY-008", "value": True, "supporting_text": "painted in multiple colors"},
            # Not stated -> never emitted by a correct extractor, and the traceability
            # authority rejects it if it ever is.
            {"attribute_id": "A-TOY-012", "value": False, "supporting_text": ""},
        ],
    )
    app = _run_app()
    _run_intake_flow(
        app,
        "Wooden building blocks for children ages 3 and up, painted in multiple "
        "colors and sold as a toy set.",
    )
    assert list(app.exception) == []
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] == "childrens_toys"
    stored = {c["attribute_id"] for c in app.session_state[ui_state.KEY_INTAKE_CANDIDATES]}
    assert stored == {"A-TOY-001", "A-TOY-008"}
    analysis = AnalysisResult.model_validate(app.session_state[ui_state.KEY_ANALYSIS])
    missing = {item.attribute_id for item in analysis.unknown.missing_information}
    assert "A-TOY-015" in missing and "A-TOY-012" in missing  # battery/magnet stay unknown
    assert t("report_section", "en") in _all_text(app)


def test_e2e_scenario_c_unsupported_product(monkeypatch) -> None:
    _install_ai_stubs(monkeypatch, suggestion=None, raw_candidates=[])
    app = _run_app()
    app.text_area[0].set_value("Bulk ground black pepper spice blend, 1 kg bag.").run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert list(app.exception) == []
    # No suggestion is fabricated; the customer chooses the unsupported boundary.
    assert t("no_suggestion", "en") in _all_text(app)
    _confirm_bundle(app, "en", category="unsupported")
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] == "unsupported"
    text = _all_text(app)
    # The rendered prose is HTML-escaped, so compare an apostrophe-free fragment.
    assert "does not appear to fall within the categories currently covered" in text
    # No fake normal report sections.
    assert t("report_attention", "en") not in text
    assert t("report_verify", "en") not in text
    assert t("report_cost", "en") not in text
    # The unsupported-scope narrative still carries its honest next-step wording.
    assert (
        narrative.build_report(
            JsonComplianceRepository(), None, "en", category_status="UNSUPPORTED"
        ).recommended_next_step
        == t("report_next_unsupported", "en")
    )


def test_e2e_scenario_d_incomplete_electronics_is_bounded(monkeypatch) -> None:
    from src.services.analysis import AnalysisResult

    _install_ai_stubs(
        monkeypatch,
        suggestion="small_consumer_electronics",
        raw_candidates=[
            {"attribute_id": "A-ELEC-002", "value": True, "supporting_text": "wireless earbuds"},
        ],
    )
    app = _run_app()
    _run_intake_flow(app, "Wireless earbuds.")
    assert list(app.exception) == []

    analysis = AnalysisResult.model_validate(app.session_state[ui_state.KEY_ANALYSIS])
    assert len(analysis.unknown.missing_information) > 20
    report = narrative.build_report(
        JsonComplianceRepository(), analysis, "en", category_status="RESOLVED"
    )
    assert 1 <= len(report.verification_items) <= narrative.MAX_VERIFICATION_ITEMS
    assert report.verification_more
    # The default page asks for a small subset, not 30+ fields.
    default_widgets = [
        widget
        for collection in (app.text_input, app.number_input, app.text_area)
        for widget in collection
        if str(widget.key or "").startswith("ir_fact_")
    ]
    assert default_widgets  # rendered only inside Technical details
    assert "further item(s) are listed in Technical details" in _all_text(app)


def test_four_visual_states_render_the_full_consumer_flow(monkeypatch) -> None:
    """EN/ZH x Light/Dark render the intake, report and advanced views cleanly."""
    from src.ui import theme

    for language in ("en", "zh"):
        for appearance in ("light", "dark"):
            app = _run_app()
            _smoke_widget(app, ui_state.KEY_LANGUAGE_CONTROL).set_value(language).run()
            _smoke_widget(app, ui_state.KEY_THEME_CONTROL).set_value(appearance).run()
            app.text_area[0].set_value("Bluetooth wireless earbuds with a battery").run()
            _smoke_button(app, t("analyze_product", language)).click().run()
            _confirm_bundle(app, language, category="small_consumer_electronics")
            assert list(app.exception) == [], (language, appearance)
            text = _all_text(app)
            assert t("report_section", language) in text
            assert t("report_overall", language) in text
            assert t("scenario_section", language) in [e.label for e in app.expander]
            assert t("technical_details_label", language) in [e.label for e in app.expander]
            if language == "en":
                offenders = [
                    value
                    for value in _visible_strings(app)
                    if CJK.search(value) and value.strip() not in ALLOWED_CJK_IN_ENGLISH
                ]
                assert offenders == []
            assert theme.TOKENS[appearance]["background"] in theme.theme_css(appearance)


def test_electronics_toy_and_unsupported_flows_render_in_both_languages(monkeypatch) -> None:
    """Offline smoke matrix: three flows x two languages, no live provider."""
    scenarios = {
        "electronics": (
            "Bluetooth wireless earbuds with a rechargeable lithium-ion battery.",
            "small_consumer_electronics",
            [
                {"attribute_id": "A-ELEC-002", "value": True, "supporting_text": "Bluetooth wireless earbuds"},
                {"attribute_id": "A-ELEC-011", "value": "lithium ion", "supporting_text": "rechargeable lithium-ion battery"},
            ],
        ),
        "toy": (
            "Wooden building blocks for children ages 3 and up.",
            "childrens_toys",
            [{"attribute_id": "A-TOY-001", "value": 36, "supporting_text": "ages 3 and up"}],
        ),
        "unsupported": (
            "Bulk ground black pepper spice blend, 1 kg bag.",
            None,
            [],
        ),
    }
    for name, (text, suggestion, candidates) in scenarios.items():
        for language in ("en", "zh"):
            _install_ai_stubs(
                monkeypatch, suggestion=suggestion, raw_candidates=candidates
            )
            app = _run_app()
            _smoke_widget(app, ui_state.KEY_LANGUAGE_CONTROL).set_value(language).run()
            app.text_area[0].set_value(text).run()
            _smoke_button(app, t("analyze_product", language)).click().run()
            assert list(app.exception) == [], (name, language)
            _confirm_bundle(
                app, language, category="unsupported" if suggestion is None else None
            )
            assert list(app.exception) == [], (name, language)

            rendered_text = _all_text(app)
            assert t("report_section", language) in rendered_text, (name, language)
            assert t("report_overall", language) in rendered_text, (name, language)
            assert t("technical_details_label", language) in [
                expander.label for expander in app.expander
            ], (name, language)
            if name == "unsupported":
                assert (
                    "does not appear to fall within the categories currently covered"
                    if language == "en"
                    else t("report_overall_unsupported", "zh")
                ) in rendered_text
            else:
                assert t("report_cost", language) in rendered_text, (name, language)


# =========================================================================== #
# Consumer UX v2 final authority repair - dual vocabulary
# =========================================================================== #
DUAL_DESCRIPTION = (
    "Wooden building blocks painted in multiple colors with Bluetooth speakers, "
    "a rechargeable lithium-ion battery, for children ages 3 and up."
)


def test_dual_vocabulary_is_the_union_of_common_toy_and_electronics(
    repository: JsonComplianceRepository,
) -> None:
    """`dual` must offer the toy + electronics + common attributes, each exactly once."""
    from src.agent.intake import (
        INTAKE_EXCLUDED_ATTRIBUTE_IDS,
        allowed_attribute_ids,
        attribute_categories_for,
    )

    common = set(allowed_attribute_ids(repository, None))
    toy = set(allowed_attribute_ids(repository, "childrens_toys"))
    electronics = set(allowed_attribute_ids(repository, "small_consumer_electronics"))
    dual = allowed_attribute_ids(repository, "dual")

    assert common and toy and electronics
    assert set(dual) == common | toy | electronics
    assert len(dual) == len(set(dual))  # de-duplicated by canonical attribute_id
    assert toy <= set(dual) and electronics <= set(dual) and common <= set(dual)
    # The union is declared explicitly because no taxonomy record is categorised "dual".
    assert attribute_categories_for("dual") == (
        "childrens_toys",
        "small_consumer_electronics",
        "common",
    )
    assert all(
        repository.get_attribute(attribute_id).category
        in {"common", "childrens_toys", "small_consumer_electronics"}
        for attribute_id in dual
    )
    # The category attribute is excluded from every intake vocabulary (see its own tests).
    assert "A-CMN-001" in INTAKE_EXCLUDED_ATTRIBUTE_IDS
    assert not (INTAKE_EXCLUDED_ATTRIBUTE_IDS & set(dual))


def test_category_attribute_is_excluded_from_the_intake_vocabulary(
    repository: JsonComplianceRepository,
) -> None:
    """A-CMN-001 stays canonical but is never an extractable customer fact."""
    from src.agent import intake as agent_intake

    taxonomy_common = {
        attribute.attribute_id
        for attribute in repository.attributes
        if attribute.category == "common"
    }
    allowed_common = set(agent_intake.allowed_attribute_ids(repository, None))
    # Exactly the category attribute is removed; every other common attribute remains.
    assert taxonomy_common - allowed_common == {"A-CMN-001"}
    assert allowed_common == taxonomy_common - {"A-CMN-001"}
    assert len(allowed_common) == len(taxonomy_common) - 1
    # The vocabulary used for the prompt never exposes it either.
    for category in (None, "small_consumer_electronics", "childrens_toys", "dual"):
        vocab = {
            entry["attribute_id"]
            for entry in agent_intake.attribute_vocabulary(repository, category)
        }
        assert "A-CMN-001" not in vocab, category
        assert len(vocab) == len(agent_intake.allowed_attribute_ids(repository, category))
    # Dual still covers toys + electronics + the remaining common attributes.
    dual = set(agent_intake.allowed_attribute_ids(repository, "dual"))
    assert {"A-TOY-001", "A-ELEC-002", "A-CMN-002", "A-CMN-005"} <= dual
    assert "A-CMN-001" not in dual
    # A-CMN-001 remains the canonical category vocabulary source (unchanged).
    assert repository.get_attribute("A-CMN-001").allowed_values
    assert pipeline.allowed_categories(repository) == list(
        repository.get_attribute("A-CMN-001").allowed_values
    )


def test_extracted_category_fact_is_rejected_and_cannot_contradict_the_human_choice(
    repository: JsonComplianceRepository, analysis_service: AnalysisService
) -> None:
    """The human-confirmed category remains the only category authority."""
    description = (
        "Wooden building blocks painted in multiple colors for children ages 3 and up."
    )
    accepted, dropped = _accept(
        repository,
        description,
        "childrens_toys",
        [
            {"attribute_id": "A-CMN-001", "value": "small_consumer_electronics", "supporting_text": "building blocks"},
            {"attribute_id": "A-TOY-001", "value": 36, "supporting_text": "ages 3 and up"},
        ],
    )
    assert "category_attribute_not_extractable:A-CMN-001" in dropped
    assert {candidate.attribute_id for candidate in accepted} == {"A-TOY-001"}

    # Even if a bundle somehow carried a category value, the analysis category comes
    # only from the human confirmation - never from the bundle.
    session = _intake_session()
    session[ui_state.KEY_DESCRIPTION] = description
    session[intake_flow.KEY_INTAKE_TEXT_USED] = description
    session[intake_flow.KEY_INTAKE_CATEGORY_USED] = "childrens_toys"
    session[intake_flow.KEY_INTAKE_CANDIDATES] = [
        {"attribute_id": "A-TOY-001", "value": 36, "supporting_text": "ages 3 and up"},
    ]
    answers = intake_flow.confirm_bundle(session, repository, "childrens_toys")
    assert answers == {"A-TOY-001": 36}
    facts = pipeline.build_facts(session[ui_state.KEY_FACT_ANSWERS])
    assert {fact.attribute_id for fact in facts} == {"A-TOY-001"}
    assert all(fact.attribute_id != "A-CMN-001" for fact in facts)
    outcome = pipeline.run_analysis(
        repository,
        analysis_service,
        pipeline.category_result_for(repository, "childrens_toys"),
        facts=facts,
    )
    assert outcome.analysis.classification.get("category") == "childrens_toys"


def test_dual_extraction_accepts_stated_toy_and_electronics_attributes(
    repository: JsonComplianceRepository,
) -> None:
    accepted, dropped = _accept(
        repository,
        DUAL_DESCRIPTION,
        "dual",
        [
            {"attribute_id": "A-TOY-001", "value": 36, "supporting_text": "ages 3 and up"},
            {"attribute_id": "A-ELEC-002", "value": True, "supporting_text": "with Bluetooth speakers"},
            # The category attribute is owned by the human category confirmation.
            {"attribute_id": "A-CMN-001", "value": "dual", "supporting_text": "building blocks"},
            # Another common attribute is still extractable when traceable.
            {"attribute_id": "A-CMN-005", "value": False, "supporting_text": "painted in multiple colors"},
            {"attribute_id": "A-NOPE-001", "value": True, "supporting_text": "building blocks"},
        ],
    )
    assert {candidate.attribute_id for candidate in accepted} == {
        "A-TOY-001",
        "A-ELEC-002",
        "A-CMN-005",
    }
    assert "category_attribute_not_extractable:A-CMN-001" in dropped
    assert any("unknown_attribute_id:A-NOPE-001" == reason for reason in dropped)
    # A toy-only vocabulary would have rejected the electronics attribute and vice versa.
    toy_only, toy_dropped = _accept(
        repository, DUAL_DESCRIPTION, "childrens_toys",
        [{"attribute_id": "A-ELEC-002", "value": True, "supporting_text": "with Bluetooth speakers"}],
    )
    assert toy_only == []
    assert "attribute_outside_category_vocabulary:A-ELEC-002" in toy_dropped


# =========================================================================== #
# Consumer UX v2 final authority repair - category/extraction binding
# =========================================================================== #
ELECTRONICS_RAW = [
    {"attribute_id": "A-ELEC-002", "value": True, "supporting_text": "Bluetooth wireless earbuds"},
    {"attribute_id": "A-ELEC-011", "value": "lithium ion", "supporting_text": "rechargeable lithium-ion battery"},
]
TOY_RAW = [
    {"attribute_id": "A-TOY-001", "value": 36, "supporting_text": "ages 3 and up"},
    {"attribute_id": "A-TOY-008", "value": True, "supporting_text": "painted in multiple colors"},
]
BOTH_RAW = [
    {"attribute_id": "A-ELEC-002", "value": True, "supporting_text": "with Bluetooth speakers"},
    {"attribute_id": "A-ELEC-011", "value": "lithium ion", "supporting_text": "rechargeable lithium-ion battery"},
    {"attribute_id": "A-TOY-001", "value": 36, "supporting_text": "ages 3 and up"},
    {"attribute_id": "A-TOY-008", "value": True, "supporting_text": "painted in multiple colors"},
]


def test_stored_extraction_bundle_is_bound_to_its_category_explicitly(
    repository: JsonComplianceRepository,
) -> None:
    """The binding is a stored value, not something inferred from candidate ids."""
    session = _intake_session()
    intake_flow.run_extraction(
        session, repository, INTAKE_DESCRIPTION, "small_consumer_electronics", None
    )
    assert intake_flow.extraction_category(session) == "small_consumer_electronics"
    assert session[intake_flow.KEY_INTAKE_CATEGORY_USED] == "small_consumer_electronics"

    intake_flow.run_extraction(session, repository, INTAKE_DESCRIPTION, "childrens_toys", None)
    assert intake_flow.extraction_category(session) == "childrens_toys"

    intake_flow.clear_intake_extraction(session)
    assert intake_flow.extraction_category(session) is None


def test_bundle_extracted_for_another_category_cannot_become_user_facts(
    repository: JsonComplianceRepository,
) -> None:
    """BLOCKER #1 regression: electronics candidates must not survive a switch to toys."""
    session = _intake_session()
    accepted, _ = _accept(
        repository, INTAKE_DESCRIPTION, "small_consumer_electronics", ELECTRONICS_RAW
    )
    session[intake_flow.KEY_INTAKE_CANDIDATES] = [c.model_dump(mode="json") for c in accepted]
    session[intake_flow.KEY_INTAKE_TEXT_USED] = session[ui_state.KEY_DESCRIPTION]
    session[intake_flow.KEY_INTAKE_CATEGORY_USED] = "small_consumer_electronics"

    # Confirming with a DIFFERENT category is refused and the stale bundle discarded.
    assert intake_flow.confirm_bundle(session, repository, "childrens_toys") == {}
    assert intake_flow.is_confirmed(session) is False
    assert session[ui_state.KEY_FACT_ANSWERS] == {}
    assert intake_flow.load_candidates(session) == ()
    assert pipeline.build_facts(session[ui_state.KEY_FACT_ANSWERS]) == []

    # The same bundle may only be confirmed for the category that produced it.
    session[intake_flow.KEY_INTAKE_CANDIDATES] = [c.model_dump(mode="json") for c in accepted]
    session[intake_flow.KEY_INTAKE_TEXT_USED] = session[ui_state.KEY_DESCRIPTION]
    session[intake_flow.KEY_INTAKE_CATEGORY_USED] = "small_consumer_electronics"
    answers = intake_flow.confirm_bundle(session, repository, "small_consumer_electronics")
    assert answers == {"A-ELEC-002": True, "A-ELEC-011": "lithium ion"}


def test_unbound_or_unrun_extraction_is_never_confirmable(
    repository: JsonComplianceRepository,
) -> None:
    """A category that was never read (or no extraction at all) has no confirmable bundle."""
    session = _intake_session()
    assert intake_flow.bundle_matches_category(session, "small_consumer_electronics") is False
    assert intake_flow.confirm_bundle(session, repository, "small_consumer_electronics") == {}

    # Bound but to a different category.
    session[intake_flow.KEY_INTAKE_TEXT_USED] = session[ui_state.KEY_DESCRIPTION]
    session[intake_flow.KEY_INTAKE_CATEGORY_USED] = "childrens_toys"
    assert intake_flow.bundle_matches_category(session, "childrens_toys") is True
    assert intake_flow.bundle_matches_category(session, "small_consumer_electronics") is False
    # An edited description makes even a matching binding stale.
    session[ui_state.KEY_DESCRIPTION] = INTAKE_DESCRIPTION + " Extra detail."
    assert intake_flow.bundle_matches_category(session, "childrens_toys") is False


def test_e2e_electronics_suggestion_accepted(monkeypatch) -> None:
    """Smoke A: suggestion -> accepted category -> extraction -> confirmation -> report."""
    calls = _install_ai_stubs(
        monkeypatch, suggestion="small_consumer_electronics", raw_candidates=ELECTRONICS_RAW
    )
    app = _run_app()
    app.text_area[0].set_value(INTAKE_DESCRIPTION).run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert list(app.exception) == []
    assert calls == ["small_consumer_electronics"]  # extraction used the suggested category
    assert app.session_state[ui_state.KEY_INTAKE_CATEGORY_USED] == "small_consumer_electronics"

    # The suggested category is the default selection, so the bundle is already bound.
    _confirm_bundle(app, "en")
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] == "small_consumer_electronics"
    facts = {f.attribute_id for f in pipeline.build_facts(app.session_state[ui_state.KEY_FACT_ANSWERS])}
    assert facts == {"A-ELEC-002", "A-ELEC-011"}
    assert t("report_section", "en") in _all_text(app)


MIXED_DESCRIPTION = (
    "Bluetooth wireless earbuds with a rechargeable lithium-ion battery, and a "
    "wooden toy set for children ages 3 and up."
)
#: Each run proposes both categories' attributes; only the in-vocabulary ones survive.
MIXED_RAW = ELECTRONICS_RAW + [
    {"attribute_id": "A-TOY-001", "value": 36, "supporting_text": "ages 3 and up"},
]


def test_e2e_electronics_suggestion_corrected_to_toys(monkeypatch) -> None:
    """Smoke B: the electronics bundle is discarded and a toy extraction is required."""
    calls = _install_ai_stubs(
        monkeypatch,
        suggestion="small_consumer_electronics",
        raw_candidates=[],
        candidates_by_category={
            "small_consumer_electronics": MIXED_RAW,
            "childrens_toys": [
                {"attribute_id": "A-ELEC-002", "value": True, "supporting_text": "Bluetooth wireless earbuds"},
                {"attribute_id": "A-TOY-001", "value": 36, "supporting_text": "ages 3 and up"},
            ],
        },
    )
    app = _run_app()
    app.text_area[0].set_value(MIXED_DESCRIPTION).run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert app.session_state[ui_state.KEY_INTAKE_CATEGORY_USED] == "small_consumer_electronics"
    # The toy attribute was proposed but is outside the electronics vocabulary.
    assert {c["attribute_id"] for c in app.session_state[ui_state.KEY_INTAKE_CANDIDATES]} == {
        "A-ELEC-002",
        "A-ELEC-011",
    }

    # Human correction to toys: the card offers the category boundary, not confirmation.
    _select_category(app, "childrens_toys")
    assert not any(b.label == t("confirm_bundle", "en") for b in app.button)
    assert _smoke_button(app, t("use_selected_category", "en"))
    assert t("extraction_category_changed", "en") in _all_text(app)
    # The stale electronics bundle is not shown for the toy category.
    assert "Bluetooth wireless earbuds" not in _all_text(app)

    _smoke_button(app, t("use_selected_category", "en")).click().run()
    assert list(app.exception) == []
    assert calls[-1] == "childrens_toys"  # re-extraction used the HUMAN category
    assert app.session_state[ui_state.KEY_INTAKE_CATEGORY_USED] == "childrens_toys"
    toy_candidates = {c["attribute_id"] for c in app.session_state[ui_state.KEY_INTAKE_CANDIDATES]}
    # Only toy vocabulary survives: the electronics candidate was rejected.
    assert toy_candidates == {"A-TOY-001"}
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] is None
    assert app.session_state[ui_state.KEY_FACT_ANSWERS] == {}

    # Second explicit confirmation, then only toy facts exist.
    _confirm_bundle(app, "en")
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] == "childrens_toys"
    facts = {f.attribute_id for f in pipeline.build_facts(app.session_state[ui_state.KEY_FACT_ANSWERS])}
    assert facts == {"A-TOY-001"}
    assert "A-ELEC-002" not in facts
    assert t("report_section", "en") in _all_text(app)


def test_e2e_no_suggestion_manual_toy_selection(monkeypatch) -> None:
    """Smoke C: no AI suggestion, the human picks toys, extraction uses the toy vocabulary."""
    calls = _install_ai_stubs(monkeypatch, suggestion=None, raw_candidates=TOY_RAW)
    app = _run_app()
    app.text_area[0].set_value("Wooden building blocks for children ages 3 and up.").run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert list(app.exception) == []
    assert t("no_suggestion", "en") in _all_text(app)
    assert app.session_state[ui_state.KEY_INTAKE_CATEGORY_USED] is None
    assert not any(b.label == t("confirm_bundle", "en") for b in app.button)

    _select_category(app, "childrens_toys")
    _smoke_button(app, t("use_selected_category", "en")).click().run()
    assert calls[-1] == "childrens_toys"
    assert app.session_state[ui_state.KEY_INTAKE_CATEGORY_USED] == "childrens_toys"
    assert {c["attribute_id"] for c in app.session_state[ui_state.KEY_INTAKE_CANDIDATES]} == {
        "A-TOY-001",
    }
    _confirm_bundle(app, "en")
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] == "childrens_toys"


def test_e2e_dual_accepts_both_vocabularies(monkeypatch) -> None:
    """Smoke D: dual extraction may accept a stated toy AND a stated electronics attribute."""
    calls = _install_ai_stubs(
        monkeypatch, suggestion="dual", raw_candidates=BOTH_RAW
    )
    app = _run_app()
    app.text_area[0].set_value(DUAL_DESCRIPTION).run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert calls == ["dual"]
    stored = {c["attribute_id"] for c in app.session_state[ui_state.KEY_INTAKE_CANDIDATES]}
    assert stored == {"A-ELEC-002", "A-ELEC-011", "A-TOY-001", "A-TOY-008"}
    _confirm_bundle(app, "en")
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] == "dual"
    facts = {f.attribute_id for f in pipeline.build_facts(app.session_state[ui_state.KEY_FACT_ANSWERS])}
    assert facts == stored


def test_e2e_unsupported_keeps_no_supported_category_facts(monkeypatch) -> None:
    """Smoke E: switching to the unsupported boundary must not preserve supported facts."""
    calls = _install_ai_stubs(
        monkeypatch,
        suggestion="small_consumer_electronics",
        raw_candidates=[],
        candidates_by_category={
            "small_consumer_electronics": ELECTRONICS_RAW,
            # The unsupported vocabulary is common-only, so this candidate is rejected
            # even though its quote is traceable.
            "unsupported": [
                {"attribute_id": "A-ELEC-002", "value": True, "supporting_text": "Bluetooth wireless earbuds"}
            ],
        },
    )
    app = _run_app()
    app.text_area[0].set_value(INTAKE_DESCRIPTION).run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert {c["attribute_id"] for c in app.session_state[ui_state.KEY_INTAKE_CANDIDATES]} == {
        "A-ELEC-002",
        "A-ELEC-011",
    }

    # The customer changes to the unsupported boundary: the supported-category bundle
    # is dropped and the extraction re-runs with the unsupported vocabulary.
    _select_category(app, "unsupported")
    _smoke_button(app, t("use_selected_category", "en")).click().run()
    assert calls[-1] == "unsupported"
    assert app.session_state[ui_state.KEY_INTAKE_CANDIDATES] == []
    assert app.session_state[ui_state.KEY_INTAKE_CATEGORY_USED] == "unsupported"
    assert app.session_state[ui_state.KEY_FACT_ANSWERS] == {}

    _confirm_bundle(app, "en")
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] == "unsupported"
    assert pipeline.build_facts(app.session_state[ui_state.KEY_FACT_ANSWERS]) == []
    assert "does not appear to fall within the categories currently covered" in _all_text(app)


def test_editing_description_still_invalidates_a_bound_extraction(monkeypatch) -> None:
    """Regression 9: the text boundary keeps working after the category-binding repair."""
    _install_ai_stubs(
        monkeypatch, suggestion="small_consumer_electronics", raw_candidates=ELECTRONICS_RAW
    )
    app = _run_app()
    app.text_area[0].set_value(INTAKE_DESCRIPTION).run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert app.session_state[ui_state.KEY_INTAKE_CATEGORY_USED] == "small_consumer_electronics"

    app.text_area[0].set_value(INTAKE_DESCRIPTION + " They come with a charging case.").run()
    assert list(app.exception) == []
    assert app.session_state[ui_state.KEY_INTAKE_CANDIDATES] == []
    assert app.session_state[ui_state.KEY_INTAKE_CATEGORY_USED] is None
    assert app.session_state[ui_state.KEY_INTAKE_CONFIRMED] is False
    assert app.session_state[ui_state.KEY_ANALYSIS] is None


def test_start_over_clears_category_binding_and_bundle(monkeypatch) -> None:
    """Regression 10: Start Over clears the whole product state including the binding."""
    _install_ai_stubs(
        monkeypatch, suggestion="small_consumer_electronics", raw_candidates=ELECTRONICS_RAW
    )
    app = _run_app()
    app.text_area[0].set_value(INTAKE_DESCRIPTION).run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    _confirm_bundle(app, "en")
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] == "small_consumer_electronics"

    _smoke_button(app, t("start_over", "en")).click().run()
    assert list(app.exception) == []
    assert app.session_state[ui_state.KEY_INTAKE_CANDIDATES] == []
    assert app.session_state[ui_state.KEY_INTAKE_CATEGORY_USED] is None
    assert app.session_state[ui_state.KEY_INTAKE_CONFIRMED] is False
    assert app.session_state[ui_state.KEY_FACT_ANSWERS] == {}
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] is None
    assert app.session_state[ui_state.KEY_ANALYSIS] is None


# =========================================================================== #
# Consumer UX v2 final authority repair - raw model warnings never rendered
# =========================================================================== #
UNSAFE_WARNING = "FCC approval is definitely required and this product is non-compliant."


def test_raw_model_warning_prose_is_never_rendered(monkeypatch) -> None:
    """BLOCKER #3 regression: only deterministic localized ambiguity copy is shown."""
    for language in ("en", "zh"):
        _install_ai_stubs(
            monkeypatch,
            suggestion="small_consumer_electronics",
            raw_candidates=ELECTRONICS_RAW,
            warnings_prose=[UNSAFE_WARNING, UNSAFE_WARNING + " Second one."],
        )
        app = _run_app()
        _smoke_widget(app, ui_state.KEY_LANGUAGE_CONTROL).set_value(language).run()
        app.text_area[0].set_value(INTAKE_DESCRIPTION).run()
        _smoke_button(app, t("analyze_product", language)).click().run()
        assert list(app.exception) == []

        # The extractor may retain them for diagnostics ...
        assert len(app.session_state[ui_state.KEY_INTAKE_WARNINGS]) == 2
        # ... but the customer UI only shows deterministic localized copy.
        visible = _visible_strings(app)
        for value in visible:
            assert UNSAFE_WARNING not in value, (language, value)
            assert "non-compliant" not in value.lower()
            assert "approval is definitely" not in value.lower()
        text = _all_text(app)
        assert t("intake_warnings_skipped", language) in text
        assert t("intake_warnings_count", language, count=2) in text
        assert t("ai_understood_warnings", language) not in text  # raw-label removed


def test_raw_warning_prose_is_absent_after_confirmation_and_in_technical_details(
    monkeypatch,
) -> None:
    from src.services.analysis import AnalysisResult

    _install_ai_stubs(
        monkeypatch,
        suggestion="small_consumer_electronics",
        raw_candidates=ELECTRONICS_RAW,
        warnings_prose=[UNSAFE_WARNING],
    )
    app = _run_app()
    app.text_area[0].set_value(INTAKE_DESCRIPTION).run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    _confirm_bundle(app, "en")
    assert list(app.exception) == []
    rendered = "\n".join(_visible_strings(app))
    assert UNSAFE_WARNING not in rendered
    assert "non-compliant" not in rendered.lower()
    # The technical view is an allowlist of canonical fields: no intake warnings at all.
    analysis = AnalysisResult.model_validate(app.session_state[ui_state.KEY_ANALYSIS])
    technical = json.dumps(presenters.technical_view(analysis, {}, "en"), default=str)
    assert UNSAFE_WARNING not in technical
    assert "warnings" not in technical.lower()


def test_no_ui_module_renders_the_raw_warning_accessor() -> None:
    """Only the diagnostics accessor may read raw warnings - never a renderer."""
    components = (UI_DIR / "components.py").read_text(encoding="utf-8")
    assert "warnings(session_state)" not in components
    assert "intake_flow.warning_count(" in components
    assert "intake_warnings_skipped" in components


# =========================================================================== #
# Consumer UX v2 final cleanup - pre-Analyze gating + category widget state
# =========================================================================== #
def test_ai_understood_is_absent_until_analyze_is_clicked() -> None:
    """Typing alone must not reveal the review, the selector or the confirm control."""
    app = _run_app()
    app.text_area[0].set_value("Bluetooth wireless earbuds with a battery").run()
    assert list(app.exception) == []
    text = _all_text(app)
    assert t("intake_section", "en") in text
    assert t("ai_understood_title", "en") not in text
    assert t("no_suggestion", "en") not in text
    assert not any(s.key == ui_state.KEY_INTAKE_CATEGORY_CONTROL for s in app.selectbox)
    assert not any(b.label == t("confirm_bundle", "en") for b in app.button)
    assert not any(b.label == t("use_selected_category", "en") for b in app.button)
    assert app.session_state[ui_state.KEY_INTAKE_TEXT_USED] == ""

    # Only the explicit Analyze action reveals the review.
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert list(app.exception) == []
    assert t("ai_understood_title", "en") in _all_text(app)
    assert any(s.key == ui_state.KEY_INTAKE_CATEGORY_CONTROL for s in app.selectbox)
    assert app.session_state[ui_state.KEY_INTAKE_TEXT_USED] == (
        "Bluetooth wireless earbuds with a battery"
    )
    # The category is never auto-confirmed.
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] is None


def test_ai_understood_hides_again_when_the_description_is_edited() -> None:
    """A description edit invalidates the submission and hides the stale review."""
    app = _run_app()
    app.text_area[0].set_value("Bluetooth wireless earbuds with a battery").run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert t("ai_understood_title", "en") in _all_text(app)

    app.text_area[0].set_value("Wooden toy blocks for toddlers").run()
    assert list(app.exception) == []
    # The review is hidden until the edited text is submitted again.
    assert t("ai_understood_title", "en") not in _all_text(app)
    assert not any(s.key == ui_state.KEY_INTAKE_CATEGORY_CONTROL for s in app.selectbox)
    assert app.session_state[ui_state.KEY_INTAKE_TEXT_USED] == ""


def test_category_selector_defaults_to_the_new_suggestion_after_start_over(
    monkeypatch,
) -> None:
    """Regression A: the previous product's category must not survive Start Over."""
    _install_ai_stubs(
        monkeypatch, suggestion="small_consumer_electronics", raw_candidates=ELECTRONICS_RAW
    )
    app = _run_app()
    app.text_area[0].set_value(INTAKE_DESCRIPTION).run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    _select_category(app, "unsupported")  # the customer wanders off the suggestion
    assert _smoke_widget(app, ui_state.KEY_INTAKE_CATEGORY_CONTROL).value == "unsupported"

    app.session_state[ui_state.KEY_BYOK_CREDENTIAL] = FAKE_KEY
    _smoke_button(app, t("start_over", "en")).click().run()
    assert list(app.exception) == []
    # Presentation state and the session credential survive; widget state does not.
    assert app.session_state[ui_state.KEY_BYOK_CREDENTIAL] == FAKE_KEY
    assert app.session_state[ui_state.KEY_LANGUAGE] == "en"
    assert ui_state.KEY_INTAKE_CATEGORY_CONTROL not in app.session_state

    # Product 2: a different suggestion must be the visible default.
    _install_ai_stubs(
        monkeypatch, suggestion="childrens_toys", raw_candidates=TOY_RAW
    )
    app.text_area[0].set_value("Wooden building blocks for children ages 3 and up.").run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert list(app.exception) == []
    selector = _smoke_widget(app, ui_state.KEY_INTAKE_CATEGORY_CONTROL)
    assert selector.value == "childrens_toys"
    assert selector.value != "unsupported"
    assert selector.value != "small_consumer_electronics"


def test_category_selector_follows_a_new_suggestion_after_a_description_edit(
    monkeypatch,
) -> None:
    """Regression B: an edited description's new suggestion wins over the old selection."""
    _install_ai_stubs(
        monkeypatch, suggestion="small_consumer_electronics", raw_candidates=ELECTRONICS_RAW
    )
    app = _run_app()
    app.text_area[0].set_value(INTAKE_DESCRIPTION).run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    _select_category(app, "small_consumer_electronics")
    assert _smoke_widget(app, ui_state.KEY_INTAKE_CATEGORY_CONTROL).value == (
        "small_consumer_electronics"
    )

    # Edit the description; the stub now suggests toys.
    _install_ai_stubs(monkeypatch, suggestion="childrens_toys", raw_candidates=TOY_RAW)
    app.text_area[0].set_value("Wooden building blocks for children ages 3 and up.").run()
    assert list(app.exception) == []
    assert ui_state.KEY_INTAKE_CATEGORY_CONTROL not in app.session_state  # stale state dropped
    _smoke_button(app, t("analyze_product", "en")).click().run()
    assert list(app.exception) == []
    assert _smoke_widget(app, ui_state.KEY_INTAKE_CATEGORY_CONTROL).value == "childrens_toys"


def test_category_widget_reset_keeps_deterministic_defaults_and_never_confirms() -> None:
    """Start Over clears product/category widget state but keeps language, theme, key."""
    session: dict[str, Any] = {}
    ui_state.initialize_state(session)
    intake_flow.initialize_intake_state(session)
    session[ui_state.KEY_LANGUAGE] = "zh"
    session[ui_state.KEY_THEME] = "dark"
    session[ui_state.KEY_BYOK_CREDENTIAL] = FAKE_KEY
    session[ui_state.KEY_DESCRIPTION] = "something"
    session[ui_state.KEY_CONFIRMED_CATEGORY] = "small_consumer_electronics"
    session[ui_state.KEY_INTAKE_CATEGORY_CONTROL] = "small_consumer_electronics"
    session[ui_state.KEY_INTAKE_CATEGORY_USED] = "small_consumer_electronics"

    ui_state.start_over(session)
    assert ui_state.KEY_INTAKE_CATEGORY_CONTROL not in session
    assert session[ui_state.KEY_INTAKE_CATEGORY_USED] is None
    assert session[ui_state.KEY_CONFIRMED_CATEGORY] is None
    assert session[ui_state.KEY_DESCRIPTION] == ""
    # Preserved presentation/auth state.
    assert session[ui_state.KEY_LANGUAGE] == "zh"
    assert session[ui_state.KEY_THEME] == "dark"
    assert ui_state.get_byok_credential(session) == FAKE_KEY

    # No suggestion available: the selector falls back to the deterministic manual choice.
    app = _run_app()
    app.text_area[0].set_value(INTAKE_DESCRIPTION).run()
    _smoke_button(app, t("analyze_product", "en")).click().run()
    choices = pipeline.consumer_category_choices(JsonComplianceRepository())
    assert _smoke_widget(app, ui_state.KEY_INTAKE_CATEGORY_CONTROL).value == choices[0]
    assert app.session_state[ui_state.KEY_CONFIRMED_CATEGORY] is None


def test_ui_uses_current_streamlit_apis() -> None:
    for path in sorted(UI_DIR.glob("*.py")) + [PROJECT_ROOT / "app.py"]:
        source = path.read_text(encoding="utf-8")
        assert "use_container_width" not in source, path.name
        assert "unsafe_allow_javascript" not in source, path.name
