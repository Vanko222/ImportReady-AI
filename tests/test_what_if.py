"""Deterministic What-if engine tests (Phase 3).

Offline only: no model, no provider, no network, no environment access. All canonical inputs come
from the real approved repository through the real deterministic engines.
"""

from __future__ import annotations

import ast
import inspect
import json
from datetime import date

import pytest

from pydantic import ValidationError

from src.repositories.compliance_repository import JsonComplianceRepository
from src.services import what_if as what_if_module
from src.services.analysis import AnalysisService
from src.services.classification import CategoryResult, CategorySource, CategoryStatus
from src.services.cost import CostConfigurationError
from src.services.risk import RiskLevel, RiskReasonCode
from src.services.what_if import (
    ChangeType,
    WhatIfConfigurationError,
    WhatIfEngine,
    WhatIfInputError,
    WhatIfOverride,
    compare_analyses,
    unassessed_what_if_result,
)
from src.state import FactOrigin, ProductFact

_REPO = JsonComplianceRepository()
_SERVICE = AnalysisService(_REPO)
_ENGINE = WhatIfEngine(_SERVICE, _REPO)

_ELEC002_REQUIRED = (
    "A-ELEC-002", "A-ELEC-003", "A-ELEC-007", "A-ELEC-008", "A-ELEC-009", "A-ELEC-010", "A-ELEC-021",
)
_ELEC012_REQUIRED = (
    "A-ELEC-011", "A-ELEC-012", "A-ELEC-013", "A-ELEC-014", "A-ELEC-015", "A-ELEC-035",
)
_TOY010_REQUIRED = ("A-TOY-001", "A-TOY-005", "A-TOY-012", "A-TOY-013", "A-TOY-014")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolved(category: str = "small_consumer_electronics") -> CategoryResult:
    return CategoryResult(
        category=category,
        category_source=CategorySource.HUMAN_CONFIRMED,
        category_status=CategoryStatus.RESOLVED,
    )


def _agent_category(category: str = "small_consumer_electronics") -> CategoryResult:
    return CategoryResult(
        category=category,
        category_source=CategorySource.AGENT_GENERATED,
        category_status=CategoryStatus.REVIEW_REQUIRED,
    )


def _valid_value(attribute):
    data_type = attribute.data_type
    if data_type == "boolean":
        return True
    if data_type == "enum":
        return attribute.allowed_values[0]
    if data_type == "multi_select":
        return [attribute.allowed_values[0]] if attribute.allowed_values else []
    if data_type == "integer":
        return 1
    if data_type in ("number", "decimal"):
        return 1.0
    if data_type in ("text", "structured_text"):
        return "test"
    if data_type == "structured_list":
        return ["test"]
    if data_type == "date":
        return date(2026, 1, 1)
    raise AssertionError(f"unhandled data_type {data_type!r}")


def _facts(attribute_ids, overrides=None, origins=None) -> list[ProductFact]:
    overrides = overrides or {}
    origins = origins or {}
    facts: list[ProductFact] = []
    for attribute_id in attribute_ids:
        attribute = _REPO.get_attribute(attribute_id)
        assert attribute is not None, attribute_id
        facts.append(
            ProductFact(
                attribute_id=attribute_id,
                value=overrides.get(attribute_id, _valid_value(attribute)),
                origin=origins.get(attribute_id, FactOrigin.USER),
            )
        )
    return facts


def _elec002_facts(deciding: bool = True) -> list[ProductFact]:
    return _facts(_ELEC002_REQUIRED, overrides={"A-ELEC-002": deciding})


def _elec012_facts(chemistry: str = "lithium ion") -> list[ProductFact]:
    return _facts(_ELEC012_REQUIRED, overrides={"A-ELEC-011": chemistry})


def _status(analysis, rule_id: str) -> str:
    return next(
        r.applicability_status.value for r in analysis.applicability.rules if r.rule_id == rule_id
    )


def _reason(analysis, rule_id: str) -> list[str]:
    entry = next(r for r in analysis.applicability.rules if r.rule_id == rule_id)
    return [code.value for code in entry.reason_codes]


def _risk_levels(analysis):
    return {
        (item.rule_id or item.gap_id): item.risk_level.value
        for item in (analysis.risk.items if analysis.risk else [])
    }


def _action_ids(analysis):
    return {item.action_id for item in (analysis.actions.items if analysis.actions else [])}


def _changes(delta, attribute: str, change_type: ChangeType) -> list[str]:
    return sorted(
        item.action_id
        for item in delta.action_changes
        if item.change_type is change_type and item.attribute_id == attribute
    )


# =========================================================================== #
# 1-2. Determinism and baseline immutability
# =========================================================================== #
def test_repeated_runs_are_byte_identical() -> None:
    facts = _elec002_facts(True)
    first = _ENGINE.run(_resolved(), facts, {"A-ELEC-002": False})
    second = _ENGINE.run(_resolved(), facts, {"A-ELEC-002": False})
    assert first.to_dict() == second.to_dict()


def test_baseline_facts_and_objects_are_never_mutated() -> None:
    facts = _elec002_facts(True)
    snapshot = [fact.model_dump(mode="json") for fact in facts]
    identities = [id(fact) for fact in facts]
    result = _ENGINE.run(_resolved(), facts, {"A-ELEC-002": False})
    assert [fact.model_dump(mode="json") for fact in facts] == snapshot
    assert [id(fact) for fact in facts] == identities
    # and the caller's collection object is untouched
    assert len(facts) == len(_ELEC002_REQUIRED)
    # the scenario itself really did change (so the immutability check is meaningful)
    assert result.after.applicability is not None
    assert _status(result.after, "R-ELEC-002") == "NOT_APPLICABLE"
    assert _status(result.before, "R-ELEC-002") == "APPLICABLE"


def test_scenario_facts_are_not_persisted_on_the_input_objects() -> None:
    facts = _facts(("A-TOY-012",))
    _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-012": False})
    assert facts[0].value is True
    assert facts[0].origin is FactOrigin.USER


# =========================================================================== #
# 3-6. Override semantics
# =========================================================================== #
def test_single_override_changes_the_canonical_verdict() -> None:
    facts = _facts(_TOY010_REQUIRED, overrides={"A-TOY-012": True})
    result = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-012": False})
    assert result.assessed is True and result.hypothetical is True
    assert _status(result.before, "R-TOY-010") == "APPLICABLE"
    assert _status(result.after, "R-TOY-010") == "NOT_APPLICABLE"
    changed = {d.rule_id: d for d in result.delta.rule_changes}
    assert changed["R-TOY-010"].before_reason_codes == ["TRIGGER_SATISFIED"]
    assert changed["R-TOY-010"].after_reason_codes == ["TRIGGER_NOT_SATISFIED"]
    assert result.overrides[0].attribute_id == "A-TOY-012"
    assert result.overrides[0].before_value is True
    assert result.overrides[0].after_value is False
    assert result.overrides[0].before_present is True


def test_override_of_an_absent_attribute_reports_before_present_false() -> None:
    # every other required fact is present; only A-TOY-012 is absent
    facts = _facts(("A-TOY-001", "A-TOY-005", "A-TOY-013", "A-TOY-014"))
    result = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-012": True})
    change = result.overrides[0]
    assert change.attribute_id == "A-TOY-012"
    assert change.before_present is False
    assert change.before_value is None
    assert change.after_value is True
    assert _status(result.before, "R-TOY-010") == "NEEDS_INFO"
    assert _status(result.after, "R-TOY-010") == "APPLICABLE"


def test_multiple_overrides_are_all_applied() -> None:
    # R-TOY-010 needs A-TOY-012; R-TOY-011 needs A-TOY-015; every other required fact is present
    facts = _facts(
        ("A-TOY-001", "A-TOY-005", "A-TOY-013", "A-TOY-014", "A-TOY-016", "A-TOY-017")
    )
    result = _ENGINE.run(
        _resolved("childrens_toys"), facts, {"A-TOY-015": False, "A-TOY-012": False}
    )
    assert [c.attribute_id for c in result.overrides] == ["A-TOY-012", "A-TOY-015"]
    assert _status(result.after, "R-TOY-010") == "NOT_APPLICABLE"
    assert _status(result.after, "R-TOY-011") == "NOT_APPLICABLE"
    changed = {d.rule_id for d in result.delta.rule_changes}
    assert {"R-TOY-010", "R-TOY-011"} <= changed
    # every reported rule change is deterministic and sorted
    assert [d.rule_id for d in result.delta.rule_changes] == sorted(changed)


def test_override_order_does_not_change_the_result() -> None:
    facts = _facts(("A-TOY-001", "A-TOY-005", "A-TOY-012", "A-TOY-013", "A-TOY-014", "A-TOY-015"))
    first = _ENGINE.run(
        _resolved("childrens_toys"),
        facts,
        [WhatIfOverride(attribute_id="A-TOY-012", value=False),
         WhatIfOverride(attribute_id="A-TOY-015", value=False)],
    )
    second = _ENGINE.run(
        _resolved("childrens_toys"),
        facts,
        [WhatIfOverride(attribute_id="A-TOY-015", value=False),
         WhatIfOverride(attribute_id="A-TOY-012", value=False)],
    )
    third = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-015": False, "A-TOY-012": False})
    assert first.to_dict() == second.to_dict() == third.to_dict()


def test_no_op_override_produces_an_empty_delta() -> None:
    facts = _facts(_TOY010_REQUIRED, overrides={"A-TOY-012": True})
    result = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-012": True})
    assert result.assessed is True
    assert result.delta.rule_changes == []
    assert result.delta.risk_changes == []
    assert result.delta.cost_changes == []
    assert result.delta.action_changes == []
    assert result.delta.counts["rule_changes"] == 0
    assert result.before.model_dump(mode="json") == result.after.model_dump(mode="json")


@pytest.mark.parametrize("overrides", [None, [], {}])
def test_missing_or_empty_overrides_are_a_harmless_no_op(overrides) -> None:
    result = _ENGINE.run(_resolved(), _elec002_facts(True), overrides)
    assert result.assessed is True
    assert result.overrides == []
    assert result.delta.counts["rule_changes"] == 0
    assert any("No hypothetical fact override" in note for note in result.notes)


# =========================================================================== #
# 7-11. Invalid input and category safety
# =========================================================================== #
def test_unknown_attribute_fails_safely() -> None:
    with pytest.raises(WhatIfInputError) as excinfo:
        _ENGINE.run(_resolved(), _elec002_facts(True), {"A-FAKE-999": True})
    assert "unknown product attribute" in str(excinfo.value)
    assert "Traceback" not in str(excinfo.value)


def test_invalid_value_fails_safely() -> None:
    with pytest.raises(WhatIfInputError) as excinfo:
        _ENGINE.run(_resolved("childrens_toys"), _facts(_TOY010_REQUIRED), {"A-TOY-012": "yes"})
    assert "invalid hypothetical value" in str(excinfo.value)


def test_invalid_values_are_never_coerced_or_treated_as_false() -> None:
    for bad in ("yes", 1, None, [], "unknown"):
        with pytest.raises(WhatIfInputError):
            _ENGINE.run(_resolved("childrens_toys"), _facts(_TOY010_REQUIRED), {"A-TOY-012": bad})


def test_duplicate_override_fails_safely() -> None:
    with pytest.raises(WhatIfInputError) as excinfo:
        _ENGINE.run(
            _resolved("childrens_toys"),
            _facts(_TOY010_REQUIRED),
            [
                WhatIfOverride(attribute_id="A-TOY-012", value=False),
                WhatIfOverride(attribute_id="A-TOY-012", value=True),
            ],
        )
    assert "duplicate override" in str(excinfo.value)


def test_blank_attribute_id_fails_safely() -> None:
    with pytest.raises(WhatIfInputError):
        _ENGINE.run(_resolved("childrens_toys"), [], [WhatIfOverride(attribute_id="   ", value=True)])


def test_category_incompatible_attribute_fails_safely() -> None:
    with pytest.raises(WhatIfInputError) as excinfo:
        _ENGINE.run(_resolved("small_consumer_electronics"), [], {"A-TOY-012": True})
    assert "not compatible with the confirmed category" in str(excinfo.value)


@pytest.mark.parametrize(
    "category_result, marker",
    [
        (
            CategoryResult(
                category=None,
                category_source=CategorySource.UNRESOLVED,
                category_status=CategoryStatus.NEEDS_INFO,
            ),
            "unresolved",
        ),
        (_agent_category(), "human confirmation"),
        (
            CategoryResult(
                category="unsupported",
                category_source=CategorySource.HUMAN_CONFIRMED,
                category_status=CategoryStatus.UNSUPPORTED,
            ),
            "not available",
        ),
    ],
)
def test_non_resolved_categories_are_unassessed(category_result, marker) -> None:
    result = _ENGINE.run(category_result, _elec002_facts(True), {"A-ELEC-002": False})
    assert result.assessed is False
    assert result.hypothetical is False
    assert result.before is None and result.after is None and result.delta is None
    assert result.overrides == []
    assert marker in result.notes[0].lower()


def test_unassessed_helper_rejects_unknown_reason() -> None:
    with pytest.raises(ValueError):
        unassessed_what_if_result("whatever")


# =========================================================================== #
# 12-13. R-ELEC-002 primary demo and reverse direction
# =========================================================================== #
def test_r_elec_002_true_to_false_primary_demo() -> None:
    facts = _elec002_facts(True)
    result = _ENGINE.run(_resolved(), facts, {"A-ELEC-002": False})

    # BEFORE
    assert _status(result.before, "R-ELEC-002") == "APPLICABLE"
    assert _reason(result.before, "R-ELEC-002") == ["TRIGGER_SATISFIED"]
    assert _risk_levels(result.before)["R-ELEC-002"] == "HIGH"
    assert "ACT-RULE-R-ELEC-002-TESTS" in _action_ids(result.before)

    # AFTER
    assert _status(result.after, "R-ELEC-002") == "NOT_APPLICABLE"
    assert _reason(result.after, "R-ELEC-002") == ["TRIGGER_NOT_SATISFIED"]
    assert "R-ELEC-002" not in _risk_levels(result.after)
    assert not [a for a in _action_ids(result.after) if a.startswith("ACT-RULE-R-ELEC-002")]

    # DELTA
    rules = {d.rule_id: d for d in result.delta.rule_changes}
    assert rules["R-ELEC-002"].change_type is ChangeType.MODIFIED
    assert rules["R-ELEC-002"].before_applicability_status == "APPLICABLE"
    assert rules["R-ELEC-002"].after_applicability_status == "NOT_APPLICABLE"

    risks = {(d.rule_id, d.gap_id): d for d in result.delta.risk_changes}
    assert risks[("R-ELEC-002", None)].change_type is ChangeType.REMOVED
    assert risks[("R-ELEC-002", None)].before_level == "HIGH"
    assert risks[("R-ELEC-002", None)].after_level is None

    removed = {
        d.action_id for d in result.delta.action_changes if d.change_type is ChangeType.REMOVED
    }
    assert {
        "ACT-RULE-R-ELEC-002-TESTS",
        "ACT-RULE-R-ELEC-002-DOCUMENTS",
        "ACT-RULE-R-ELEC-002-ACTIONS",
        "ACT-RULE-R-ELEC-002-LABELING",
    } <= removed
    assert result.delta.cost_changes == []
    assert result.delta.counts["rule_modified"] == 1

    # the baseline facts are untouched by the scenario
    assert [f.value for f in facts if f.attribute_id == "A-ELEC-002"] == [True]


def test_r_elec_002_false_to_true_reverse_direction() -> None:
    facts = _elec002_facts(False)
    result = _ENGINE.run(_resolved(), facts, {"A-ELEC-002": True})

    assert _status(result.before, "R-ELEC-002") == "NOT_APPLICABLE"
    assert "R-ELEC-002" not in _risk_levels(result.before)
    assert _status(result.after, "R-ELEC-002") == "APPLICABLE"
    assert _reason(result.after, "R-ELEC-002") == ["TRIGGER_SATISFIED"]
    assert _risk_levels(result.after)["R-ELEC-002"] == "HIGH"

    added_actions = {
        d.action_id for d in result.delta.action_changes if d.change_type is ChangeType.ADDED
    }
    assert "ACT-RULE-R-ELEC-002-TESTS" in added_actions
    assert any(
        d.rule_id == "R-ELEC-002" and d.change_type is ChangeType.ADDED
        for d in result.delta.risk_changes
    )


# =========================================================================== #
# 14-17. Delta correctness per layer
# =========================================================================== #
def test_rule_delta_comparison_fields() -> None:
    # A-TOY-013 is the single missing required fact; the override completes the rule's inputs
    facts = _facts(("A-TOY-001", "A-TOY-005", "A-TOY-012", "A-TOY-014"))
    result = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-013": True})
    delta = next(d for d in result.delta.rule_changes if d.rule_id == "R-TOY-010")
    assert delta.before_applicability_status == "NEEDS_INFO"
    assert delta.after_applicability_status == "APPLICABLE"
    assert "MISSING_REQUIRED_FACTS" in delta.before_reason_codes
    assert delta.after_reason_codes == ["TRIGGER_SATISFIED"]
    assert "A-TOY-013" in delta.before_missing_attribute_ids
    assert delta.after_missing_attribute_ids == []
    assert delta.before_rule_status == delta.after_rule_status == "EFFECTIVE"
    assert delta.before_evidence_status == delta.after_evidence_status == "VERIFIED"


def test_rule_delta_is_empty_when_nothing_canonical_changed() -> None:
    facts = _facts(_TOY010_REQUIRED, overrides={"A-TOY-012": True})
    result = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-005": True})
    assert not [d for d in result.delta.rule_changes if d.rule_id == "R-TOY-010"]


def test_risk_delta_identity_and_levels() -> None:
    facts = _facts(("A-TOY-001", "A-TOY-005", "A-TOY-012", "A-TOY-014"))
    result = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-013": True})
    delta = next(d for d in result.delta.risk_changes if d.rule_id == "R-TOY-010")
    assert delta.change_type is ChangeType.MODIFIED
    assert delta.before_level == "REVIEW"
    assert delta.after_level == "HIGH"
    assert delta.gap_id is None


def test_risk_delta_reports_modified_for_a_changed_reason_code() -> None:
    before = _SERVICE.analyze(_resolved("childrens_toys"), _facts(_TOY010_REQUIRED))
    after = before.model_copy(deep=True)
    item = next(i for i in after.risk.items if i.rule_id == "R-TOY-010")
    item.risk_level = RiskLevel.REVIEW
    delta = compare_analyses(before, after)
    changed = next(d for d in delta.risk_changes if d.rule_id == "R-TOY-010")
    assert changed.change_type is ChangeType.MODIFIED
    assert changed.before_level == "HIGH"
    assert changed.after_level == "REVIEW"


def test_cost_delta_is_empty_because_cost_v1_is_category_level() -> None:
    """With the category fixed, product-fact overrides cannot change category-level cost references."""
    for category_result, facts, overrides in (
        (_resolved(), _elec002_facts(True), {"A-ELEC-002": False}),
        (_resolved("childrens_toys"), _facts(_TOY010_REQUIRED), {"A-TOY-012": False}),
    ):
        result = _ENGINE.run(category_result, facts, overrides)
        assert result.delta.cost_changes == []
        assert result.delta.counts["cost_changes"] == 0
        # the cost layer is still present and identical on both sides
        assert result.before.cost.assessed is True and result.after.cost.assessed is True
        assert result.before.cost.model_dump() == result.after.cost.model_dump()


def test_action_delta_added_removed_and_modified() -> None:
    facts = _facts(("A-TOY-001", "A-TOY-005", "A-TOY-012", "A-TOY-014"))
    result = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-013": True})
    changes = {d.action_id: d for d in result.delta.action_changes}
    assert changes["ACT-INFO-A-TOY-013"].change_type is ChangeType.REMOVED
    assert changes["ACT-RULE-R-TOY-010-TESTS"].change_type is ChangeType.ADDED
    assert changes["ACT-RULE-R-TOY-010-TESTS"].action_type == "CURRENT_OBLIGATION"
    assert changes["ACT-RULE-R-TOY-010-TESTS"].after_canonical_text == _REPO.get_rule(
        "R-TOY-010"
    ).required_tests
    # MODIFIED is exercised on a deep copy so the comparator branch is covered deterministically
    before = _SERVICE.analyze(_resolved("childrens_toys"), facts)
    after = before.model_copy(deep=True)
    after.actions.items[0].reason_code = "CHANGED_FOR_TEST"
    modified = compare_analyses(before, after)
    assert modified.action_changes
    assert modified.action_changes[0].change_type is ChangeType.MODIFIED


def test_action_delta_never_uses_list_position() -> None:
    facts = _facts(_TOY010_REQUIRED, overrides={"A-TOY-012": True})
    result = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-012": False})
    assert all(d.action_id for d in result.delta.action_changes)
    assert len({d.action_id for d in result.delta.action_changes}) == len(
        result.delta.action_changes
    )


# =========================================================================== #
# 18. Missing-information scenario
# =========================================================================== #
def test_override_completes_a_missing_fact_and_recomputes_downstream_layers() -> None:
    # every required fact of R-TOY-010 is present except A-TOY-013
    facts = _facts(("A-TOY-001", "A-TOY-005", "A-TOY-012", "A-TOY-014"))
    result = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-013": True})

    assert _status(result.before, "R-TOY-010") == "NEEDS_INFO"
    assert _status(result.after, "R-TOY-010") == "APPLICABLE"
    assert "ACT-INFO-A-TOY-013" in _action_ids(result.before)
    assert "ACT-INFO-A-TOY-013" not in _action_ids(result.after)
    assert "ACT-RULE-R-TOY-010-TESTS" in _action_ids(result.after)

    removed = {
        d.action_id for d in result.delta.action_changes if d.change_type is ChangeType.REMOVED
    }
    added = {
        d.action_id for d in result.delta.action_changes if d.change_type is ChangeType.ADDED
    }
    assert "ACT-INFO-A-TOY-013" in removed
    assert "ACT-RULE-R-TOY-010-TESTS" in added
    assert result.delta.counts["action_removed"] >= 1
    assert result.delta.counts["action_added"] >= 1
    assert result.delta.counts["risk_modified"] >= 1


# =========================================================================== #
# 19. R-ELEC-012 safety lock
# =========================================================================== #
def test_r_elec_012_never_becomes_a_transport_obligation() -> None:
    facts = _elec012_facts("lithium ion")
    result = _ENGINE.run(_resolved(), facts, {"A-ELEC-011": "lithium metal"})

    for side in (result.before, result.after):
        assert _status(side, "R-ELEC-012") == "REVIEW_REQUIRED"
        assert _reason(side, "R-ELEC-012") == ["TRIGGER_LOGIC_NOT_MODELED"]
        assert _risk_levels(side).get("R-ELEC-012") != "HIGH"
        assert not [
            item
            for item in side.actions.items
            if item.rule_id == "R-ELEC-012" and item.action_type.value == "CURRENT_OBLIGATION"
        ]
    # the chemistry override does change the recorded fact, but never creates a transport verdict
    assert result.overrides[0].before_value == "lithium ion"
    assert result.overrides[0].after_value == "lithium metal"
    assert not [
        d
        for d in result.delta.risk_changes
        if d.rule_id == "R-ELEC-012" and d.change_type is ChangeType.ADDED
    ]


# =========================================================================== #
# 20. Serialization
# =========================================================================== #
def test_serialization_is_stable_sorted_and_json_safe() -> None:
    facts = _facts(("A-TOY-001", "A-TOY-005", "A-TOY-012", "A-TOY-013", "A-TOY-014", "A-TOY-015"))
    result = _ENGINE.run(
        _resolved("childrens_toys"),
        facts,
        [WhatIfOverride(attribute_id="A-TOY-015", value=False),
         WhatIfOverride(attribute_id="A-TOY-012", value=False)],
    )
    payload = result.to_dict()
    assert set(payload) == {
        "assessed", "category", "hypothetical", "overrides", "before", "after", "delta", "notes",
    }
    assert payload["hypothetical"] is True
    assert payload["before"] is not None and payload["after"] is not None
    assert [c["attribute_id"] for c in payload["overrides"]] == ["A-TOY-012", "A-TOY-015"]
    delta = payload["delta"]
    assert [d["rule_id"] for d in delta["rule_changes"]] == sorted(
        d["rule_id"] for d in delta["rule_changes"]
    )
    assert [d["action_id"] for d in delta["action_changes"]] == sorted(
        d["action_id"] for d in delta["action_changes"]
    )
    keys = set(delta["counts"])
    assert {"rule_changes", "risk_changes", "cost_changes", "action_changes"} <= keys
    assert {"rule_added", "risk_removed", "cost_modified", "action_added"} <= keys
    text = json.dumps(payload)
    for token in ("NaN", "Infinity", "-Infinity", "0x"):
        assert token not in text
    assert json.loads(text) == payload
    # BEFORE and AFTER are clearly distinguishable
    assert payload["before"]["applicability"] != payload["after"]["applicability"]


def test_serialization_round_trips_through_pydantic() -> None:
    result = _ENGINE.run(_resolved(), _elec002_facts(True), {"A-ELEC-002": False})
    dumped = result.to_dict()
    assert what_if_module.WhatIfResult.model_validate(dumped).to_dict() == dumped


# =========================================================================== #
# 21. No provider / network / environment dependency
# =========================================================================== #
def test_module_has_no_provider_network_clock_or_entropy_dependency() -> None:
    source = inspect.getsource(what_if_module)
    modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    allowed_roots = {"__future__", "collections", "enum", "typing", "pydantic", "src"}
    for module in modules:
        assert module.split(".")[0] in allowed_roots, module
    for forbidden in (
        "import openai", "import requests", "import boto3", "import random", "import os",
        "import time", "import uuid", "import datetime", "from strands", "from src.agent",
        "from src.certification", "os.environ", "open(", "eval(", "exec(", "subprocess",
    ):
        assert forbidden not in source, forbidden


def test_run_accepts_only_structured_inputs() -> None:
    parameters = list(inspect.signature(WhatIfEngine.run).parameters)
    assert parameters == ["self", "category_result", "product_facts", "overrides"]
    for forbidden in ("text", "prose", "question", "scenario", "model", "llm", "agent"):
        assert forbidden not in parameters


def test_repository_and_service_are_injected_not_reconstructed() -> None:
    parameters = list(inspect.signature(WhatIfEngine.__init__).parameters)
    assert parameters == ["self", "analysis_service", "repository"]


# =========================================================================== #
# 22-24. Review status, dual category, error model
# =========================================================================== #
def test_what_if_does_not_change_review_status() -> None:
    """BEFORE/AFTER carry whatever the canonical service produced; What-if never rewrites it."""
    facts = _elec002_facts(True)
    result = _ENGINE.run(_resolved(), facts, {"A-ELEC-002": False})
    baseline = _SERVICE.analyze(_resolved(), facts)
    assert result.before.review.status == baseline.review.status
    assert result.before.review.triggers == baseline.review.triggers
    assert result.before.review.reviewer_actions == baseline.review.reviewer_actions
    assert result.after.review.status == _SERVICE.analyze(
        _resolved(), [f for f in facts if f.attribute_id != "A-ELEC-002"]
        + [ProductFact(attribute_id="A-ELEC-002", value=False, origin=FactOrigin.USER)]
    ).review.status
    # no field on the What-if contract can approve, resolve or dismiss human review
    assert "review" not in what_if_module.WhatIfResult.model_fields
    assert "review_status" not in what_if_module.WhatIfResult.model_fields


def test_dual_category_allows_both_category_attributes() -> None:
    facts = _facts(("A-TOY-001", "A-TOY-005", "A-TOY-012", "A-TOY-013", "A-TOY-014"))
    result = _ENGINE.run(_resolved("dual"), facts, {"A-TOY-012": False})
    assert result.category == "dual"
    assert _status(result.after, "R-TOY-010") == "NOT_APPLICABLE"


def test_configuration_error_type_exists_and_is_distinct() -> None:
    assert issubclass(WhatIfInputError, ValueError)
    assert issubclass(WhatIfConfigurationError, ValueError)
    assert not issubclass(WhatIfConfigurationError, WhatIfInputError)


def test_non_user_only_baseline_fact_is_not_canonical_for_before_value() -> None:
    facts = _facts(("A-TOY-012",), origins={"A-TOY-012": FactOrigin.CLASSIFIER})
    result = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-012": True})
    change = result.overrides[0]
    assert change.before_present is False
    assert change.before_value is None


# =========================================================================== #
# Final human-review bug fixes
# =========================================================================== #
@pytest.mark.parametrize(
    ("category", "value"),
    [
        ("small_consumer_electronics", "childrens_toys"),
        ("childrens_toys", "dual"),
        ("dual", "small_consumer_electronics"),
    ],
)
def test_category_attribute_cannot_be_overridden(category, value) -> None:
    with pytest.raises(WhatIfInputError) as excinfo:
        _ENGINE.run(_resolved(category), [], {"A-CMN-001": value})
    assert "product category cannot be changed by What-if v1" in str(excinfo.value)
    # the rejected value is never echoed back to the caller
    assert value not in str(excinfo.value)

    with pytest.raises(WhatIfInputError):
        _ENGINE.run(
            _resolved(category), [], [WhatIfOverride(attribute_id="A-CMN-001", value=value)]
        )


def test_category_attribute_override_is_rejected_even_with_other_overrides() -> None:
    facts = _facts(("A-TOY-012",))
    with pytest.raises(WhatIfInputError):
        _ENGINE.run(
            _resolved("childrens_toys"),
            facts,
            [WhatIfOverride(attribute_id="A-TOY-012", value=False),
             WhatIfOverride(attribute_id="A-CMN-001", value="dual")],
        )


@pytest.mark.parametrize(
    "category_result",
    [
        CategoryResult(
            category="small_consumer_electronics",
            category_source=CategorySource.AGENT_GENERATED,
            category_status=CategoryStatus.RESOLVED,
        ),
        CategoryResult(
            category="unsupported",
            category_source=CategorySource.HUMAN_CONFIRMED,
            category_status=CategoryStatus.RESOLVED,
        ),
        CategoryResult(
            category="uncertain",
            category_source=CategorySource.HUMAN_CONFIRMED,
            category_status=CategoryStatus.RESOLVED,
        ),
        CategoryResult(
            category="not_a_category",
            category_source=CategorySource.HUMAN_CONFIRMED,
            category_status=CategoryStatus.RESOLVED,
        ),
        CategoryResult(
            category=None,
            category_source=CategorySource.HUMAN_CONFIRMED,
            category_status=CategoryStatus.RESOLVED,
        ),
    ],
)
def test_inconsistent_resolved_category_results_fail_closed(category_result) -> None:
    with pytest.raises(WhatIfConfigurationError) as excinfo:
        _ENGINE.run(category_result, _elec002_facts(True), {"A-ELEC-002": False})
    assert "RESOLVED what-if category" in str(excinfo.value)
    # the boundary stops it first: no downstream cost/engine configuration error leaks out
    assert not isinstance(excinfo.value, CostConfigurationError)


def test_resolved_supported_category_still_works() -> None:
    for category in ("childrens_toys", "small_consumer_electronics", "dual"):
        result = _ENGINE.run(_resolved(category), [], {})
        assert result.assessed is True
        assert result.category == category


def test_risk_delta_reports_a_missing_information_shrink() -> None:
    """Same level and same deciding reason, but the canonical missing list shrinks."""
    facts = _facts(("A-TOY-001", "A-TOY-005", "A-TOY-012"))  # A-TOY-013 and A-TOY-014 missing
    result = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-013": True})

    before_item = next(i for i in result.before.risk.items if i.rule_id == "R-TOY-010")
    after_item = next(i for i in result.after.risk.items if i.rule_id == "R-TOY-010")
    assert before_item.risk_level is after_item.risk_level is RiskLevel.REVIEW
    assert before_item.reason_code is after_item.reason_code is RiskReasonCode.MISSING_REQUIRED_FACTS
    assert sorted(before_item.missing_attribute_ids) == ["A-TOY-013", "A-TOY-014"]
    assert sorted(after_item.missing_attribute_ids) == ["A-TOY-014"]

    delta = next(d for d in result.delta.risk_changes if d.rule_id == "R-TOY-010")
    assert delta.change_type is ChangeType.MODIFIED
    assert delta.before_level == delta.after_level == "REVIEW"
    assert delta.before_reason_code == delta.after_reason_code == "MISSING_REQUIRED_FACTS"
    assert delta.before_missing_attribute_ids == ["A-TOY-013", "A-TOY-014"]
    assert delta.after_missing_attribute_ids == ["A-TOY-014"]
    assert delta.before_reason_codes_ordered == ["MISSING_REQUIRED_FACTS"]
    assert delta.after_reason_codes_ordered == ["MISSING_REQUIRED_FACTS"]


def test_unchanged_risk_items_still_produce_no_delta() -> None:
    facts = _facts(_TOY010_REQUIRED, overrides={"A-TOY-012": True})
    result = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-005": True})
    assert not [d for d in result.delta.risk_changes if d.rule_id == "R-TOY-010"]


def test_contradictory_baseline_facts_are_never_a_canonical_before_value() -> None:
    ordered = [
        ProductFact(attribute_id="A-TOY-012", value=True),
        ProductFact(attribute_id="A-TOY-012", value=False),
    ]
    reversed_order = list(reversed(ordered))
    first = _ENGINE.run(_resolved("childrens_toys"), ordered, {"A-TOY-012": False})
    second = _ENGINE.run(_resolved("childrens_toys"), reversed_order, {"A-TOY-012": False})
    for result in (first, second):
        change = result.overrides[0]
        assert change.before_present is False
        assert change.before_value is None
    # the canonical metadata does not depend on fact order
    assert [c.model_dump(mode="json") for c in first.overrides] == [
        c.model_dump(mode="json") for c in second.overrides
    ]
    assert first.to_dict() == second.to_dict()


def test_invalid_baseline_fact_is_never_a_canonical_before_value() -> None:
    facts = [ProductFact(attribute_id="A-TOY-012", value="yes")]
    result = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-012": False})
    change = result.overrides[0]
    assert change.before_present is False
    assert change.before_value is None


def test_valid_baseline_fact_is_the_canonical_before_value() -> None:
    facts = [ProductFact(attribute_id="A-TOY-012", value=True)]
    result = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-012": False})
    change = result.overrides[0]
    assert change.before_present is True
    assert change.before_value is True


@pytest.mark.parametrize("payload", [[{"value": True}], ["bad"], [123], [None], [["A-TOY-012"]]])
def test_malformed_sequence_overrides_raise_input_error(payload) -> None:
    with pytest.raises(WhatIfInputError) as excinfo:
        _ENGINE.run(_resolved("childrens_toys"), _facts(("A-TOY-012",)), payload)
    assert not isinstance(excinfo.value, ValidationError)
    assert "malformed hypothetical override" in str(excinfo.value)
    assert "Traceback" not in str(excinfo.value)


def test_mapping_overrides_still_behave_normally() -> None:
    facts = _facts(_TOY010_REQUIRED, overrides={"A-TOY-012": True})
    result = _ENGINE.run(_resolved("childrens_toys"), facts, {"A-TOY-012": False})
    assert result.assessed is True
    assert result.overrides[0].attribute_id == "A-TOY-012"
    assert result.overrides[0].before_value is True
