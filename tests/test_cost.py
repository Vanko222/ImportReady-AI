"""Cost Engine (Cost v1, Phase 1) offline tests.

Deterministic, offline, no model and no network. Real approved-data locks read the approved
repository; malformed/configuration cases use an in-memory stub repository, so no data file is
ever edited to manufacture an invalid case.

Group map: A model contract, B real-data inventory, C DIRECT, D PLANNING_ONLY, E QUOTE_REQUIRED,
F DISPLAY_ONLY, G category semantics, H unsupported safety lock, I platform safety lock, J no-data,
K fail-closed validation, L determinism, M aggregation safety, N no dependencies, O no prose
decisions.
"""

from __future__ import annotations

import ast
import json
import inspect
from types import SimpleNamespace

import pytest

from src.models import CalculatorUse, ComplianceCost, EvidenceStatus
from src.repositories.compliance_repository import JsonComplianceRepository
from src.services import cost as cost_module
from src.services.cost import (
    CostAssessment,
    CostCalculationStatus,
    CostConfigurationError,
    CostEngine,
    CostItem,
    _cost_item,
    unassessed_cost_assessment,
)
from src.state import KnowledgeSnapshot

S = CostCalculationStatus
DIRECT = CalculatorUse.DIRECT
PLANNING_ONLY = CalculatorUse.PLANNING_ONLY
QUOTE_REQUIRED = CalculatorUse.QUOTE_REQUIRED
DISPLAY_ONLY = CalculatorUse.DISPLAY_ONLY

_real_repo = JsonComplianceRepository()


@pytest.fixture(scope="module")
def real_repo() -> JsonComplianceRepository:
    return _real_repo


@pytest.fixture(scope="module")
def real_engine(real_repo: JsonComplianceRepository) -> CostEngine:
    return CostEngine(real_repo)


# --------------------------------------------------------------------------- #
# Stub repository + builders
# --------------------------------------------------------------------------- #
class StubRepository:
    """In-memory ``ComplianceRepository`` for controlled cost fixtures."""

    def __init__(self, costs=(), category_map=None) -> None:
        self._costs = list(costs)
        self._category_map = dict(category_map or {})
        self.calls: list[str] = []

    def get_attribute(self, attribute_id: str):
        return None

    def get_rule(self, rule_id: str):
        return None

    def get_source(self, source_id: str):
        return None

    def get_cost(self, cost_id: str):
        for cost in self._costs:
            if cost.cost_id == cost_id:
                return cost
        return None

    def get_rules_for_category(self, category: str):
        return []

    def get_costs_for_category(self, category: str):
        self.calls.append(category)
        if self._category_map:
            return list(self._category_map.get(category, []))
        return [c for c in self._costs if c.category == category]

    def get_known_gaps(self):
        return []

    def get_pending_policy_updates(self):
        return []

    def validate_references(self) -> None:
        return None

    def knowledge_snapshot(self) -> KnowledgeSnapshot:
        return KnowledgeSnapshot(schema_version="1.0", generated_at="2026-09-13T00:00:00Z")


def make_cost(
    cost_id: str = "C-TEST-001",
    *,
    category: str = "childrens_toys",
    cost_item: str = "Test cost item",
    calculator_use: CalculatorUse = DIRECT,
    exact_amount: float | None = 100.0,
    low_amount: float | None = None,
    high_amount: float | None = None,
    currency: str = "USD",
    evidence_status: EvidenceStatus = EvidenceStatus.VERIFIED,
    price_status: str = "PUBLIC_PRICE",
    confidence: str | None = "HIGH",
    pricing_basis: str = "per device",
    source_id: str = "S001",
    additional_source_ids=(),
    source_date: str = "2026-01-01",
    checked_date: str = "2026-09-09",
    notes: str = "test note",
) -> ComplianceCost:
    return ComplianceCost(
        cost_id=cost_id,
        category=category,
        cost_item=cost_item,
        exact_amount=exact_amount,
        low_amount=low_amount,
        high_amount=high_amount,
        currency=currency,
        pricing_basis=pricing_basis,
        scope_included="test scope",
        important_exclusions="test exclusions",
        source_id=source_id,
        additional_source_ids=list(additional_source_ids),
        source_date=source_date,
        checked_date=checked_date,
        price_status=price_status,
        evidence_status=evidence_status,
        confidence=confidence,
        calculator_use=calculator_use,
        notes=notes,
    )


def make_raw_record(cost_id: str = "C-RAW-001", **overrides):
    """A non-pydantic canonical record stand-in.

    The strict ``ComplianceCost`` model forbids a few defects the engine must still catch (for
    example ``source_id=None`` or a non-finite amount), so low-level validation tests feed the
    engine a plain attribute object instead of mutating production data.
    """
    fields: dict[str, object] = {
        "cost_id": cost_id,
        "category": "childrens_toys",
        "cost_item": "Raw cost item",
        "exact_amount": 100.0,
        "low_amount": None,
        "high_amount": None,
        "currency": "USD",
        "pricing_basis": "per device",
        "scope_included": "scope",
        "important_exclusions": "exclusions",
        "price_status": "PUBLIC_PRICE",
        "evidence_status": EvidenceStatus.VERIFIED,
        "confidence": "HIGH",
        "calculator_use": DIRECT,
        "source_id": "S001",
        "additional_source_ids": [],
        "source_date": "2026-01-01",
        "checked_date": "2026-09-09",
        "notes": "raw note",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _ids(assessment: CostAssessment) -> list[str]:
    return [item.cost_id for item in assessment.items]


def _by_id(assessment: CostAssessment) -> dict[str, CostItem]:
    return {item.cost_id: item for item in assessment.items}


def _all_amounts(assessment: CostAssessment) -> list[float]:
    amounts: list[float] = []
    for item in assessment.items:
        for value in (item.exact_amount, item.low_amount, item.high_amount):
            if value is not None:
                amounts.append(value)
    return amounts


# =========================================================================== #
# A. Model contract
# =========================================================================== #
def test_a_status_enum_values_are_exact() -> None:
    assert {member.value for member in S} == {
        "DIRECT", "PLANNING_ONLY", "QUOTE_REQUIRED", "DISPLAY_ONLY",
    }
    assert not hasattr(S, "NEEDS_INPUT")


def test_a_assessment_has_no_total_or_monetary_aggregate_field() -> None:
    fields = set(CostAssessment.model_fields)
    assert fields == {
        "assessed", "category", "items", "counts", "currencies", "total_available",
        "total_unavailable_reason", "missing_inputs", "notes",
    }
    for forbidden in ("total", "subtotal", "converted_total", "fx_rate", "estimated_total",
                      "payable_total", "landed_cost"):
        assert forbidden not in fields
    # only two non-numeric *flags* mention a total, and both default to False/None
    assert CostAssessment.model_fields["total_available"].annotation is bool
    assert CostAssessment.model_fields["total_available"].default is False
    assert CostAssessment.model_fields["total_unavailable_reason"].default is None


def test_a_item_has_no_computed_monetary_field() -> None:
    assert set(CostItem.model_fields) == {
        "cost_id", "category", "cost_item", "calculation_status", "exact_amount",
        "low_amount", "high_amount", "currency", "pricing_basis", "scope_included",
        "important_exclusions", "price_status", "evidence_status", "confidence",
        "source_ids", "source_date", "checked_date", "notes", "title",
    }


def test_a_unassessed_serialization_is_strict_and_complete() -> None:
    dumped = unassessed_cost_assessment("no_data").model_dump(mode="json")
    assert dumped == {
        "assessed": False,
        "category": None,
        "items": [],
        "counts": {"DIRECT": 0, "PLANNING_ONLY": 0, "QUOTE_REQUIRED": 0, "DISPLAY_ONLY": 0},
        "currencies": [],
        "total_available": False,
        "total_unavailable_reason": (
            "No cost total is available because no approved cost reference records exist for this "
            "confirmed category."
        ),
        "missing_inputs": [],
        "notes": ["No approved cost reference records exist for the confirmed category."],
    }
    assert json.loads(json.dumps(dumped)) == dumped


@pytest.mark.parametrize(
    "reason", ["unresolved", "unsupported", "unconfirmed", "no_data"]
)
def test_a_unassessed_helper_uses_a_closed_vocabulary(reason: str) -> None:
    assessment = unassessed_cost_assessment(reason, category="childrens_toys")
    assert assessment.assessed is False
    assert assessment.items == []
    assert assessment.category == "childrens_toys"
    assert assessment.total_available is False
    assert assessment.total_unavailable_reason is not None
    assert len(assessment.notes) == 1
    wording = " ".join(assessment.notes + [assessment.total_unavailable_reason]).lower()
    for forbidden in ("zero", "$0", "free", "no costs", "not applicable", "nothing to pay"):
        assert forbidden not in wording, forbidden


def test_a_unassessed_helper_rejects_arbitrary_reason() -> None:
    with pytest.raises(ValueError):
        unassessed_cost_assessment("whatever the caller wants")


def test_a_counts_always_cover_all_four_statuses(real_engine: CostEngine) -> None:
    for category in ("childrens_toys", "small_consumer_electronics", "dual"):
        assessment = real_engine.assess(category)
        assert set(assessment.counts) == {
            "DIRECT", "PLANNING_ONLY", "QUOTE_REQUIRED", "DISPLAY_ONLY",
        }
        assert sum(assessment.counts.values()) == len(assessment.items)
        assert "NOT_APPLICABLE" not in assessment.counts


def test_a_currencies_are_sorted_and_unique(real_engine: CostEngine) -> None:
    dual = real_engine.assess("dual")
    assert dual.currencies == sorted(set(dual.currencies)) == ["HKD", "USD"]


# =========================================================================== #
# B. Real approved-data inventory
# =========================================================================== #
def test_b_raw_dataset_inventory(real_repo: JsonComplianceRepository) -> None:
    assert len(real_repo.costs) == 32
    assert real_repo.counts["costs"] == 32


def test_b_category_inventory(real_engine: CostEngine) -> None:
    toys = real_engine.assess("childrens_toys")
    elec = real_engine.assess("small_consumer_electronics")
    dual = real_engine.assess("dual")
    assert len(toys.items) == 14
    assert len(elec.items) == 16
    assert len(dual.items) == 30
    assert toys.counts == {"DIRECT": 5, "PLANNING_ONLY": 4, "QUOTE_REQUIRED": 2, "DISPLAY_ONLY": 3}
    assert elec.counts == {"DIRECT": 1, "PLANNING_ONLY": 9, "QUOTE_REQUIRED": 5, "DISPLAY_ONLY": 1}
    assert dual.counts == {"DIRECT": 6, "PLANNING_ONLY": 13, "QUOTE_REQUIRED": 7, "DISPLAY_ONLY": 4}


def test_b_dual_is_the_deduplicated_union(real_engine: CostEngine) -> None:
    toys = set(_ids(real_engine.assess("childrens_toys")))
    elec = set(_ids(real_engine.assess("small_consumer_electronics")))
    dual = _ids(real_engine.assess("dual"))
    assert set(dual) == toys | elec
    assert len(dual) == len(set(dual)) == 30


def test_b_special_records_are_excluded_from_product_retrieval(
    real_repo: JsonComplianceRepository, real_engine: CostEngine
) -> None:
    # Both records exist in the approved data ...
    assert real_repo.get_cost("C-P-001") is not None
    assert real_repo.get_cost("COST-005") is not None
    assert real_repo.get_cost("C-P-001").category == "amazon_us_platform"
    assert real_repo.get_cost("COST-005").category == "unsupported"
    # ... and are never surfaced by a product-category assessment.
    for category in ("childrens_toys", "small_consumer_electronics", "dual"):
        ids = _ids(real_engine.assess(category))
        assert "C-P-001" not in ids
        assert "COST-005" not in ids
        assert {item.category for item in real_engine.assess(category).items} <= {
            "childrens_toys", "small_consumer_electronics",
        }


# =========================================================================== #
# C. DIRECT
# =========================================================================== #
_DIRECT_LOCK = {
    "C-T-001": (5600.0, "HKD"),
    "C-T-002": (3500.0, "HKD"),
    "C-T-003": (3550.0, "HKD"),
    "C-T-004": (4550.0, "HKD"),
    "COST-001": (0.0, "USD"),
    "COST-002": (35.0, "USD"),
}


def test_c_direct_records_are_locked_verbatim(real_engine: CostEngine) -> None:
    dual = _by_id(real_engine.assess("dual"))
    for cost_id, (amount, currency) in _DIRECT_LOCK.items():
        item = dual[cost_id]
        assert item.calculation_status is S.DIRECT
        assert item.exact_amount == amount
        assert item.currency == currency
        assert item.low_amount is None and item.high_amount is None
        assert item.evidence_status == "VERIFIED"
        assert item.title == f"Cost reference: {cost_id}"


def test_c_direct_never_creates_a_range_rounding_or_conversion(real_engine: CostEngine) -> None:
    for category in ("childrens_toys", "small_consumer_electronics", "dual"):
        assessment = real_engine.assess(category)
        for item in assessment.items:
            if item.calculation_status is not S.DIRECT:
                continue
            assert item.exact_amount is not None
            assert item.low_amount is None and item.high_amount is None
            # exact pass-through: no rounding artefact and no FX conversion
            assert item.exact_amount == float(item.exact_amount)
            assert item.currency in ("USD", "HKD")


def test_c_amounts_are_verbatim_from_the_repository(
    real_repo: JsonComplianceRepository, real_engine: CostEngine
) -> None:
    for category in ("childrens_toys", "small_consumer_electronics"):
        assessment = real_engine.assess(category)
        assert len(assessment.items) == len(real_repo.get_costs_for_category(category))
        expected = {c.cost_id: c for c in real_repo.get_costs_for_category(category)}
        for item in assessment.items:
            record = expected[item.cost_id]
            assert item.exact_amount == record.exact_amount
            assert item.low_amount == record.low_amount
            assert item.high_amount == record.high_amount
            assert item.currency == record.currency
            assert item.price_status == record.price_status
            assert item.notes == record.notes


# =========================================================================== #
# D. PLANNING_ONLY
# =========================================================================== #
def test_d_planning_only_stays_planning_only(real_engine: CostEngine) -> None:
    planning = [i for i in real_engine.assess("dual").items if i.calculation_status is S.PLANNING_ONLY]
    assert len(planning) == 13
    for item in planning:
        assert item.calculation_status is not S.DIRECT
        assert item.exact_amount is not None or (item.low_amount is not None and item.high_amount is not None)
    assert any(i.low_amount is not None and i.high_amount is not None for i in planning)
    assert any(i.exact_amount is not None for i in planning)


def test_d_planning_only_notes_are_flagged(real_engine: CostEngine) -> None:
    assessment = real_engine.assess("small_consumer_electronics")
    assert any("planning-only" in note for note in assessment.notes)
    assert assessment.total_available is False


# =========================================================================== #
# E. QUOTE_REQUIRED
# =========================================================================== #
def test_e_quote_required_has_no_amount_and_never_becomes_zero(real_engine: CostEngine) -> None:
    assessment = real_engine.assess("dual")
    quotes = [i for i in assessment.items if i.calculation_status is S.QUOTE_REQUIRED]
    assert len(quotes) == 7
    for item in quotes:
        assert item.exact_amount is None
        assert item.low_amount is None
        assert item.high_amount is None
        assert item.evidence_status == "NOT_FOUND"


def test_e_quote_required_note_is_present(real_engine: CostEngine) -> None:
    assessment = real_engine.assess("childrens_toys")
    assert any("quote-required" in note for note in assessment.notes)


def test_e_quote_required_amounts_are_never_normalized_to_zero(real_engine: CostEngine) -> None:
    dual = real_engine.assess("dual")
    quotes = [i for i in dual.items if i.calculation_status is S.QUOTE_REQUIRED]
    zero_quotes = [i for i in quotes if 0.0 in (i.exact_amount, i.low_amount, i.high_amount)]
    assert zero_quotes == []
    # the only approved zero amount is the DIRECT ASTM read-only access record
    assert [i.cost_id for i in dual.items if 0.0 in (i.exact_amount,)] == ["COST-001"]
    assert _by_id(dual)["COST-001"].calculation_status is S.DIRECT


# =========================================================================== #
# F. DISPLAY_ONLY
# =========================================================================== #
def test_f_display_only_is_first_class_and_non_calculating(real_engine: CostEngine) -> None:
    assessment = real_engine.assess("dual")
    display = [i for i in assessment.items if i.calculation_status is S.DISPLAY_ONLY]
    assert len(display) == 4
    assert {i.cost_id for i in display} == {"C-E-007", "C-T-010", "COST-006", "COST-007"}
    for item in display:
        assert item.calculation_status is not S.PLANNING_ONLY
        assert item.calculation_status is not S.DIRECT
    assert any("display-only" in note for note in assessment.notes)


def test_f_display_only_reference_values_are_preserved(real_engine: CostEngine) -> None:
    items = _by_id(real_engine.assess("dual"))
    assert items["C-E-007"].exact_amount == 5000.0
    assert items["COST-007"].exact_amount == 2000.0
    assert items["C-T-010"].low_amount == 538.0 and items["C-T-010"].high_amount == 758.0


def test_f_display_only_never_enters_a_total(real_engine: CostEngine) -> None:
    assessment = real_engine.assess("dual")
    assert assessment.total_available is False
    assert assessment.total_unavailable_reason is not None
    assert not hasattr(assessment, "total")


# =========================================================================== #
# G. Category semantics
# =========================================================================== #
def test_g_supported_categories_are_assessed(real_engine: CostEngine) -> None:
    for category, size in (("childrens_toys", 14), ("small_consumer_electronics", 16), ("dual", 30)):
        assessment = real_engine.assess(category)
        assert assessment.assessed is True
        assert assessment.category == category
        assert len(assessment.items) == size
        assert assessment.total_available is False


@pytest.mark.parametrize("category", [None, "", "   "])
def test_g_missing_category_is_unresolved(real_engine: CostEngine, category) -> None:
    assessment = real_engine.assess(category)
    assert assessment.assessed is False
    assert assessment.items == []
    assert assessment.counts == {s.value: 0 for s in S}
    assert assessment.category is None
    assert "unresolved" in assessment.notes[0].lower()


def test_g_uncertain_and_unsupported_are_unassessed(real_engine: CostEngine) -> None:
    uncertain = real_engine.assess("uncertain")
    assert uncertain.assessed is False and uncertain.items == []
    assert uncertain.category == "uncertain"
    unsupported = real_engine.assess("unsupported")
    assert unsupported.assessed is False and unsupported.items == []
    assert unsupported.category == "unsupported"
    assert "not available" in unsupported.notes[0].lower()


@pytest.mark.parametrize("category", ["amazon_us_platform", "not_a_category", "toys", "common"])
def test_g_unknown_category_fails_closed(real_engine: CostEngine, category: str) -> None:
    with pytest.raises(CostConfigurationError):
        real_engine.assess(category)


def test_g_closed_category_vocabulary() -> None:
    assert set(cost_module._PRODUCT_CATEGORY_TARGETS) == {
        "childrens_toys", "small_consumer_electronics", "dual", "uncertain", "unsupported",
    }


# =========================================================================== #
# H. Unsupported safety lock
# =========================================================================== #
def test_h_unsupported_never_exposes_cost_005(
    real_repo: JsonComplianceRepository, real_engine: CostEngine
) -> None:
    assessment = real_engine.assess("unsupported")
    assert assessment.assessed is False
    assert assessment.items == []
    assert _ids(assessment) == []
    assert assessment.counts == {s.value: 0 for s in S}
    assert assessment.total_available is False
    assert real_repo.get_cost("COST-005") is not None  # the record exists but is not permission
    serialized = json.dumps(assessment.model_dump(mode="json"))
    assert "COST-005" not in serialized
    assert "36.8" not in serialized


def test_h_unsupported_does_not_read_the_unsupported_category() -> None:
    record = make_cost("COST-005", category="unsupported", exact_amount=36.8)
    repo = StubRepository(costs=[record])
    assessment = CostEngine(repo).assess("unsupported")
    assert assessment.assessed is False
    assert "COST-005" not in json.dumps(assessment.model_dump(mode="json"))
    assert "unsupported" not in repo.calls  # no data fetch happened at all


# =========================================================================== #
# I. Platform safety lock
# =========================================================================== #
def test_i_platform_record_is_unreachable(real_repo: JsonComplianceRepository, real_engine: CostEngine) -> None:
    assert real_repo.get_cost("C-P-001").category == "amazon_us_platform"
    for category in ("childrens_toys", "small_consumer_electronics", "dual", "unsupported"):
        serialized = json.dumps(real_engine.assess(category).model_dump(mode="json"))
        assert "C-P-001" not in serialized
        assert "amazon_us_platform" not in serialized
    with pytest.raises(CostConfigurationError):
        real_engine.assess("amazon_us_platform")


# =========================================================================== #
# J. No data
# =========================================================================== #
def test_j_supported_category_with_zero_records_is_unassessed_not_zero() -> None:
    repo = StubRepository(category_map={"childrens_toys": []})
    assessment = CostEngine(repo).assess("childrens_toys")
    assert assessment.assessed is False
    assert assessment.items == []
    assert assessment.total_available is False
    assert assessment.currencies == []
    assert assessment.counts == {s.value: 0 for s in S}
    assert "No approved cost reference records exist" in assessment.notes[0]


def test_j_dual_with_zero_records_is_unassessed() -> None:
    repo = StubRepository(category_map={"childrens_toys": [], "small_consumer_electronics": []})
    assessment = CostEngine(repo).assess("dual")
    assert assessment.assessed is False and assessment.items == []
    assert assessment.category == "dual"


# =========================================================================== #
# K. Fail-closed validation
# =========================================================================== #
def _assess_one(cost: ComplianceCost):
    return CostEngine(StubRepository(costs=[cost])).assess(cost.category)


def test_k_blank_cost_id_fails_closed() -> None:
    with pytest.raises(CostConfigurationError):
        _assess_one(make_cost(""))


def test_k_duplicate_cost_id_fails_closed() -> None:
    first = make_cost("C-DUP-001")
    second = make_cost("C-DUP-001", cost_item="Second record")
    with pytest.raises(CostConfigurationError):
        CostEngine(StubRepository(costs=[first, second])).assess("childrens_toys")


def test_k_blank_currency_fails_closed() -> None:
    with pytest.raises(CostConfigurationError):
        _assess_one(make_cost(currency=""))


def test_k_exact_and_range_together_fails_closed() -> None:
    with pytest.raises(CostConfigurationError):
        _assess_one(make_cost(calculator_use=PLANNING_ONLY, exact_amount=10.0, low_amount=1.0, high_amount=2.0))


@pytest.mark.parametrize("low,high", [(1.0, None), (None, 2.0)])
def test_k_partial_range_fails_closed(low, high) -> None:
    with pytest.raises(CostConfigurationError):
        _assess_one(make_cost(calculator_use=PLANNING_ONLY, exact_amount=None, low_amount=low, high_amount=high))


def test_k_low_greater_than_high_fails_closed() -> None:
    with pytest.raises(CostConfigurationError):
        _assess_one(make_cost(calculator_use=PLANNING_ONLY, exact_amount=None, low_amount=500.0, high_amount=100.0))


@pytest.mark.parametrize("kwargs", [
    {"exact_amount": -1.0},
    {"calculator_use": PLANNING_ONLY, "exact_amount": None, "low_amount": -5.0, "high_amount": 10.0},
    {"calculator_use": PLANNING_ONLY, "exact_amount": None, "low_amount": 5.0, "high_amount": -10.0},
])
def test_k_negative_amounts_fail_closed(kwargs) -> None:
    kwargs.setdefault("calculator_use", DIRECT)
    with pytest.raises(CostConfigurationError):
        _assess_one(make_cost(**kwargs))


def test_k_direct_without_exact_amount_fails_closed() -> None:
    with pytest.raises(CostConfigurationError):
        _assess_one(make_cost(calculator_use=DIRECT, exact_amount=None))


def test_k_direct_with_range_fails_closed() -> None:
    with pytest.raises(CostConfigurationError):
        _assess_one(make_cost(calculator_use=DIRECT, exact_amount=None, low_amount=1.0, high_amount=2.0))


@pytest.mark.parametrize("evidence", [EvidenceStatus.UNVERIFIED, EvidenceStatus.CONFLICT, EvidenceStatus.NOT_FOUND])
def test_k_direct_without_verified_evidence_fails_closed(evidence) -> None:
    with pytest.raises(CostConfigurationError):
        _assess_one(make_cost(calculator_use=DIRECT, evidence_status=evidence))


@pytest.mark.parametrize("kwargs", [
    {"exact_amount": 10.0},
    {"exact_amount": None, "low_amount": 1.0, "high_amount": 2.0},
])
def test_k_quote_required_with_amount_fails_closed(kwargs) -> None:
    with pytest.raises(CostConfigurationError):
        _assess_one(make_cost(
            calculator_use=QUOTE_REQUIRED,
            evidence_status=EvidenceStatus.NOT_FOUND,
            price_status="QUOTE_REQUIRED",
            **kwargs,
        ))


def test_k_not_found_with_amount_fails_closed() -> None:
    with pytest.raises(CostConfigurationError):
        _assess_one(make_cost(
            calculator_use=PLANNING_ONLY, evidence_status=EvidenceStatus.NOT_FOUND, exact_amount=10.0,
        ))


def test_k_category_mismatch_fails_closed() -> None:
    record = make_cost("C-MISMATCH-001", category="unsupported", exact_amount=1.0)
    repo = StubRepository(category_map={"childrens_toys": [record]})
    with pytest.raises(CostConfigurationError):
        CostEngine(repo).assess("childrens_toys")


def test_k_canonical_uncertainty_states_do_not_fail_closed(real_engine: CostEngine) -> None:
    """HKD, expired price status, old dates, null confidence and quote/display/planning states are
    canonical, not configuration errors."""
    items = _by_id(real_engine.assess("dual"))
    assert items["C-T-001"].currency == "HKD"
    assert items["C-T-010"].price_status == "EXPIRED_PROMOTIONAL_PRICE"
    assert items["COST-001"].confidence is None
    assert items["C-E-009"].calculation_status is S.QUOTE_REQUIRED
    assert items["C-E-007"].calculation_status is S.DISPLAY_ONLY
    assert items["C-T-005"].calculation_status is S.PLANNING_ONLY
    assert items["C-E-008"].source_date == "2024-07-30"


def test_k_stub_malformed_fixtures_raise_the_typed_error() -> None:
    record = make_cost("C-BAD-001", low_amount=1.0)
    with pytest.raises(CostConfigurationError):
        _cost_item(record)


# --------------------------------------------------------------------------- #
# K.1 Non-finite monetary values
# --------------------------------------------------------------------------- #
NAN = float("nan")
INF = float("inf")


@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_k1_direct_non_finite_exact_amount_fails_closed(value: float) -> None:
    with pytest.raises(CostConfigurationError):
        _assess_one(make_cost(calculator_use=DIRECT, exact_amount=value))


def test_k1_planning_only_non_finite_low_amount_fails_closed() -> None:
    with pytest.raises(CostConfigurationError):
        _assess_one(make_cost(
            calculator_use=PLANNING_ONLY, exact_amount=None, low_amount=NAN, high_amount=10.0,
        ))


@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_k1_planning_only_non_finite_high_amount_fails_closed(value: float) -> None:
    with pytest.raises(CostConfigurationError):
        _assess_one(make_cost(
            calculator_use=PLANNING_ONLY, exact_amount=None, low_amount=1.0, high_amount=value,
        ))


def test_k1_non_finite_survives_into_a_stub_repository_as_an_error() -> None:
    """A NaN amount must never reach an assessment, even via a whole-engine call."""
    repo = StubRepository(costs=[make_cost("C-NAN-001", exact_amount=NAN)])
    with pytest.raises(CostConfigurationError):
        CostEngine(repo).assess("childrens_toys")


@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_k1_non_finite_error_is_raised_before_comparisons(value: float) -> None:
    """NaN defeats ordinary comparisons, so finiteness must be checked first."""
    with pytest.raises(CostConfigurationError, match="is not a finite number"):
        _cost_item(make_cost(calculator_use=DIRECT, exact_amount=value))


def test_k1_valid_assessments_serialize_without_non_finite_tokens(real_engine: CostEngine) -> None:
    for category in ("childrens_toys", "small_consumer_electronics", "dual", "unsupported"):
        payload = json.dumps(real_engine.assess(category).model_dump(mode="json"))
        for token in ("NaN", "Infinity", "-Infinity"):
            assert token not in payload, token


def test_k1_finite_amounts_still_serialize_exactly() -> None:
    repo = StubRepository(costs=[
        make_cost("C-FIN-001", exact_amount=0.0),
        make_cost("C-FIN-002", exact_amount=35.0),
        make_cost("C-FIN-003", calculator_use=PLANNING_ONLY, exact_amount=None,
                  low_amount=300.0, high_amount=930.0),
    ])
    payload = CostEngine(repo).assess("childrens_toys").model_dump(mode="json")
    amounts = {item["cost_id"]: (item["exact_amount"], item["low_amount"], item["high_amount"])
               for item in payload["items"]}
    assert amounts == {
        "C-FIN-001": (0.0, None, None),
        "C-FIN-002": (35.0, None, None),
        "C-FIN-003": (None, 300.0, 930.0),
    }
    assert "NaN" not in json.dumps(payload)


# --------------------------------------------------------------------------- #
# K.2 Source provenance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("source_id", ["", "   ", "\t"])
def test_k2_blank_primary_source_id_fails_closed(source_id: str) -> None:
    with pytest.raises(CostConfigurationError, match="blank primary source_id"):
        _cost_item(make_cost("C-SRC-BAD-001", source_id=source_id))


def test_k2_missing_primary_source_id_fails_closed() -> None:
    with pytest.raises(CostConfigurationError, match="blank primary source_id"):
        _cost_item(make_raw_record("C-SRC-BAD-002", source_id=None))


@pytest.mark.parametrize("additional", [["S002", ""], ["S002", "   "], [""]])
def test_k2_blank_additional_source_id_fails_closed(additional) -> None:
    with pytest.raises(CostConfigurationError, match="blank additional_source_id"):
        _cost_item(make_cost("C-SRC-BAD-003", additional_source_ids=additional))


def test_k2_duplicate_source_ids_still_deduplicate() -> None:
    item = _cost_item(make_cost(
        "C-SRC-OK-001", source_id="S001", additional_source_ids=["S002", "S001", "S003", "S002"],
    ))
    assert item.source_ids == ["S001", "S002", "S003"]


def test_k2_real_data_sources_are_preserved(real_engine: CostEngine) -> None:
    items = _by_id(real_engine.assess("dual"))
    assert items["C-E-001"].source_ids == ["S058"]
    assert items["COST-002"].source_ids == ["S047"]
    for item in items.values():
        assert item.source_ids and item.source_ids[0].strip()


def test_k2_internal_builder_is_private() -> None:
    """The public API stays ``CostEngine.assess``; the low-level builder is internal only."""
    assert not hasattr(cost_module, "cost_item")
    assert hasattr(cost_module, "_cost_item")
    public = {name for name in dir(cost_module) if not name.startswith("_")}
    assert "cost_item" not in public
    assert public & {"CostEngine", "CostAssessment", "CostItem", "CostConfigurationError",
                     "CostCalculationStatus", "unassessed_cost_assessment"}


# =========================================================================== #
# L. Determinism
# =========================================================================== #
def test_l_repeated_calls_are_identical(real_engine: CostEngine) -> None:
    first = real_engine.assess("dual").model_dump(mode="json")
    second = real_engine.assess("dual").model_dump(mode="json")
    assert first == second


def test_l_items_are_ordered_by_cost_id(real_engine: CostEngine) -> None:
    for category in ("childrens_toys", "small_consumer_electronics", "dual"):
        ids = _ids(real_engine.assess(category))
        assert ids == sorted(ids)


def test_l_repository_record_order_does_not_affect_output(real_repo: JsonComplianceRepository) -> None:
    forward = [c for c in real_repo.costs if c.category == "childrens_toys"]
    repo_forward = StubRepository(category_map={"childrens_toys": forward})
    repo_reversed = StubRepository(category_map={"childrens_toys": list(reversed(forward))})
    assert (
        CostEngine(repo_forward).assess("childrens_toys").model_dump(mode="json")
        == CostEngine(repo_reversed).assess("childrens_toys").model_dump(mode="json")
    )


def test_l_source_ids_are_deduplicated_in_first_seen_order() -> None:
    record = make_cost(
        "C-SRC-001", source_id="S001", additional_source_ids=["S002", "S001", "S003", "S002"],
    )
    item = _cost_item(record)
    assert item.source_ids == ["S001", "S002", "S003"]


def test_l_dual_union_deduplicates_by_cost_id() -> None:
    record = make_cost("C-UNION-001", category="childrens_toys")
    mirrored = make_cost("C-UNION-001", category="small_consumer_electronics")
    repo = StubRepository(category_map={
        "childrens_toys": [record], "small_consumer_electronics": [mirrored],
    })
    assessment = CostEngine(repo).assess("dual")
    assert _ids(assessment) == ["C-UNION-001"]
    assert assessment.counts["DIRECT"] == 1


# =========================================================================== #
# M. Aggregation safety
# =========================================================================== #
def test_m_no_totals_for_any_fixture_shape() -> None:
    direct_only = StubRepository(costs=[
        make_cost("C-ONE-001", exact_amount=10.0),
        make_cost("C-ONE-002", exact_amount=20.0),
    ])
    one_currency = StubRepository(costs=[
        make_cost("C-ONE-003", exact_amount=10.0),
        make_cost("C-ONE-004", calculator_use=PLANNING_ONLY, exact_amount=None,
                  low_amount=1.0, high_amount=2.0),
    ])
    mixed_currency = StubRepository(costs=[
        make_cost("C-MIX-001", exact_amount=10.0, currency="USD"),
        make_cost("C-MIX-002", exact_amount=5600.0, currency="HKD"),
    ])
    for repo in (direct_only, one_currency, mixed_currency):
        assessment = CostEngine(repo).assess("childrens_toys")
        assert assessment.assessed is True
        assert assessment.total_available is False
        assert assessment.total_unavailable_reason == cost_module._TOTAL_UNAVAILABLE_WITH_ITEMS
        dumped = assessment.model_dump(mode="json")
        assert not any(
            key in dumped for key in ("total", "subtotal", "converted_total", "fx_rate",
                                      "estimated_total", "payable_total", "landed_cost")
        )


def test_m_mixed_currencies_are_preserved_not_converted() -> None:
    repo = StubRepository(costs=[
        make_cost("C-MIX-010", exact_amount=10.0, currency="USD"),
        make_cost("C-MIX-011", exact_amount=5600.0, currency="HKD"),
    ])
    assessment = CostEngine(repo).assess("childrens_toys")
    assert assessment.currencies == ["HKD", "USD"]
    assert _by_id(assessment)["C-MIX-011"].exact_amount == 5600.0
    assert assessment.total_available is False


def test_m_real_category_results_never_total(real_engine: CostEngine) -> None:
    for category in ("childrens_toys", "small_consumer_electronics", "dual", "unsupported"):
        assessment = real_engine.assess(category)
        assert assessment.total_available is False
        assert assessment.total_unavailable_reason is not None


def test_m_even_all_direct_records_do_not_produce_a_total() -> None:
    repo = StubRepository(costs=[
        make_cost("C-ALL-001", exact_amount=100.0),
        make_cost("C-ALL-002", exact_amount=200.0),
        make_cost("C-ALL-003", exact_amount=300.0),
    ])
    assessment = CostEngine(repo).assess("childrens_toys")
    assert assessment.counts["DIRECT"] == 3
    assert assessment.total_available is False
    assert 600.0 not in _all_amounts(assessment)


# =========================================================================== #
# N. No dependencies / no risk input
# =========================================================================== #
def test_n_engine_has_no_risk_applicability_or_model_input() -> None:
    parameters = list(inspect.signature(CostEngine.assess).parameters)
    assert parameters == ["self", "category"]
    for forbidden in ("risk", "risk_assessment", "level", "applicability", "rules", "text",
                      "prose", "summary", "model", "llm", "agent"):
        assert forbidden not in parameters, forbidden


def test_n_module_has_no_provider_network_or_service_imports() -> None:
    source = inspect.getsource(cost_module)
    modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert modules
    for module in modules:
        assert module == "__future__" or module.split(".")[0] in {"enum", "math", "typing", "pydantic"}
    for forbidden in ("import openai", "import requests", "import boto3", "import random",
                      "import os", "import time", "from strands", "from src.agent",
                      "from src.services.risk", "from src.services.applicability",
                      "from src.services.analysis", "from src.certification",
                      "os.environ", "open("):
        assert forbidden not in source, forbidden


def test_n_repository_is_injected_and_not_reconstructed() -> None:
    repo = StubRepository(costs=[make_cost("C-INJ-001")])
    engine = CostEngine(repo)
    assert engine.assess("childrens_toys").items
    assert repo.calls == ["childrens_toys"]


def test_n_dual_reads_exactly_the_two_product_categories() -> None:
    repo = StubRepository()
    CostEngine(repo).assess("dual")
    assert repo.calls == ["childrens_toys", "small_consumer_electronics"]


# =========================================================================== #
# O. No prose decisions
# =========================================================================== #
def test_o_prose_text_cannot_change_any_decision() -> None:
    plain = make_cost("C-PROSE-001", pricing_basis="per device", notes="plain note")
    flowery = make_cost(
        "C-PROSE-001",
        pricing_basis="add-on per response full transmitter, 12 hours, multiply by quantity 50",
        notes="This is mandatory, payable and required. Landed cost total = 999999 USD.",
        cost_item="full transmitter certification incl. every fee",
        price_status="QUOTE_REQUIRED",
    )
    first = _cost_item(plain)
    second = _cost_item(flowery)
    assert first.calculation_status is second.calculation_status is S.DIRECT
    assert first.exact_amount == second.exact_amount == 100.0
    assert first.low_amount == second.low_amount is None
    assert first.high_amount == second.high_amount is None
    assert first.currency == second.currency == "USD"
    # the canonical text itself is echoed verbatim, and nothing else changes
    assert second.pricing_basis == flowery.pricing_basis
    assert second.notes == flowery.notes
    assert second.cost_item == flowery.cost_item
    assert second.price_status == "QUOTE_REQUIRED"


def test_o_prose_text_cannot_change_assessment_flags() -> None:
    plain_repo = StubRepository(costs=[make_cost("C-PROSE-010")])
    prose_repo = StubRepository(costs=[make_cost("C-PROSE-010", pricing_basis="per hour", notes="landed cost $1")])
    plain = CostEngine(plain_repo).assess("childrens_toys").model_dump(mode="json")
    prose = CostEngine(prose_repo).assess("childrens_toys").model_dump(mode="json")
    plain["items"][0]["pricing_basis"] = prose["items"][0]["pricing_basis"] = "X"
    plain["items"][0]["notes"] = prose["items"][0]["notes"] = "X"
    assert plain == prose
    assert plain["assessed"] is True
    assert plain["total_available"] is False
