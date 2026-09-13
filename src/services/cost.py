"""Deterministic Cost Engine (Cost v1, Phase 1: engine + models only).

Turns the approved ``compliance_costs.json`` records - read through the existing
``ComplianceRepository`` - into a traceable, **category-level** :class:`CostAssessment`. The engine
is **purely deterministic**:

* no network, no credential, no I/O, no environment variable, no clock, no randomness;
* **no prose is parsed** - ``pricing_basis``, ``scope_included``, ``important_exclusions``,
  ``notes`` and ``price_status`` are display-only canonical text;
* **no monetary arithmetic**: every amount is copied verbatim (no rounding, no currency
  conversion, no FX) and **no totals exist** (``total_available`` is always ``False``);
* no LLM, no provider and no risk/applicability input, so a risk level can never change a cost.

Cost v1 is a **category-level cost reference** assessment. It is not a landed-cost, customs-duty,
tariff, tax, platform-fee, shipping or rule-level applicability calculation. The approved data
contains no ``rule_id``, ``attribute_id``, formula, quantity, unit, rate, cost type or
alternative-group linkage, so none is invented and none is inferred from prose.

Integration with ``AnalysisService`` is deferred to a later phase; this module does not modify or
import any other application service.

Design authority: ``Cost_Engine_PreImplementation_Inspection_Report.md`` (approved). The repository
cost interface (``get_cost`` / ``get_costs_for_category``) and the ``ComplianceCost`` /
``CalculatorUse`` models already exist, so no other production module changes.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


class CostCalculationStatus(str, Enum):
    """Verbatim ``CalculatorUse`` vocabulary. ``DIRECT`` is an exposable reference amount - never a
    claim that the amount is mandatory, payable or applicable to a specific product."""

    DIRECT = "DIRECT"                  # approved exact reference amount, directly exposable
    PLANNING_ONLY = "PLANNING_ONLY"    # estimate/reference range for planning, never a quote
    QUOTE_REQUIRED = "QUOTE_REQUIRED"  # no approved numeric amount exists at all
    DISPLAY_ONLY = "DISPLAY_ONLY"      # informational value, non-calculating


# The closed canonical product-category vocabulary (``A-CMN-001.allowed_values``) mapped to the
# repository cost categories Cost v1 may read. ``uncertain``/``unsupported`` read nothing: a
# terminal clarification is never permission to retrieve records.
_PRODUCT_CATEGORY_TARGETS: dict[str, tuple[str, ...]] = {
    "childrens_toys": ("childrens_toys",),
    "small_consumer_electronics": ("small_consumer_electronics",),
    "dual": ("childrens_toys", "small_consumer_electronics"),
    "uncertain": (),
    "unsupported": (),
}

# Closed vocabulary for the unassessed helper. Arbitrary caller prose is never exposed as a note.
_UNASSESSED_NOTE: dict[str, str] = {
    "unresolved": "Cost cannot be assessed because the product category is unresolved.",
    "unsupported": "Verified cost references are not available for this unsupported category.",
    "unconfirmed": "Cost was not assessed because the category has not been human-confirmed.",
    "no_data": "No approved cost reference records exist for the confirmed category.",
}
_UNASSESSED_TOTAL_REASON: dict[str, str] = {
    "unresolved": "No cost total is available because the product category is unresolved.",
    "unsupported": "No cost total is available because this category is not supported.",
    "unconfirmed": "No cost total is available because the category has not been human-confirmed.",
    "no_data": (
        "No cost total is available because no approved cost reference records exist for this "
        "confirmed category."
    ),
}

# Fixed assessment-level notes. Every one is a constant template; none is derived from record prose.
_NOTE_CATEGORY_REFERENCE = (
    "Cost items are category-level reference records; they are not rule-level applicability or "
    "payable-cost determinations."
)
_NOTE_PLANNING_ONLY = (
    "{count} planning-only record(s) are provider or regulatory estimates for planning, not quotes."
)
_NOTE_QUOTE_REQUIRED = "{count} quote-required record(s) have no approved numeric amount."
_NOTE_DISPLAY_ONLY = "{count} display-only record(s) are informational and non-calculating."

# The single fixed reason every Cost v1 assessment carries: additivity is not a structured property
# of the approved data, so no total (grand, subtotal, min/max or per-currency) may be produced.
_TOTAL_UNAVAILABLE_WITH_ITEMS = (
    "No cost total is calculated: approved cost references cannot be safely aggregated because "
    "they may contain mixed currencies, different pricing bases, alternative or add-on records, "
    "and non-calculating statuses."
)

_TITLE_TEMPLATE = "Cost reference: {cost_id}"


class CostConfigurationError(ValueError):
    """The canonical cost data is self-contradictory (a data/wiring fault, never a cost verdict)."""


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class CostItem(BaseModel):
    """One canonical cost reference record. All monetary values are verbatim pass-through."""

    cost_id: str
    category: str
    cost_item: str
    calculation_status: CostCalculationStatus

    exact_amount: float | None = None
    low_amount: float | None = None
    high_amount: float | None = None

    currency: str

    pricing_basis: str
    scope_included: str
    important_exclusions: str

    price_status: str
    evidence_status: str
    confidence: str | None = None

    source_ids: list[str] = Field(default_factory=list)
    source_date: str
    checked_date: str

    notes: str
    title: str = ""


class CostAssessment(BaseModel):
    """Category-level cost reference assessment. There is deliberately **no** total field, and
    ``total_available`` is ``False`` for every Cost v1 assessment."""

    assessed: bool
    category: str | None = None
    items: list[CostItem] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    currencies: list[str] = Field(default_factory=list)
    total_available: bool = False
    total_unavailable_reason: str | None = None
    missing_inputs: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Deterministic helpers
# --------------------------------------------------------------------------- #


def _text(value: Any) -> str | None:
    """Canonical value as a plain string (enum values included); ``None`` stays ``None``."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _source_ids(cost: Any) -> list[str]:
    """``[source_id, *additional_source_ids]`` de-duplicated in first-seen order.

    ``_validate_cost`` has already rejected a blank primary or additional source, so no source is
    ever silently dropped here.
    """
    ordered: list[str] = []
    for raw in (_text(getattr(cost, "source_id", None)),):
        if raw:
            ordered.append(raw)
    for raw in getattr(cost, "additional_source_ids", None) or ():
        text = _text(raw)
        if text:
            ordered.append(text)
    seen: set[str] = set()
    deduped: list[str] = []
    for source_id in ordered:
        if source_id in seen:
            continue
        seen.add(source_id)
        deduped.append(source_id)
    return deduped


def _amounts(cost: Any) -> tuple[float | None, float | None, float | None]:
    return (
        getattr(cost, "exact_amount", None),
        getattr(cost, "low_amount", None),
        getattr(cost, "high_amount", None),
    )


def _validate_cost(cost: Any) -> str:
    """Fail closed on any canonical contradiction. Returns the validated ``cost_id``.

    Every condition here is a data/wiring fault; normal uncertainty states (planning-only,
    display-only, quote-required with no amount, an old source date, an expired promotional price
    status, a missing confidence label, a non-USD currency) are canonical and never errors.
    """
    cost_id = _text(getattr(cost, "cost_id", None))
    if cost_id is None or not cost_id.strip():
        raise CostConfigurationError("canonical cost record with a blank cost_id")

    currency = _text(getattr(cost, "currency", None))
    if currency is None or not currency.strip():
        raise CostConfigurationError(f"cost {cost_id!r}: blank currency")

    # Provenance must survive: a canonical cost reference without its primary source is a data
    # fault, never a silently source-less item. Reference *existence* stays the repository's job.
    source_id = _text(getattr(cost, "source_id", None))
    if source_id is None or not source_id.strip():
        raise CostConfigurationError(f"cost {cost_id!r}: blank primary source_id")
    for raw in getattr(cost, "additional_source_ids", None) or ():
        additional = _text(raw)
        if additional is None or not additional.strip():
            raise CostConfigurationError(f"cost {cost_id!r}: blank additional_source_id")

    status = _text(getattr(cost, "calculator_use", None))
    if status not in {member.value for member in CostCalculationStatus}:
        raise CostConfigurationError(f"cost {cost_id!r}: unsupported calculator_use {status!r}")

    exact, low, high = _amounts(cost)
    evidence_status = _text(getattr(cost, "evidence_status", None))

    # Finiteness comes first: NaN and +/-Infinity are not canonical money. NaN in particular
    # defeats every ordinary comparison below and serializes to a non-standard ``NaN`` token.
    for field_name, value in (("exact_amount", exact), ("low_amount", low), ("high_amount", high)):
        if value is not None and not math.isfinite(value):
            raise CostConfigurationError(
                f"cost {cost_id!r}: {field_name} is not a finite number ({value!r})"
            )

    if exact is not None and (low is not None or high is not None):
        raise CostConfigurationError(
            f"cost {cost_id!r}: exact_amount and a range bound are both present"
        )
    if (low is None) != (high is None):
        raise CostConfigurationError(
            f"cost {cost_id!r}: only one range bound is present (low/high must be paired)"
        )
    if low is not None and high is not None and low > high:
        raise CostConfigurationError(
            f"cost {cost_id!r}: low_amount {low!r} exceeds high_amount {high!r}"
        )
    for field_name, value in (("exact_amount", exact), ("low_amount", low), ("high_amount", high)):
        if value is not None and value < 0:
            raise CostConfigurationError(f"cost {cost_id!r}: {field_name} is negative ({value!r})")

    if status == CostCalculationStatus.DIRECT.value:
        if exact is None:
            raise CostConfigurationError(
                f"cost {cost_id!r}: DIRECT record without an exact_amount"
            )
        if low is not None or high is not None:
            raise CostConfigurationError(
                f"cost {cost_id!r}: DIRECT record must not carry range bounds"
            )
        if evidence_status != "VERIFIED":
            raise CostConfigurationError(
                f"cost {cost_id!r}: DIRECT record with evidence_status {evidence_status!r}"
            )

    if status == CostCalculationStatus.QUOTE_REQUIRED.value and (
        exact is not None or low is not None or high is not None
    ):
        raise CostConfigurationError(
            f"cost {cost_id!r}: QUOTE_REQUIRED record must not carry a numeric amount"
        )

    if evidence_status == "NOT_FOUND" and (
        exact is not None or low is not None or high is not None
    ):
        raise CostConfigurationError(
            f"cost {cost_id!r}: evidence_status NOT_FOUND must not carry a numeric amount"
        )

    return cost_id


def _cost_item(cost: Any) -> CostItem:
    """Internal builder: one validated :class:`CostItem` from a canonical ``ComplianceCost``.

    Private by design - it bypasses the category gates, so it must never be used to construct items
    for records the public :meth:`CostEngine.assess` path excludes. All monetary values are copied
    verbatim.
    """
    cost_id = _validate_cost(cost)
    exact, low, high = _amounts(cost)
    return CostItem(
        cost_id=cost_id,
        category=_text(getattr(cost, "category", None)) or "",
        cost_item=_text(getattr(cost, "cost_item", None)) or "",
        calculation_status=CostCalculationStatus(_text(getattr(cost, "calculator_use", None))),
        exact_amount=exact,
        low_amount=low,
        high_amount=high,
        currency=_text(getattr(cost, "currency", None)) or "",
        pricing_basis=_text(getattr(cost, "pricing_basis", None)) or "",
        scope_included=_text(getattr(cost, "scope_included", None)) or "",
        important_exclusions=_text(getattr(cost, "important_exclusions", None)) or "",
        price_status=_text(getattr(cost, "price_status", None)) or "",
        evidence_status=_text(getattr(cost, "evidence_status", None)) or "",
        confidence=_text(getattr(cost, "confidence", None)),
        source_ids=_source_ids(cost),
        source_date=_text(getattr(cost, "source_date", None)) or "",
        checked_date=_text(getattr(cost, "checked_date", None)) or "",
        notes=_text(getattr(cost, "notes", None)) or "",
        title=_TITLE_TEMPLATE.format(cost_id=cost_id),
    )


def _counts(items: Sequence[CostItem]) -> dict[str, int]:
    """Per-status counts; all four keys are always present, and the sum equals ``len(items)``."""
    counts = {status.value: 0 for status in CostCalculationStatus}
    for item in items:
        counts[item.calculation_status.value] += 1
    return counts


def _currencies(items: Sequence[CostItem]) -> list[str]:
    return sorted({item.currency for item in items})


def _notes(items: Sequence[CostItem]) -> list[str]:
    """Fixed, uncertainty-preserving assessment notes (never a payable statement)."""
    notes = [_NOTE_CATEGORY_REFERENCE]
    planning = sum(1 for i in items if i.calculation_status is CostCalculationStatus.PLANNING_ONLY)
    if planning:
        notes.append(_NOTE_PLANNING_ONLY.format(count=planning))
    quote = sum(1 for i in items if i.calculation_status is CostCalculationStatus.QUOTE_REQUIRED)
    if quote:
        notes.append(_NOTE_QUOTE_REQUIRED.format(count=quote))
    display = sum(1 for i in items if i.calculation_status is CostCalculationStatus.DISPLAY_ONLY)
    if display:
        notes.append(_NOTE_DISPLAY_ONLY.format(count=display))
    return notes


# --------------------------------------------------------------------------- #
# Unassessed assessments (no category-level cost assessment happened)
# --------------------------------------------------------------------------- #


def unassessed_cost_assessment(reason: str, *, category: str | None = None) -> CostAssessment:
    """``assessed=False`` with zero items and no total for a category that could not be assessed.

    ``reason`` is one of the closed internal vocabulary members ``"unresolved"``,
    ``"unsupported"``, ``"unconfirmed"`` or ``"no_data"``. Missing cost data is never rendered as
    zero, "free" or "no cost".
    """
    if reason not in _UNASSESSED_NOTE:
        raise ValueError(f"unknown unassessed cost reason: {reason!r}")
    return CostAssessment(
        assessed=False,
        category=category,
        items=[],
        counts=_counts(()),
        currencies=[],
        total_available=False,
        total_unavailable_reason=_UNASSESSED_TOTAL_REASON[reason],
        missing_inputs=[],
        notes=[_UNASSESSED_NOTE[reason]],
    )


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class CostEngine:
    """Deterministic category-level cost reference assessment over the approved cost data."""

    def __init__(self, repository: Any) -> None:
        self._repository = repository

    # -- public API -------------------------------------------------------- #

    def assess(self, category: str | None) -> CostAssessment:
        """Assess the approved cost references for one canonical product category.

        ``None``/blank and ``uncertain`` are unresolved; ``unsupported`` never reads data at all
        (the raw ``category == "unsupported"`` record is not permission to cost an unsupported
        product). ``dual`` reads the union of the two product categories. Any other string - for
        example ``"amazon_us_platform"`` - is not a canonical product category and fails closed,
        so arbitrary caller text can never retrieve an arbitrary repository cost category.

        There are deliberately **no** risk, applicability, prose or model parameters, so neither
        model output nor a risk level can influence a cost assessment.

        Raises :class:`CostConfigurationError` on self-contradictory canonical data.
        """
        name = _text(category)
        if name is None or not name.strip():
            return unassessed_cost_assessment("unresolved")
        name = name.strip()
        if name not in _PRODUCT_CATEGORY_TARGETS:
            raise CostConfigurationError(
                f"category {name!r} is not a canonical product category; Cost v1 never retrieves "
                "arbitrary repository cost categories"
            )

        targets = _PRODUCT_CATEGORY_TARGETS[name]
        if not targets:
            reason = "unsupported" if name == "unsupported" else "unresolved"
            return unassessed_cost_assessment(reason, category=name)

        records: list[Any] = []
        seen: set[str] = set()
        for target in targets:
            fetched = list(self._repository.get_costs_for_category(target))
            _require_unique_cost_ids(fetched, target)
            for record in fetched:
                cost_id = _require_matching_category(record, target)
                if cost_id in seen:
                    # Union semantics for ``dual``: the same canonical record read through both
                    # category paths is included once, in first-seen order.
                    continue
                seen.add(cost_id)
                records.append(record)

        if not records:
            return unassessed_cost_assessment("no_data", category=name)

        items = [_cost_item(record) for record in records]
        items.sort(key=lambda item: item.cost_id)
        return CostAssessment(
            assessed=True,
            category=name,
            items=items,
            counts=_counts(items),
            currencies=_currencies(items),
            total_available=False,
            total_unavailable_reason=_TOTAL_UNAVAILABLE_WITH_ITEMS,
            missing_inputs=[],
            notes=_notes(items),
        )


def _require_unique_cost_ids(records: Sequence[Any], target: str) -> None:
    """Fail closed on a blank or duplicated ``cost_id`` inside one category fetch."""
    seen: set[str] = set()
    for record in records:
        cost_id = _text(getattr(record, "cost_id", None))
        if cost_id is None or not cost_id.strip():
            raise CostConfigurationError(f"canonical cost record with a blank cost_id in {target!r}")
        if cost_id in seen:
            raise CostConfigurationError(f"duplicate canonical cost identifier: {cost_id!r}")
        seen.add(cost_id)


def _require_matching_category(record: Any, target: str) -> str:
    """Fail closed if the repository returned a record outside the queried category.

    This is defense in depth: a ``category == "unsupported"`` or ``"amazon_us_platform"`` record
    can never be surfaced through a product-category assessment, even if a repository
    misbehaves. It is a wiring fault, never a cost verdict.
    """
    record_category = _text(getattr(record, "category", None))
    if record_category != target:
        raise CostConfigurationError(
            f"cost record category {record_category!r} does not match queried category {target!r}"
        )
    return _text(getattr(record, "cost_id", None)) or ""
