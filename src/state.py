"""Runtime analysis-case state foundation (P0-1).

Provides structured ``CaseState`` plus the small models it needs now:
``ProductFact``, ``KnowledgeSnapshot`` and the controlled value enums for
analysis status and fact origin. Later P0 layers (classification, rule
evaluation, risk, cost, report) will populate fields that already exist here
as safely-typed placeholders.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.models import _BaseRecord


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_case_id() -> str:
    """Stable, unique case identifier: ``case_<uuid4 hex>``."""
    return f"case_{uuid.uuid4().hex}"


class AnalysisStatus(str, Enum):
    """Workflow status of a product-analysis case.

    Deliberately separate from the future Low/Medium/High risk level.
    """

    IN_PROGRESS = "IN_PROGRESS"
    NEEDS_INFO = "NEEDS_INFO"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    COMPLETE = "COMPLETE"
    UNSUPPORTED = "UNSUPPORTED"


class FactOrigin(str, Enum):
    """Approved origins for a stored product fact."""

    USER = "USER"
    CLASSIFIER = "CLASSIFIER"
    DERIVED = "DERIVED"


class ProductFact(_BaseRecord):
    """A single structured product fact referenced by canonical attribute ID.

    No inference logic lives here in P0-1; this is only a structured container.
    """

    attribute_id: str
    value: Any
    origin: FactOrigin = FactOrigin.USER


class KnowledgeSnapshot(_BaseRecord):
    """Lightweight identity of the trusted knowledge used for an analysis.

    Carries only dataset metadata (schema_version + generated_at), never a copy
    of the knowledge base itself.
    """

    schema_version: str
    generated_at: str


class CaseState(BaseModel):
    """Structured state of one product-analysis case.

    All mutable collections use ``default_factory`` so every instance receives
    independent containers. ``model_dump(mode="json")`` gives a JSON-compatible
    representation; ``model_dump_json`` gives a JSON string.
    """

    case_id: str = Field(default_factory=_new_case_id)
    created_at: datetime = Field(default_factory=_utc_now)
    raw_product_input: str | None = None
    classification: dict[str, Any] | None = None
    product_facts: list[ProductFact] = Field(default_factory=list)
    missing_attribute_ids: list[str] = Field(default_factory=list)
    rule_results: list[dict[str, Any]] = Field(default_factory=list)
    clarification_history: list[dict[str, Any]] = Field(default_factory=list)
    risk_result: dict[str, Any] | None = None
    cost_result: dict[str, Any] | None = None
    analysis_status: AnalysisStatus = AnalysisStatus.IN_PROGRESS
    knowledge_snapshot: KnowledgeSnapshot | None = None

    def add_fact(self, fact: ProductFact) -> None:
        """Store one product fact and retire it from missing_attribute_ids."""
        self.product_facts.append(fact)
        if fact.attribute_id in self.missing_attribute_ids:
            self.missing_attribute_ids.remove(fact.attribute_id)

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-compatible dict (datetimes/ids serialized as plain values)."""
        return self.model_dump(mode="json")
