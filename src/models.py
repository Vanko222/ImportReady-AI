"""Pydantic domain models for the approved Data Schema v1 JSON knowledge base.

Models mirror the actual record structures present in ``data/*.json`` rather
than an idealized schema. Optional fields are optional because the approved
files legitimately contain nulls for those fields.

File layout note: every approved JSON file shares the same top-level envelope
(``schema_version``, ``generated_at``, ``source_files``, ``records``); only the
record types differ per file.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class EvidenceStatus(str, Enum):
    """Controlled evidence_status vocabulary shared by the approved files."""

    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    CONFLICT = "CONFLICT"
    NOT_FOUND = "NOT_FOUND"


class RuleStatus(str, Enum):
    """Controlled rule lifecycle status vocabulary."""

    EFFECTIVE = "EFFECTIVE"
    PROPOSED = "PROPOSED"
    SUPERSEDED = "SUPERSEDED"
    WATCHLIST = "WATCHLIST"
    UNKNOWN = "UNKNOWN"


class CalculatorUse(str, Enum):
    """Cost calculator_use semantics preserved from the approved data."""

    DIRECT = "DIRECT"
    PLANNING_ONLY = "PLANNING_ONLY"
    QUOTE_REQUIRED = "QUOTE_REQUIRED"
    DISPLAY_ONLY = "DISPLAY_ONLY"


class _BaseRecord(BaseModel):
    """Strict record base: unknown keys are treated as structural drift."""

    model_config = ConfigDict(extra="forbid")


class SourceFileRef(_BaseRecord):
    """One entry of the top-level source_files provenance list."""

    file_name: str
    sha256: str
    role: str


class _EnvelopeBase(BaseModel):
    """Shared top-level envelope keys required by Data Schema v1."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    generated_at: str
    source_files: list[SourceFileRef]


# --------------------------------------------------------------------------- #
# product_taxonomy.json
# --------------------------------------------------------------------------- #
class ProductAttribute(_BaseRecord):
    attribute_id: str
    category: str
    group: str
    attribute_name: str
    data_type: str
    allowed_values: list[str]
    why_it_matters: str
    triggered_rule_ids: list[str]
    mvp_priority: str
    evidence_status: EvidenceStatus


class ProductTaxonomyFile(_EnvelopeBase):
    records: list[ProductAttribute]


# --------------------------------------------------------------------------- #
# compliance_rules.json
# --------------------------------------------------------------------------- #
class ComplianceRule(_BaseRecord):
    rule_id: str
    category: str
    requirement: str
    requirement_type: str
    jurisdiction: str
    authority: str
    trigger_conditions: str
    required_attribute_ids: list[str]
    required_tests: str
    required_documents: str
    seller_importer_actions: str
    labeling_manual_requirements: str
    clarification_question: str | None = None
    missing_information_blocks_decision: bool | None = None
    runtime_status_if_missing: str | None = None
    risk_if_missing: str
    evidence_status: EvidenceStatus
    rule_status: RuleStatus
    effective_update_date: str
    source_ids: list[str]
    mvp_priority: str
    applicability_notes: str


class ComplianceRulesFile(_EnvelopeBase):
    records: list[ComplianceRule]


# --------------------------------------------------------------------------- #
# policy_sources.json
# --------------------------------------------------------------------------- #
class PolicySource(_BaseRecord):
    source_id: str
    authority: str
    page_document_title: str
    source_type: str
    source_tier: str
    jurisdiction: str | None = None
    canonical_url: str
    topics: list[str]
    evidence_status: EvidenceStatus
    check_method: str | None = None
    last_checked: str
    last_known_update: str
    access_notes: str | None = None
    login_required: str | None = None
    future_update_check_path: str
    update_check_status: str
    short_supporting_evidence: str
    last_content_hash: str | None = None


class PolicySourcesFile(_EnvelopeBase):
    records: list[PolicySource]


# --------------------------------------------------------------------------- #
# compliance_costs.json
# --------------------------------------------------------------------------- #
class ComplianceCost(_BaseRecord):
    cost_id: str
    category: str
    cost_item: str
    exact_amount: float | None = None
    low_amount: float | None = None
    high_amount: float | None = None
    currency: str
    pricing_basis: str
    scope_included: str
    important_exclusions: str
    source_id: str
    additional_source_ids: list[str] = Field(default_factory=list)
    source_date: str
    checked_date: str
    price_status: str
    evidence_status: EvidenceStatus
    confidence: str | None = None
    calculator_use: CalculatorUse
    notes: str


class ComplianceCostsFile(_EnvelopeBase):
    records: list[ComplianceCost]


# --------------------------------------------------------------------------- #
# known_gaps.json
# --------------------------------------------------------------------------- #
class KnownGap(_BaseRecord):
    gap_id: str
    area: str
    related_rule_ids: list[str]
    related_source_ids: list[str]
    evidence_status: EvidenceStatus
    rule_status: RuleStatus | None = None
    issue: str
    attempt_result: str
    mvp_handling: str


class KnownGapsFile(_EnvelopeBase):
    records: list[KnownGap]


# --------------------------------------------------------------------------- #
# pending_policy_updates.json
# --------------------------------------------------------------------------- #
class PendingPolicyUpdate(_BaseRecord):
    """Placeholder for pending policy-update records.

    The approved file currently contains zero records, so no record schema is
    observable yet. The model stays permissive on purpose: it only requires a
    JSON object and never drops unknown keys. Tighten once a real record
    schema is approved.
    """

    model_config = ConfigDict(extra="allow")


class PendingPolicyUpdatesFile(_EnvelopeBase):
    records: list[PendingPolicyUpdate]
