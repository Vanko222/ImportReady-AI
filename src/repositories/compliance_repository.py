"""JsonComplianceRepository — trusted-data access layer for the six approved
Data Schema v1 files.

The repository:
- resolves the data directory (configurable, defaults to <repo>/data);
- loads every approved JSON file once at construction;
- validates top-level envelope + per-record Pydantic models;
- indexes canonical records by ID;
- validates cross-file references at startup;
- exposes safe query methods.

All trusted-data problems surface as typed exceptions defined here; nothing is
silently repaired or fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from src import config
from src.models import (
    ComplianceCost,
    ComplianceCostsFile,
    ComplianceRule,
    ComplianceRulesFile,
    KnownGap,
    KnownGapsFile,
    PendingPolicyUpdate,
    PendingPolicyUpdatesFile,
    PolicySource,
    PolicySourcesFile,
    ProductAttribute,
    ProductTaxonomyFile,
)
from src.state import KnowledgeSnapshot


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class TrustedDataError(Exception):
    """Base class for all trusted-data/repository failures."""


class DataDirectoryError(TrustedDataError):
    """Data directory is missing or does not contain the approved files."""


class DatasetLoadError(TrustedDataError):
    """A JSON file could not be read or parsed."""


class DatasetValidationError(TrustedDataError):
    """A JSON file failed top-level or record-level structural validation."""


class DuplicateIDError(TrustedDataError):
    """A dataset contains duplicate canonical IDs."""


class ReferenceIntegrityError(TrustedDataError):
    """A cross-file reference does not resolve to a real record."""


# --------------------------------------------------------------------------- #
# Repository
# --------------------------------------------------------------------------- #
class JsonComplianceRepository:
    """Loads and serves the approved trusted compliance knowledge base."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir).resolve() if data_dir else config.get_data_dir()

        self._load_all()

        # ID indexes built from validated records.
        self._attributes: dict[str, ProductAttribute] = {}
        self._rules: dict[str, ComplianceRule] = {}
        self._sources: dict[str, PolicySource] = {}
        self._costs: dict[str, ComplianceCost] = {}
        self._gaps: dict[str, KnownGap] = {}
        self._pending_updates: list[PendingPolicyUpdate] = []

        # Top-level metadata (same across all files in the approved baseline).
        self._schema_version: str = config.EXPECTED_SCHEMA_VERSION
        self._generated_at: str = self._taxonomy.generated_at

        self._index_records()
        self._assert_ids_unique()
        self.validate_references()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load_all(self) -> None:
        if not self._data_dir.is_dir():
            raise DataDirectoryError(f"Trusted data directory not found: {self._data_dir}")

        loaders: dict[str, Callable[[dict[str, Any]], Any]] = {
            "product_taxonomy.json": self._load_product_taxonomy,
            "compliance_rules.json": self._load_compliance_rules,
            "policy_sources.json": self._load_policy_sources,
            "compliance_costs.json": self._load_compliance_costs,
            "known_gaps.json": self._load_known_gaps,
            "pending_policy_updates.json": self._load_pending_updates,
        }

        for file_name, loader in loaders.items():
            path = self._data_dir / file_name
            if not path.is_file():
                raise DatasetLoadError(f"Missing trusted data file: {path}")
            loader(self._read_json(path))

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            # Invalid UTF-8 is a trusted-data problem, not a raw decoding leak.
            raise DatasetLoadError(
                f"Invalid UTF-8 encoding in {path.name}: {exc}"
            ) from exc
        except OSError as exc:  # pragma: no cover - filesystem edge cases
            raise DatasetLoadError(f"Cannot read {path}: {exc}") from exc
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DatasetLoadError(f"Invalid JSON in {path.name}: {exc}") from exc
        if not isinstance(doc, dict):
            raise DatasetValidationError(
                f"{path.name}: top-level JSON must be an object, got {type(doc).__name__}"
            )
        return doc

    # Per-file typed parsing. A dedicated loader per type keeps the mapping
    # explicit and avoids generic reflection gymnastics.
    def _load_product_taxonomy(self, doc: dict[str, Any]) -> None:
        self._taxonomy = self._parse_file(doc, ProductTaxonomyFile, "product_taxonomy.json")

    def _load_compliance_rules(self, doc: dict[str, Any]) -> None:
        self._rules_file = self._parse_file(doc, ComplianceRulesFile, "compliance_rules.json")

    def _load_policy_sources(self, doc: dict[str, Any]) -> None:
        self._sources_file = self._parse_file(doc, PolicySourcesFile, "policy_sources.json")

    def _load_compliance_costs(self, doc: dict[str, Any]) -> None:
        self._costs_file = self._parse_file(doc, ComplianceCostsFile, "compliance_costs.json")

    def _load_known_gaps(self, doc: dict[str, Any]) -> None:
        self._gaps_file = self._parse_file(doc, KnownGapsFile, "known_gaps.json")

    def _load_pending_updates(self, doc: dict[str, Any]) -> None:
        self._pending_file = self._parse_file(doc, PendingPolicyUpdatesFile, "pending_policy_updates.json")

    def _parse_file(self, doc: dict[str, Any], file_model: type[BaseModel], file_name: str) -> BaseModel:
        """Validate top-level envelope and schema_version for one JSON file."""
        required = {"schema_version", "generated_at", "source_files", "records"}
        missing = sorted(required - doc.keys())
        if missing:
            raise DatasetValidationError(
                f"{file_name}: missing required top-level key(s): {', '.join(missing)}"
            )
        try:
            parsed = file_model.model_validate(doc)
        except ValidationError as exc:
            raise DatasetValidationError(
                f"{file_name}: record structure invalid: {exc}"
            ) from exc
        if parsed.schema_version != config.EXPECTED_SCHEMA_VERSION:
            raise DatasetValidationError(
                f"{file_name}: schema_version must be "
                f"{config.EXPECTED_SCHEMA_VERSION!r}, got {parsed.schema_version!r}"
            )
        self._generated_at = parsed.generated_at
        return parsed

    # ------------------------------------------------------------------ #
    # Indexing + reference validation
    # ------------------------------------------------------------------ #
    def _index_records(self) -> None:
        self._attributes = {r.attribute_id: r for r in self._taxonomy.records}
        self._rules = {r.rule_id: r for r in self._rules_file.records}
        self._sources = {r.source_id: r for r in self._sources_file.records}
        self._costs = {r.cost_id: r for r in self._costs_file.records}
        self._gaps = {r.gap_id: r for r in self._gaps_file.records}
        self._pending_updates = list(self._pending_file.records)

    def _assert_ids_unique(self) -> None:
        # Detect duplicates on the raw record lists before dict indexing could
        # silently collapse them.
        self._assert_unique_list("attribute_id", [r.attribute_id for r in self._taxonomy.records])
        self._assert_unique_list("rule_id", [r.rule_id for r in self._rules_file.records])
        self._assert_unique_list("source_id", [r.source_id for r in self._sources_file.records])
        self._assert_unique_list("cost_id", [r.cost_id for r in self._costs_file.records])
        self._assert_unique_list("gap_id", [r.gap_id for r in self._gaps_file.records])

    def _assert_unique_list(self, id_kind: str, ids: list[str]) -> None:
        seen: set[str] = set()
        for id_ in ids:
            if id_ in seen:
                raise DuplicateIDError(f"Duplicate {id_kind} in trusted data: {id_!r}")
            seen.add(id_)

    def validate_references(self) -> None:
        """Raise ReferenceIntegrityError if any cross-file reference is broken."""
        problems: list[str] = []

        # 1. rule.required_attribute_ids -> product attribute
        for rule in self._rules.values():
            for ref in rule.required_attribute_ids:
                if ref not in self._attributes:
                    problems.append(
                        f"rule {rule.rule_id}: required_attribute_id {ref!r} has no matching product attribute"
                    )

        # 2. rule.source_ids -> policy source
        for rule in self._rules.values():
            for ref in rule.source_ids:
                if ref not in self._sources:
                    problems.append(
                        f"rule {rule.rule_id}: source_id {ref!r} has no matching policy source"
                    )

        # 3. attribute.triggered_rule_ids -> compliance rule
        for attribute in self._attributes.values():
            for ref in attribute.triggered_rule_ids:
                if ref not in self._rules:
                    problems.append(
                        f"attribute {attribute.attribute_id}: triggered_rule_id {ref!r} has no matching rule"
                    )

        # 4. cost.source_id / additional_source_ids -> policy source
        for cost in self._costs.values():
            if cost.source_id not in self._sources:
                problems.append(
                    f"cost {cost.cost_id}: source_id {cost.source_id!r} has no matching policy source"
                )
            for ref in cost.additional_source_ids:
                if ref not in self._sources:
                    problems.append(
                        f"cost {cost.cost_id}: additional_source_id {ref!r} has no matching policy source"
                    )

        # 5. known-gap references -> rules / sources
        for gap in self._gaps.values():
            for ref in gap.related_rule_ids:
                if ref not in self._rules:
                    problems.append(
                        f"gap {gap.gap_id}: related_rule_id {ref!r} has no matching rule"
                    )
            for ref in gap.related_source_ids:
                if ref not in self._sources:
                    problems.append(
                        f"gap {gap.gap_id}: related_source_id {ref!r} has no matching policy source"
                    )

        # 6. pending update references are checked only when records exist.
        for idx, update in enumerate(self._pending_updates):
            extra = getattr(update, "__pydantic_extra__", None) or {}
            for ref in [extra.get("source_id"), extra.get("source_ids")]:
                for r in (ref if isinstance(ref, list) else [ref]):
                    if r is not None and r not in self._sources:
                        problems.append(
                            f"pending update record {idx}: source reference {r!r} has no matching policy source"
                        )

        if problems:
            preview = problems if len(problems) <= 10 else problems[:10] + [
                f"... and {len(problems) - 10} more"
            ]
            raise ReferenceIntegrityError(
                "Broken cross-file references in trusted data:\n- " + "\n- ".join(preview)
            )

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def generated_at(self) -> str:
        return self._generated_at

    @property
    def attributes(self) -> tuple[ProductAttribute, ...]:
        return tuple(self._attributes.values())

    @property
    def rules(self) -> tuple[ComplianceRule, ...]:
        return tuple(self._rules.values())

    @property
    def sources(self) -> tuple[PolicySource, ...]:
        return tuple(self._sources.values())

    @property
    def costs(self) -> tuple[ComplianceCost, ...]:
        return tuple(self._costs.values())

    @property
    def known_gaps(self) -> tuple[KnownGap, ...]:
        return tuple(self._gaps.values())

    @property
    def pending_policy_updates(self) -> tuple[PendingPolicyUpdate, ...]:
        return tuple(self._pending_updates)

    # Lookup semantics: a known canonical ID returns the record; an unknown ID
    # returns None. Never a fabricated fallback.
    def get_attribute(self, attribute_id: str) -> ProductAttribute | None:
        return self._attributes.get(attribute_id)

    def get_rule(self, rule_id: str) -> ComplianceRule | None:
        return self._rules.get(rule_id)

    def get_source(self, source_id: str) -> PolicySource | None:
        return self._sources.get(source_id)

    def get_cost(self, cost_id: str) -> ComplianceCost | None:
        return self._costs.get(cost_id)

    def get_rules_for_category(self, category: str) -> list[ComplianceRule]:
        return [r for r in self._rules.values() if r.category == category]

    def get_costs_for_category(self, category: str) -> list[ComplianceCost]:
        return [c for c in self._costs.values() if c.category == category]

    def get_known_gaps(self) -> list[KnownGap]:
        return list(self._gaps.values())

    def get_pending_policy_updates(self) -> list[PendingPolicyUpdate]:
        return list(self._pending_updates)

    def knowledge_snapshot(self) -> KnowledgeSnapshot:
        """Lightweight snapshot identifying the loaded trusted knowledge."""
        return KnowledgeSnapshot(
            schema_version=self._schema_version,
            generated_at=self._generated_at,
        )

    @property
    def counts(self) -> dict[str, int]:
        """Baseline regression counts keyed by dataset role."""
        return {
            "attributes": len(self._attributes),
            "rules": len(self._rules),
            "sources": len(self._sources),
            "costs": len(self._costs),
            "known_gaps": len(self._gaps),
            "pending_policy_updates": len(self._pending_updates),
        }
