"""Repository abstraction for the trusted compliance knowledge base.

``ComplianceRepository`` is a lightweight structural interface
(``typing.Protocol``) representing the public read/query contract of the
repository layer, so services depend on the abstraction rather than on the
concrete JSON implementation.

Exact-category semantics: ``get_rules_for_category(category)`` returns only
records whose ``category`` equals the requested value. Combining ``common``
rules with classified-category rules is the responsibility of the future
compliance/service layer, not of this repository interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.models import (
    ComplianceCost,
    ComplianceRule,
    KnownGap,
    PendingPolicyUpdate,
    PolicySource,
    ProductAttribute,
)
from src.state import KnowledgeSnapshot


@runtime_checkable
class ComplianceRepository(Protocol):
    """Read/query contract satisfied by trusted-data repository implementations."""

    def get_attribute(self, attribute_id: str) -> ProductAttribute | None: ...

    def get_rule(self, rule_id: str) -> ComplianceRule | None: ...

    def get_source(self, source_id: str) -> PolicySource | None: ...

    def get_cost(self, cost_id: str) -> ComplianceCost | None: ...

    def get_rules_for_category(self, category: str) -> list[ComplianceRule]: ...

    def get_costs_for_category(self, category: str) -> list[ComplianceCost]: ...

    def get_known_gaps(self) -> list[KnownGap]: ...

    def get_pending_policy_updates(self) -> list[PendingPolicyUpdate]: ...

    def validate_references(self) -> None: ...

    def knowledge_snapshot(self) -> KnowledgeSnapshot: ...
