"""AnalysisService for the P0-2 agent layer.

Responsibility: repository query, evidence aggregation, canonical result
assembly, review trigger generation.

It does NOT classify products, approve compliance, calculate risk, or
calculate cost. Its output always separates three kinds of content:
``verified`` (from the repository), ``agent_suggestions`` (candidate content
requiring human review), and ``unknown`` (missing / not-evaluated).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.models import ComplianceRule, EvidenceStatus, PolicySource, RuleStatus
from src.repositories.base import ComplianceRepository
from src.services.classification import CategoryResult, CategorySource, CategoryStatus


class ComplianceFinding(BaseModel):
    rule_id: str
    category: str
    requirement: str
    requirement_type: str
    authority: str
    jurisdiction: str
    evidence_status: str
    rule_status: str
    required_attribute_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
    mvp_priority: str | None = None


class EvidenceReference(BaseModel):
    source_id: str
    authority: str
    title: str
    url: str
    source_type: str | None = None
    source_tier: str | None = None
    evidence_status: str
    last_checked: str | None = None
    update_check_status: str | None = None


class MissingInformation(BaseModel):
    attribute_id: str
    attribute_name: str | None = None


class KnownLimitation(BaseModel):
    gap_id: str
    area: str
    issue: str
    mvp_handling: str


class VerifiedFacts(BaseModel):
    compliance_information: list[ComplianceFinding] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    known_limitations: list[KnownLimitation] = Field(default_factory=list)


class UnknownInfo(BaseModel):
    missing_information: list[MissingInformation] = Field(default_factory=list)
    not_evaluated: dict[str, str] = Field(
        default_factory=lambda: {
            "applicability": "NOT_EVALUATED",
            "risk": "NOT_EVALUATED",
            "cost": "NOT_AVAILABLE",
        }
    )


class Review(BaseModel):
    status: str
    triggers: list[str] = Field(default_factory=list)
    reviewer_actions: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Canonical analysis result with strict verified/suggestion/unknown split."""

    knowledge: dict[str, str]
    classification: dict[str, Any]
    verified: VerifiedFacts = Field(default_factory=VerifiedFacts)
    agent_suggestions: list[dict[str, Any]] = Field(default_factory=list)
    unknown: UnknownInfo = Field(default_factory=UnknownInfo)
    review: Review

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class AnalysisService:
    """Aggregates verified compliance knowledge for a resolved category."""

    def __init__(self, repository: ComplianceRepository) -> None:
        self._repository = repository

    def analyze(
        self,
        category_result: CategoryResult,
        provided_attribute_ids: list[str] | None = None,
    ) -> AnalysisResult:
        provided = set(provided_attribute_ids or [])
        knowledge = self._knowledge()
        classification = category_result.model_dump(mode="json")
        status = category_result.category_status

        if status == CategoryStatus.NEEDS_INFO:
            return AnalysisResult(
                knowledge=knowledge,
                classification=classification,
                review=Review(
                    status="NEEDS_INFO",
                    triggers=["category_unresolved"],
                    reviewer_actions=["provide --category <category>"],
                ),
            )
        if status == CategoryStatus.UNSUPPORTED:
            return AnalysisResult(
                knowledge=knowledge,
                classification=classification,
                review=Review(
                    status="UNSUPPORTED",
                    triggers=["unsupported_category"],
                    reviewer_actions=["stop: no verified compliance data for this category"],
                ),
            )

        rules = self._rules_for(category_result.category)
        missing = self._missing(rules, provided)
        triggers, actions = self._review(category_result, rules, missing)
        review_status = self._review_status(category_result, missing)

        return AnalysisResult(
            knowledge=knowledge,
            classification=classification,
            verified=VerifiedFacts(
                compliance_information=[self._finding(rule) for rule in rules],
                evidence=self._evidence(rules),
                known_limitations=self._limitations(rules),
            ),
            agent_suggestions=self._suggestions(category_result),
            unknown=UnknownInfo(
                missing_information=[
                    MissingInformation(
                        attribute_id=attribute_id,
                        attribute_name=self._attribute_name(attribute_id),
                    )
                    for attribute_id in missing
                ]
            ),
            review=Review(status=review_status, triggers=triggers, reviewer_actions=actions),
        )

    def evidence_for(self, rule_id: str) -> dict[str, Any]:
        """Aggregate evidence sources for a single canonical rule ID."""
        rule = self._repository.get_rule(rule_id)
        if rule is None:
            return {
                "ok": False,
                "error": {
                    "type": "unresolved",
                    "message": f"rule_id {rule_id!r} not found in verified data",
                },
            }
        sources: list[dict[str, Any]] = []
        for source_id in rule.source_ids:
            source = self._repository.get_source(source_id)
            if source is not None:
                sources.append(self._evidence_ref(source).model_dump(mode="json"))
        return {
            "ok": True,
            "rule_id": rule_id,
            "requirement": rule.requirement,
            "evidence_status": rule.evidence_status.value,
            "rule_status": rule.rule_status.value,
            "sources": sources,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _knowledge(self) -> dict[str, str]:
        snapshot = self._repository.knowledge_snapshot()
        return {
            "schema_version": snapshot.schema_version,
            "generated_at": snapshot.generated_at,
        }

    def _rules_for(self, category: str | None) -> list[ComplianceRule]:
        if not category:
            return []
        # Retrieval only: common rules apply to every supported category, and
        # "dual" means both relevant category paths. Final list is de-duplicated
        # by canonical rule_id in a deterministic order (common first).
        if category == "dual":
            targets = ["common", "childrens_toys", "small_consumer_electronics"]
        else:
            targets = ["common", category]
        rules: list[ComplianceRule] = []
        seen: set[str] = set()
        for target in targets:
            for rule in self._repository.get_rules_for_category(target):
                if rule.rule_id not in seen:
                    seen.add(rule.rule_id)
                    rules.append(rule)
        return rules

    def _missing(self, rules: list[ComplianceRule], provided: set[str]) -> list[str]:
        required: list[str] = []
        for rule in rules:
            for attribute_id in rule.required_attribute_ids:
                if attribute_id not in required:
                    required.append(attribute_id)
        return [attribute_id for attribute_id in required if attribute_id not in provided]

    def _finding(self, rule: ComplianceRule) -> ComplianceFinding:
        return ComplianceFinding(
            rule_id=rule.rule_id,
            category=rule.category,
            requirement=rule.requirement,
            requirement_type=rule.requirement_type,
            authority=rule.authority,
            jurisdiction=rule.jurisdiction,
            evidence_status=rule.evidence_status.value,
            rule_status=rule.rule_status.value,
            required_attribute_ids=list(rule.required_attribute_ids),
            source_ids=list(rule.source_ids),
            clarification_question=rule.clarification_question,
            mvp_priority=rule.mvp_priority,
        )

    def _evidence(self, rules: list[ComplianceRule]) -> list[EvidenceReference]:
        seen: set[str] = set()
        refs: list[EvidenceReference] = []
        for rule in rules:
            for source_id in rule.source_ids:
                if source_id in seen:
                    continue
                seen.add(source_id)
                source = self._repository.get_source(source_id)
                if source is not None:
                    refs.append(self._evidence_ref(source))
        return refs

    def _evidence_ref(self, source: PolicySource) -> EvidenceReference:
        return EvidenceReference(
            source_id=source.source_id,
            authority=source.authority,
            title=source.page_document_title,
            url=source.canonical_url,
            source_type=source.source_type,
            source_tier=source.source_tier,
            evidence_status=source.evidence_status.value,
            last_checked=source.last_checked,
            update_check_status=source.update_check_status,
        )

    def _limitations(self, rules: list[ComplianceRule]) -> list[KnownLimitation]:
        rule_ids = {rule.rule_id for rule in rules}
        limitations: list[KnownLimitation] = []
        for gap in self._repository.get_known_gaps():
            if set(gap.related_rule_ids) & rule_ids:
                limitations.append(
                    KnownLimitation(
                        gap_id=gap.gap_id,
                        area=gap.area,
                        issue=gap.issue,
                        mvp_handling=gap.mvp_handling,
                    )
                )
        return limitations

    def _review(
        self,
        category_result: CategoryResult,
        rules: list[ComplianceRule],
        missing: list[str],
    ) -> tuple[list[str], list[str]]:
        triggers: list[str] = []
        if category_result.category_status == CategoryStatus.REVIEW_REQUIRED:
            # Explicit classification uncertainty must not hide behind the
            # generic engines_not_implemented trigger.
            triggers.append("category_review_required")
        if missing:
            triggers.append("missing_information")
        if any(rule.evidence_status != EvidenceStatus.VERIFIED for rule in rules):
            triggers.append("insufficient_evidence")
        if any(rule.rule_status != RuleStatus.EFFECTIVE for rule in rules):
            triggers.append("rule_not_effective")
        # Applicability / risk / cost engines are not implemented yet; a verified
        # analysis can therefore never be auto-approved as compliant.
        triggers.append("engines_not_implemented")

        actions: list[str] = []
        if category_result.category_status == CategoryStatus.REVIEW_REQUIRED:
            actions.append("confirm_or_correct_category")
        if missing:
            actions.append("provide_missing_attributes")
        actions.append("human_review_required")
        return triggers, actions

    def _review_status(self, category_result: CategoryResult, missing: list[str]) -> str:
        if category_result.category_status == CategoryStatus.REVIEW_REQUIRED:
            return "REVIEW_REQUIRED"
        if missing:
            return "NEEDS_INFO"
        # A resolved category with no missing info still cannot be approved
        # without the not-yet-implemented engines.
        return "REVIEW_REQUIRED"

    def _suggestions(self, category_result: CategoryResult) -> list[dict[str, Any]]:
        if category_result.category_source == CategorySource.AGENT_GENERATED:
            return [
                {
                    "kind": "category_suggestion",
                    "category": category_result.category,
                    "classifier_confidence": category_result.classifier_confidence,
                    "note": "agent-generated category; requires human confirmation",
                }
            ]
        return []

    def _attribute_name(self, attribute_id: str) -> str | None:
        attribute = self._repository.get_attribute(attribute_id)
        return attribute.attribute_name if attribute is not None else None
