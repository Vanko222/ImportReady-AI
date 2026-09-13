"""AnalysisService for the P0-2 agent layer.

Responsibility: repository query, evidence aggregation, canonical result
assembly, review trigger generation.

It does NOT classify products, approve compliance, or calculate cost; the
compliance risk assessment is delegated to the deterministic Risk Engine and
attached to the result. Its output always separates three kinds of content:
``verified`` (from the repository), ``agent_suggestions`` (candidate content
requiring human review), and ``unknown`` (missing / not-evaluated).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from src.models import ComplianceRule, EvidenceStatus, PolicySource, RuleStatus
from src.repositories.base import ComplianceRepository
from src.services.applicability import (
    ApplicabilityEngine,
    ApplicabilityReasonCode,
    ApplicabilityResult,
    ApplicabilityStatus,
)
from src.services.classification import CategoryResult, CategorySource, CategoryStatus
from src.services.cost import CostAssessment, CostEngine, unassessed_cost_assessment
from src.services.risk import RiskAssessment, RiskEngine, unassessed_assessment
from src.state import ProductFact

# Deciding reason codes that represent a canonical INPUT DEFECT rather than a
# knowledge gap. These outrank MISSING_REQUIRED_FACTS in the review-status
# decision.
_FACT_DEFECT_REASON_CODES: frozenset[ApplicabilityReasonCode] = frozenset(
    {
        ApplicabilityReasonCode.CONTRADICTORY_FACTS,
        ApplicabilityReasonCode.INVALID_FACT_VALUE,
    }
)

# ``unknown.not_evaluated`` names the components that were NOT evaluated. A component that actually
# ran is removed from the reported mapping rather than replaced by another sentinel, and every
# remaining entry is preserved. The template is a constant and is always copied before use.
_NOT_EVALUATED_DEFAULT: dict[str, str] = {
    "risk": "NOT_EVALUATED",
    "cost": "NOT_AVAILABLE",
}


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
    # Applicability, risk and cost are canonical and live in AnalysisResult; nothing else is
    # implemented. ``not_evaluated`` names the components that were NOT evaluated, so an entry is
    # dropped on a path where its deterministic engine actually ran.
    not_evaluated: dict[str, str] = Field(
        default_factory=lambda: dict(_NOT_EVALUATED_DEFAULT)
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
    # Canonical applicability, present only when the category is human-confirmed
    # (CategoryStatus.RESOLVED). ``None`` means "not evaluated because the
    # category is not yet confirmed" — it never means NOT_APPLICABLE or "no
    # requirements".
    applicability: ApplicabilityResult | None = None
    # Canonical deterministic risk assessment, produced only when applicability
    # was actually evaluated. ``assessed=False`` with ``level=None`` means the
    # product could not be assessed — it never means "no risk" (RiskLevel.NONE).
    risk: RiskAssessment | None = None
    # Canonical deterministic category-level cost reference assessment. ``assessed``
    # is True only for a human-confirmed supported category. ``total_available`` is
    # always False in Cost v1 and there is deliberately no total field.
    cost: CostAssessment | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class AnalysisService:
    """Aggregates verified compliance knowledge for a resolved category."""

    def __init__(self, repository: ComplianceRepository) -> None:
        self._repository = repository
        # Exactly one engine per service; never re-instantiated per rule.
        self._applicability_engine = ApplicabilityEngine(repository)
        # Read-only deterministic risk engine over the same verified repository.
        self._risk_engine = RiskEngine(repository)
        # Read-only deterministic category-level cost engine over the same repository.
        self._cost_engine = CostEngine(repository)

    def analyze(
        self,
        category_result: CategoryResult,
        product_facts: Sequence[ProductFact] | None = None,
    ) -> AnalysisResult:
        facts = list(product_facts or [])
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
                # Not assessed: the category is not confirmed, so no rules were
                # evaluated. ``not_evaluated`` keeps "risk": "NOT_EVALUATED".
                risk=unassessed_assessment("unresolved"),
                # Cost is not assessed either; the CostEngine is never called here.
                cost=unassessed_cost_assessment("unresolved"),
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
                # Not assessed: there is no verified data for this category.
                risk=unassessed_assessment("unsupported"),
                # Explicitly unsupported: the CostEngine is never asked for the raw
                # "unsupported" cost-data category, so COST-005 can never surface.
                cost=unassessed_cost_assessment(
                    "unsupported", category=category_result.category
                ),
            )

        rules = self._rules_for(category_result.category)

        # Definitive applicability runs ONLY for a canonically confirmed
        # (RESOLVED) category. An agent-generated REVIEW_REQUIRED category still
        # shows candidate compliance information, but never a canonical
        # applicability verdict and never rule-based missing information.
        applicability = (
            self._applicability_engine.evaluate([rule.rule_id for rule in rules], facts)
            if status == CategoryStatus.RESOLVED
            else None
        )
        missing = (
            list(applicability.missing_attribute_ids) if applicability is not None else []
        )

        # Deterministic risk assessment runs on exactly the path where canonical
        # applicability was actually evaluated, and consumes the same deduplicated
        # rule list. Every other path gets the approved unassessed contract
        # (assessed=False, level=None) and never RiskLevel.NONE.
        if applicability is not None:
            risk = self._risk_engine.assess(rules, applicability)
        else:
            risk = unassessed_assessment("unresolved")

        # Deterministic cost is category-level and depends only on the canonical classification
        # result — never on Agent prose, risk output or applicability verdicts. A human-confirmed
        # (RESOLVED) category is assessed by the CostEngine; an unconfirmed agent-suggested
        # category never exposes assessed cost references.
        if applicability is not None:
            cost = self._cost_engine.assess(category_result.category)
        else:
            cost = unassessed_cost_assessment("unconfirmed", category=category_result.category)

        # ``not_evaluated`` reports only what was NOT evaluated: a component that actually ran is
        # removed and every remaining entry is preserved. A fresh copy is always used, so the
        # module-level template is never mutated.
        not_evaluated = dict(_NOT_EVALUATED_DEFAULT)
        if risk.assessed:
            not_evaluated.pop("risk", None)
        if cost.assessed:
            not_evaluated.pop("cost", None)

        triggers, actions = self._review(category_result, rules, missing, applicability)
        review_status = self._review_status(category_result, applicability)

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
                ],
                not_evaluated=not_evaluated,
            ),
            review=Review(status=review_status, triggers=triggers, reviewer_actions=actions),
            applicability=applicability,
            risk=risk,
            cost=cost,
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
        applicability: ApplicabilityResult | None,
    ) -> tuple[list[str], list[str]]:
        triggers: list[str] = []
        if category_result.category_status == CategoryStatus.REVIEW_REQUIRED:
            # Explicit classification uncertainty must remain visible.
            triggers.append("category_review_required")
        if missing:
            triggers.append("missing_information")
        if any(rule.evidence_status != EvidenceStatus.VERIFIED for rule in rules):
            triggers.append("insufficient_evidence")
        if any(rule.rule_status != RuleStatus.EFFECTIVE for rule in rules):
            triggers.append("rule_not_effective")
        # Cost availability is deliberately NOT a compliance review trigger: it is a separate
        # category-level reference feature, so a PLANNING_ONLY or QUOTE_REQUIRED cost record can
        # never turn a compliance review into REVIEW_REQUIRED.
        # Only a canonical, evaluated applicability result may raise this. A
        # ``None`` applicability (unconfirmed category) is already represented
        # by category_review_required and must not be reported as this.
        if applicability is not None and any(
            result.applicability_status == ApplicabilityStatus.REVIEW_REQUIRED
            for result in applicability.rules
        ):
            triggers.append("applicability_review_required")

        actions: list[str] = []
        if category_result.category_status == CategoryStatus.REVIEW_REQUIRED:
            actions.append("confirm_or_correct_category")
        if missing:
            actions.append("provide_missing_attributes")
        actions.append("human_review_required")
        return triggers, actions

    def _review_status(
        self,
        category_result: CategoryResult,
        applicability: ApplicabilityResult | None,
    ) -> str:
        """Global review status, driven by canonical deciding reason codes.

        ``ApplicabilityResult.missing_attribute_ids`` alone is NOT a NEEDS_INFO
        signal: an attribute is unusable when it was never supplied, when the
        supplied canonical value is invalid, when USER facts contradict, or when
        only untrusted facts exist. Only the canonical deciding reason
        (``reason_codes[0]``) distinguishes a genuine information gap from an
        input defect, so this method consumes the engine result rather than
        re-deriving the cause from raw facts.
        """
        if category_result.category_status != CategoryStatus.RESOLVED:
            return "REVIEW_REQUIRED"

        if applicability is not None:
            deciding = [
                result.reason_codes[0]
                for result in applicability.rules
                if result.reason_codes
            ]
            # A canonical input defect takes precedence over a knowledge gap.
            if any(code in _FACT_DEFECT_REASON_CODES for code in deciding):
                return "REVIEW_REQUIRED"
            if any(
                code == ApplicabilityReasonCode.MISSING_REQUIRED_FACTS
                for code in deciding
            ):
                return "NEEDS_INFO"

        # A human-confirmed category without a canonical information gap is still reported as
        # REVIEW_REQUIRED: the MVP does not assert "no human review needed" for a resolved
        # category. Cost availability never participates in this decision.
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
