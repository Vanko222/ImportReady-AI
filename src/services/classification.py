"""Category ownership for the P0-2 agent layer.

The verified dataset provides no production-ready ``product description ->
category`` mapping, so the MVP does NOT implement keyword/rule/automatic
mapping. The default classifier is :class:`HumanClassifier` (CLI ``--category``).

An LLM may suggest a category through :class:`AgentClassifier`, which is always
normalized to ``agent_generated`` / ``REVIEW_REQUIRED`` and can never become
``RESOLVED``. Every agent-generated value is validated in code against the
verified category vocabulary.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Literal, Protocol

from pydantic import BaseModel

from src.repositories.base import ComplianceRepository


class CategorySource(str, Enum):
    """Where a category value came from (MVP values only)."""

    HUMAN_CONFIRMED = "human_confirmed"
    AGENT_GENERATED = "agent_generated"
    UNRESOLVED = "unresolved"
    # Reserved for a future verified Core classifier (NOT implemented):
    # CORE_CLASSIFIER = "core_classifier"


class CategoryStatus(str, Enum):
    """Workflow status of a classification result."""

    RESOLVED = "RESOLVED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    NEEDS_INFO = "NEEDS_INFO"
    UNSUPPORTED = "UNSUPPORTED"


class CategoryResult(BaseModel):
    """Single classification outcome (category + provenance + status).

    ``classifier_confidence`` is optional and only ever represents classifier /
    model output confidence. It MUST NOT represent compliance certainty and
    MUST NOT affect applicability, risk, or final compliance status.
    """

    category: str | None = None
    category_source: CategorySource
    category_status: CategoryStatus
    classifier_confidence: float | None = None


class CategorySuggestionPayload(BaseModel):
    """Explicit structured-output contract for an LLM category suggestion.

    ``decision`` disambiguates three cases that a single nullable ``category``
    cannot: a model that proposes a category, a model that intentionally
    declines for lack of information, and a model that produced inconsistent
    output. Pydantic validates only the structure here; the category VALUE is
    later validated against ``allowed_category_values(repository)``.
    """

    decision: Literal["SUGGEST_CATEGORY", "NEEDS_INFO"]
    category: str | None = None


class Classifier(Protocol):
    """Interface satisfied by category classifiers."""

    def classify(
        self, product_description: str, provided_category: str | None = None
    ) -> CategoryResult: ...


def allowed_category_values(repository: ComplianceRepository) -> list[str]:
    """Read the canonical category vocabulary from verified taxonomy data.

    ``A-CMN-001`` is the verified attribute whose ``allowed_values`` defines the
    product-category vocabulary; this is data plumbing, not a category decision.
    """
    attribute = repository.get_attribute("A-CMN-001")
    if attribute is None:
        return []
    return list(attribute.allowed_values)


class HumanClassifier:
    """Maps an explicitly human-provided category to a ``CategoryResult``.

    Only a human-confirmed value becomes ``RESOLVED``. ``unsupported`` /
    ``uncertain`` are treated as terminal clarifications, never as guesses.
    """

    def __init__(self, allowed_values: list[str]) -> None:
        self._allowed_values = list(allowed_values)

    def classify(
        self, product_description: str, provided_category: str | None = None
    ) -> CategoryResult:
        value = (provided_category or "").strip()
        if not value:
            return CategoryResult(
                category=None,
                category_source=CategorySource.UNRESOLVED,
                category_status=CategoryStatus.NEEDS_INFO,
            )
        if value not in self._allowed_values:
            return CategoryResult(
                category=None,
                category_source=CategorySource.UNRESOLVED,
                category_status=CategoryStatus.NEEDS_INFO,
            )
        if value == "unsupported":
            return CategoryResult(
                category=value,
                category_source=CategorySource.HUMAN_CONFIRMED,
                category_status=CategoryStatus.UNSUPPORTED,
            )
        if value == "uncertain":
            return CategoryResult(
                category=value,
                category_source=CategorySource.HUMAN_CONFIRMED,
                category_status=CategoryStatus.NEEDS_INFO,
            )
        return CategoryResult(
            category=value,
            category_source=CategorySource.HUMAN_CONFIRMED,
            category_status=CategoryStatus.RESOLVED,
        )


class StubClassifier:
    """Deterministic test double returning a fixed ``CategoryResult``."""

    def __init__(self, result: CategoryResult) -> None:
        self._result = result

    def classify(
        self, product_description: str, provided_category: str | None = None
    ) -> CategoryResult:
        return self._result


class AgentClassifier:
    """Delegates to an LLM suggestion function and normalizes it in code.

    ``suggest_fn`` returns the model's raw suggested category (or ``None``); this
    class validates it against the verified vocabulary and NEVER produces
    ``RESOLVED``. The model suggestion is ``agent_generated``.
    """

    def __init__(
        self, suggest_fn: Callable[[str], str | None], allowed_values: list[str]
    ) -> None:
        self._suggest_fn = suggest_fn
        self._allowed_values = list(allowed_values)

    def classify(
        self, product_description: str, provided_category: str | None = None
    ) -> CategoryResult:
        suggested = self._suggest_fn(product_description)
        return agent_suggestion(suggested, self._allowed_values)


def agent_suggestion(
    category: str | None,
    allowed_values: list[str],
    classifier_confidence: float | None = None,
) -> CategoryResult:
    """Validate and normalize an LLM category suggestion. NEVER ``RESOLVED``.

    - A valid supported category -> ``agent_generated`` + ``REVIEW_REQUIRED``.
    - ``unsupported`` -> ``agent_generated`` + ``REVIEW_REQUIRED`` (only human
      confirmation may declare a product unsupported).
    - empty / ``uncertain`` / invalid / out-of-vocabulary -> ``NEEDS_INFO``.
    """
    value = (category or "").strip()
    if value in ("", "uncertain") or value not in allowed_values:
        return CategoryResult(
            category=None,
            category_source=CategorySource.AGENT_GENERATED,
            category_status=CategoryStatus.NEEDS_INFO,
            classifier_confidence=classifier_confidence,
        )
    return CategoryResult(
        category=value,
        category_source=CategorySource.AGENT_GENERATED,
        category_status=CategoryStatus.REVIEW_REQUIRED,
        classifier_confidence=classifier_confidence,
    )
