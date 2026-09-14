"""Configurable model provider factory.

Reads ``MODEL_PROVIDER`` / ``MODEL_ID`` / ``AWS_REGION`` from the environment.
No credentials are read from ``.env`` automatically and none are logged.

Provider access is governed by an **exact** ``(provider_id, model_id)``
combination registry. A provider name alone never authorises an arbitrary model
id: an unregistered combination is refused on every path, and a new model is
onboarded by explicitly registering its exact combination as ``EXPERIMENTAL``.

The registry owns provider identity, the endpoint strategy (base URL) and the
credential scope. Step builders never re-read ``MODEL_ID`` and never accept a
caller-supplied base URL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from strands.models.model import Model

# Official DeepSeek OpenAI-compatible endpoint (no /v1 suffix is required by the
# adapter: the OpenAI SDK appends the Chat Completions path itself).
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# DeepSeek V4 enables thinking mode by default. ImportReady's Strands tool-call flow
# does not replay `reasoning_content` across tool-call turns, so the official Chat
# Completions option is sent explicitly, scoped to the exact `deepseek-v4-flash`
# combination only. The OpenAI SDK accepts `extra_body` as a documented
# `chat.completions.create` keyword argument and the Strands OpenAIModel forwards
# combination `params` into that call verbatim, so this is the smallest supported
# mechanism (no message rewriting, no reasoning_content handling).
#
# Historical evidence kept in the regression suite: a `thinking`/`extra_body`
# payload sent to the *legacy* DeepSeek model id returned HTTP 400, which is why the
# option is declared per combination and the legacy id declares none.
_DEEPSEEK_V4_NON_THINKING_PARAMS: dict[str, Any] = {
    "extra_body": {"thinking": {"type": "disabled"}},
}


class CompatibilityStatus(str, Enum):
    """Certification state of one exact provider + model combination."""

    VERIFIED = "VERIFIED"  # full certification passed + recorded human sign-off
    EXPERIMENTAL = "EXPERIMENTAL"  # registered for developer/certification use only
    UNSUPPORTED = "UNSUPPORTED"  # recognised but deliberately refused


@dataclass(frozen=True)
class ModelCombination:
    """One exact, independently certified ``(provider_id, model_id)`` pair."""

    provider_id: str
    model_id: str  # exact; never a wildcard or free-form value
    compatibility_status: CompatibilityStatus
    ui_exposed: bool = False  # consumer-visible only when True AND VERIFIED
    certification_ref: str | None = None
    notes: str = ""
    # Provider request options scoped to THIS exact combination only. A combination
    # that does not declare any is built with no extra request parameters, so a
    # model-specific option can never leak to another model or provider.
    request_params: Mapping[str, Any] = field(default_factory=dict)


# build(combination, runtime_fields, credential) -> Strands model object
#   combination    : the already-resolved registered combination (owns model_id)
#   runtime_fields : provider-scoped NON-SECRET runtime values, e.g. AWS_REGION
#   credential     : provider-scoped secret, or None (e.g. the AWS credential chain)
ProviderBuilder = Callable[[ModelCombination, Mapping[str, str], str | None], object]


@dataclass(frozen=True)
class ProviderDescriptor:
    """A registered provider: endpoint strategy, credential scope, combinations."""

    provider_id: str
    display_name: str
    credential_scope: tuple[str, ...]  # credential inputs handed to this builder
    runtime_env: tuple[str, ...]  # required non-secret runtime configuration names
    base_url: str | None  # ImportReady-owned; never caller-supplied
    build: ProviderBuilder
    combinations: tuple[ModelCombination, ...] = ()


def _build_none(
    combination: ModelCombination, runtime_fields: Mapping[str, str], credential: str | None
) -> object:
    """The offline deterministic path has no model object."""
    return None


def _build_bedrock(
    combination: ModelCombination, runtime_fields: Mapping[str, str], credential: str | None
) -> object:
    from strands.models.bedrock import BedrockModel

    return BedrockModel(
        model_id=combination.model_id,
        region_name=runtime_fields["AWS_REGION"],
    )


def _openai_compatible_builder(base_url: str) -> ProviderBuilder:
    """Return a builder bound to an ImportReady-owned endpoint.

    Request parameters are read from the resolved **combination**, so a
    model-specific option is sent only for the exact combination that declares it.
    """

    def _build(
        combination: ModelCombination, runtime_fields: Mapping[str, str], credential: str | None
    ) -> object:
        from strands.models.openai import OpenAIModel

        params = dict(getattr(combination, "request_params", None) or {})
        return OpenAIModel(
            client_args={"base_url": base_url, "api_key": credential},
            model_id=combination.model_id,
            **({"params": params} if params else {}),
        )

    return _build


PROVIDER_REGISTRY: dict[str, ProviderDescriptor] = {
    "none": ProviderDescriptor(
        provider_id="none",
        display_name="Offline deterministic mode",
        credential_scope=(),
        runtime_env=(),
        base_url=None,
        build=_build_none,
        combinations=(
            ModelCombination(
                provider_id="none",
                model_id="none",
                compatibility_status=CompatibilityStatus.VERIFIED,
                ui_exposed=False,
                notes="Internal offline mode; never offered as a consumer provider choice.",
            ),
        ),
    ),
    "bedrock": ProviderDescriptor(
        provider_id="bedrock",
        display_name="Amazon Bedrock",
        credential_scope=(),  # AWS credential chain; not the API_KEY variable
        runtime_env=("AWS_REGION",),
        base_url=None,
        build=_build_bedrock,
        # No exact Bedrock model id is approved yet. Onboarding a Bedrock model
        # means registering its exact combination as EXPERIMENTAL first.
        combinations=(),
    ),
    "deepseek": ProviderDescriptor(
        provider_id="deepseek",
        display_name="DeepSeek (OpenAI-compatible)",
        credential_scope=("API_KEY",),
        runtime_env=(),
        base_url=_DEEPSEEK_BASE_URL,
        build=_openai_compatible_builder(_DEEPSEEK_BASE_URL),
        combinations=(
            ModelCombination(
                provider_id="deepseek",
                model_id="deepseek-v4-flash",
                # PROMOTED after the final live certification passed human review:
                # 28 PASS / 0 FAIL / 0 BLOCKED / 6 NOT_RUN, requests 11/33 (see
                # DeepSeek_V4_Final_Certification_Report.md §0.13).
                compatibility_status=CompatibilityStatus.VERIFIED,
                ui_exposed=True,
                # Explicit non-thinking mode, scoped to this exact combination.
                request_params=_DEEPSEEK_V4_NON_THINKING_PARAMS,
                notes=(
                    "CERTIFIED (human-approved): official DeepSeek V4 Flash on the "
                    "OpenAI-compatible Chat Completions endpoint with thinking mode explicitly "
                    "disabled. This exact combination is the only consumer-exposed DeepSeek "
                    "combination; no other DeepSeek model inherits this status."
                ),
            ),
            ModelCombination(
                provider_id="deepseek",
                model_id="deepseek-flash",
                compatibility_status=CompatibilityStatus.EXPERIMENTAL,
                ui_exposed=False,
                notes=(
                    "Legacy/stale DeepSeek model id, retained only so historical development "
                    "runs fail loudly instead of being silently reused. It is NOT the formal "
                    "certification target, is not VERIFIED and is never consumer-exposed."
                ),
            ),
            ModelCombination(
                provider_id="deepseek",
                model_id="deepseek-chat",
                compatibility_status=CompatibilityStatus.UNSUPPORTED,
                notes="Recognised DeepSeek model id; deliberately not used by ImportReady.",
            ),
        ),
    ),
}


def _combination_index(
    registry: Mapping[str, ProviderDescriptor],
) -> dict[tuple[str, str], ModelCombination]:
    index: dict[tuple[str, str], ModelCombination] = {}
    for descriptor in registry.values():
        for combination in descriptor.combinations:
            index[(descriptor.provider_id, combination.model_id)] = combination
    return index


COMBINATION_INDEX: dict[tuple[str, str], ModelCombination] = _combination_index(PROVIDER_REGISTRY)


def resolve_combination(provider_id: str, model_id: str) -> ModelCombination | None:
    """Return the registered combination, or ``None`` when it is not registered.

    ``None`` means "not consumer-eligible" and is refused by every caller; it is
    never a licence to fall back to a free-form model id.
    """
    if not provider_id or not model_id:
        return None
    return COMBINATION_INDEX.get((str(provider_id).strip().lower(), str(model_id).strip()))


def _registered_model_ids(provider_id: str) -> str:
    descriptor = PROVIDER_REGISTRY.get(provider_id)
    if descriptor is None:
        return ""
    return ", ".join(combination.model_id for combination in descriptor.combinations)


def _read_credential(provider: str, descriptor: ProviderDescriptor) -> str | None:
    """Read this provider's credential from the names it alone declares."""
    if not descriptor.credential_scope:
        return None
    if any(not (os.environ.get(name) or "").strip() for name in descriptor.credential_scope):
        raise ValueError(
            f"MODEL_PROVIDER={provider} requires {' and '.join(descriptor.credential_scope)}"
        )
    return (os.environ.get(descriptor.credential_scope[0]) or "").strip()


def build_model() -> Model | None:
    """Return a Strands model instance, or ``None`` for the offline path.

    Development / CLI / certification path. ``MODEL_PROVIDER=none`` (the default)
    is the deterministic offline path. A registered ``EXPERIMENTAL`` or
    ``VERIFIED`` combination is allowed here; an unregistered or ``UNSUPPORTED``
    combination is refused. The ``ui_exposed`` gate belongs to the consumer path.
    """
    provider = os.environ.get("MODEL_PROVIDER", "none").strip().lower()
    if provider in ("", "none"):
        return None

    descriptor = PROVIDER_REGISTRY.get(provider)
    if descriptor is None:
        supported = ", ".join(PROVIDER_REGISTRY)
        raise ValueError(f"Unsupported MODEL_PROVIDER: {provider!r} (supported: {supported})")

    model_id = os.environ.get("MODEL_ID", "").strip()
    required = ("MODEL_ID",) + descriptor.runtime_env
    if any(not (os.environ.get(name) or "").strip() for name in required):
        names = " and ".join(required)
        suffix = " environment variables" if len(required) > 1 else ""
        raise ValueError(f"MODEL_PROVIDER={provider} requires {names}{suffix}")

    combination = resolve_combination(provider, model_id)
    if combination is None:
        raise ValueError(
            f"Unregistered model combination: MODEL_PROVIDER={provider!r} MODEL_ID={model_id!r} "
            f"(registered for {provider}: {_registered_model_ids(provider) or 'none'})"
        )
    if combination.compatibility_status is CompatibilityStatus.UNSUPPORTED:
        raise ValueError(
            f"UNSUPPORTED model combination: MODEL_PROVIDER={provider!r} MODEL_ID={model_id!r} "
            "is recognised but deliberately refused"
        )

    # MODEL_ID is only a lookup key: the resolved combination owns the model id.
    credential = _read_credential(provider, descriptor)
    runtime_fields = {
        name: (os.environ.get(name) or "").strip() for name in descriptor.runtime_env
    }
    return descriptor.build(combination, runtime_fields, credential)  # type: ignore[return-value]


@dataclass(frozen=True)
class RuntimeProviderConfig:
    """Request-scoped provider configuration for a hosted (BYOK) runtime.

    ``credential`` is a secret supplied for this request only. It must never
    mutate ``os.environ``, and must never reach ``.env``, Git, a database,
    ``CaseState``, ``ProductFact``, evidence, logs, status reports, exception
    messages, or any cache.
    """

    provider_id: str
    model_id: str
    # repr=False: the request-scoped secret must never surface through logging,
    # debugging, or any other implicit string conversion.
    credential: str | None = field(default=None, repr=False)
    runtime_fields: Mapping[str, str] = field(default_factory=dict)


def build_model_from_runtime_config(config: RuntimeProviderConfig) -> Model:
    """Build a model from a request-scoped config; consumer/BYOK runtime path.

    Only an exact ``VERIFIED`` combination with ``ui_exposed=True`` is allowed.
    The endpoint comes from the provider descriptor, the credential stays
    request-scoped (never written to ``os.environ``), and only the resolved
    provider's declared runtime fields and credential reach its builder.
    """
    provider = str(config.provider_id or "").strip().lower()
    descriptor = PROVIDER_REGISTRY.get(provider)
    if descriptor is None:
        supported = ", ".join(PROVIDER_REGISTRY)
        raise ValueError(f"Unsupported MODEL_PROVIDER: {provider!r} (supported: {supported})")

    combination = resolve_combination(provider, config.model_id)
    if combination is None:
        model_id = str(config.model_id or "").strip()
        raise ValueError(
            f"Unregistered model combination: MODEL_PROVIDER={provider!r} MODEL_ID={model_id!r} "
            f"(registered for {provider}: {_registered_model_ids(provider) or 'none'})"
        )
    if (
        combination.compatibility_status is not CompatibilityStatus.VERIFIED
        or not combination.ui_exposed
    ):
        raise ValueError(
            f"Not available for consumer runtime use: MODEL_PROVIDER={provider!r} "
            f"MODEL_ID={combination.model_id!r} "
            f"(status={combination.compatibility_status.value}, ui_exposed={combination.ui_exposed})"
        )

    runtime_fields = {
        name: str(config.runtime_fields.get(name, "") or "").strip()
        for name in descriptor.runtime_env
    }
    if any(not value for value in runtime_fields.values()):
        raise ValueError(
            f"MODEL_PROVIDER={provider} requires "
            f"{' and '.join(descriptor.runtime_env)} runtime configuration"
        )

    credential: str | None = None
    if descriptor.credential_scope:
        if config.credential is None or not str(config.credential).strip():
            raise ValueError(
                f"MODEL_PROVIDER={provider} requires "
                f"{' and '.join(descriptor.credential_scope)}"
            )
        credential = str(config.credential).strip()

    return descriptor.build(combination, runtime_fields, credential)  # type: ignore[return-value]


def verified_combinations() -> list[dict]:
    """Read-only source for a consumer provider picker: ``VERIFIED`` + exposed."""
    combinations: list[dict] = []
    for descriptor in PROVIDER_REGISTRY.values():
        for combination in descriptor.combinations:
            if (
                combination.compatibility_status is CompatibilityStatus.VERIFIED
                and combination.ui_exposed
            ):
                combinations.append(
                    {
                        "provider_id": descriptor.provider_id,
                        "display_name": descriptor.display_name,
                        "model_id": combination.model_id,
                        "certification_ref": combination.certification_ref,
                    }
                )
    return combinations


def provider_status_report() -> list[dict]:
    """Read-only diagnostics. Contains status metadata only, never credentials."""
    report: list[dict] = []
    for descriptor in PROVIDER_REGISTRY.values():
        report.append(
            {
                "provider_id": descriptor.provider_id,
                "display_name": descriptor.display_name,
                "base_url": descriptor.base_url,
                "credential_scope": list(descriptor.credential_scope),
                "runtime_env": list(descriptor.runtime_env),
                "combinations": [
                    {
                        "model_id": combination.model_id,
                        "compatibility_status": combination.compatibility_status.value,
                        "ui_exposed": combination.ui_exposed,
                        "certification_ref": combination.certification_ref,
                        "notes": combination.notes,
                    }
                    for combination in descriptor.combinations
                ],
            }
        )
    return report
