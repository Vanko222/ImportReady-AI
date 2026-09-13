"""Consumer UI session state, credential handling and provider resolution.

Two hard rules live here:

1. **BYOK credentials are session-only.** The key is stored exclusively under
   :data:`KEY_BYOK_CREDENTIAL` in Streamlit ``session_state``. It is never
   written to ``.env``/disk/database/logs/Git, never copied into ``CaseState``,
   ``ProductFact`` or any analysis result, and never placed in a
   ``st.cache_data`` / ``st.cache_resource`` function.
2. **No arbitrary provider configuration.** Consumer provider selection is read
   from the existing provider registry (:func:`src.agent.model_factory.verified_combinations`),
   so only ``VERIFIED`` + ``ui_exposed`` combinations are offered. No Base URL
   and no free-form model id is ever accepted from the consumer.

This module deliberately contains no compliance, risk, cost, action or
applicability logic; it only manages presentation/session state and delegates
model construction to the accepted provider factory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, MutableMapping, Protocol

from src.agent.model_factory import (
    PROVIDER_REGISTRY,
    CompatibilityStatus,
    RuntimeProviderConfig,
    build_model,
    build_model_from_runtime_config,
    resolve_combination,
    verified_combinations,
)
from src.ui.i18n import normalize_language  # re-exported for the UI layer

# --------------------------------------------------------------------------- #
# Session keys (single source of truth, never string literals at call sites)
# --------------------------------------------------------------------------- #
KEY_LANGUAGE = "ir_language"
KEY_MODE = "ir_access_mode"
KEY_PROVIDER = "ir_byok_provider"
KEY_BYOK_CREDENTIAL = "ir_byok_credential"  # the only place a consumer key may live
KEY_DESCRIPTION = "ir_product_description"
KEY_SUGGESTION = "ir_category_suggestion"
KEY_SUGGESTION_ERROR = "ir_suggestion_error"
KEY_CONFIRMED_CATEGORY = "ir_confirmed_category"
KEY_FACT_ANSWERS = "ir_fact_answers"
KEY_ANALYSIS = "ir_analysis"
KEY_ANALYSIS_META = "ir_analysis_meta"
KEY_WHAT_IF = "ir_what_if"
KEY_WHAT_IF_INPUTS = "ir_what_if_inputs"
KEY_FLASH = "ir_flash"

MODE_DEMO = "demo"
MODE_BYOK = "byok"
ACCESS_MODES: tuple[str, ...] = (MODE_DEMO, MODE_BYOK)

#: Every session key that belongs to one product analysis. ``start_over`` clears
#: exactly these and preserves presentation preferences (language) plus, by
#: deliberate decision, the session-only BYOK key (Clear Key remains explicit).
ANALYSIS_SESSION_KEYS: tuple[str, ...] = (
    KEY_DESCRIPTION,
    KEY_SUGGESTION,
    KEY_SUGGESTION_ERROR,
    KEY_CONFIRMED_CATEGORY,
    KEY_FACT_ANSWERS,
    KEY_ANALYSIS,
    KEY_ANALYSIS_META,
    KEY_WHAT_IF,
    KEY_WHAT_IF_INPUTS,
    KEY_FLASH,
)

#: Fixed mask. It is a constant, never derived from the key material, so no part
#: of a stored credential can ever be echoed back to the browser.
CREDENTIAL_MASK = "••••••••"

# Safe, closed error vocabulary for the UI (never a traceback, never SDK text).
ERROR_PROVIDER_NOT_CONFIGURED = "provider_not_configured"
ERROR_PROVIDER_NOT_VERIFIED = "provider_not_verified"
ERROR_PROVIDER_NONE_AVAILABLE = "provider_none_available"
ERROR_PROVIDER_MISSING_KEY = "provider_missing_key"
ERROR_PROVIDER_REJECTED = "provider_rejected"
ERROR_PROVIDER_UNAVAILABLE = "provider_unavailable"
ERROR_ANALYSIS_FAILED = "analysis_failed"
ERROR_WHAT_IF_INPUT = "what_if_input"
ERROR_WHAT_IF_CONFIGURATION = "what_if_configuration"
ERROR_UNSUPPORTED_CATEGORY = "unsupported_category"


class SessionLike(Protocol):
    """The subset of ``st.session_state`` this module uses (dict-like)."""

    def __contains__(self, key: object) -> bool: ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def pop(self, key: str, default: Any = ...) -> Any: ...


@dataclass(frozen=True)
class ProviderOption:
    """One consumer-selectable provider, read from the existing registry."""

    provider_id: str
    display_name: str
    model_id: str
    certification_ref: str | None = None


@dataclass(frozen=True)
class DemoProviderStatus:
    """Read-only status of the server-side Competition Demo provider path.

    ``reason`` is ``None`` only when a consumer-ready provider is available;
    otherwise it is ``"not_configured"`` (no server provider configured) or
    ``"not_verified"`` (configured, but the exact registered combination is not
    ``VERIFIED`` and therefore not consumer-ready).
    """

    available: bool
    provider_id: str | None = None
    display_name: str | None = None
    reason: str | None = None


@dataclass
class ModelResolution:
    """Outcome of resolving a model for one UI action.

    ``credential_present`` is a boolean only; the credential itself is never
    carried on this object (so it cannot leak through logging or serialization).
    """

    model: Any | None = None
    provider_id: str | None = None
    display_name: str | None = None
    error_code: str | None = None
    credential_present: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.model is not None and self.error_code is None


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def initialize_state(session_state: MutableMapping[str, Any]) -> None:
    """Apply deterministic defaults without clobbering an existing session."""
    defaults: dict[str, Any] = {
        KEY_LANGUAGE: "en",
        KEY_MODE: MODE_DEMO,
        KEY_PROVIDER: None,
        KEY_BYOK_CREDENTIAL: None,
        KEY_DESCRIPTION: "",
        KEY_SUGGESTION: None,
        KEY_SUGGESTION_ERROR: None,
        KEY_CONFIRMED_CATEGORY: None,
        KEY_FACT_ANSWERS: {},
        KEY_ANALYSIS: None,
        KEY_ANALYSIS_META: None,
        KEY_WHAT_IF: None,
        KEY_WHAT_IF_INPUTS: {},
        KEY_FLASH: None,
    }
    for key, value in defaults.items():
        if key not in session_state:
            session_state[key] = value


def start_over(session_state: MutableMapping[str, Any]) -> None:
    """Clear the current analysis, preserving language and the session BYOK key."""
    for key in ANALYSIS_SESSION_KEYS:
        session_state.pop(key, None)
    session_state[KEY_FACT_ANSWERS] = {}
    session_state[KEY_WHAT_IF_INPUTS] = {}
    initialize_state(session_state)


def reset_what_if(session_state: MutableMapping[str, Any]) -> None:
    """Drop the hypothetical scenario and its raw inputs; the baseline is untouched."""
    session_state[KEY_WHAT_IF] = None
    session_state[KEY_WHAT_IF_INPUTS] = {}


def set_flash(session_state: MutableMapping[str, Any], key: str, level: str = "info") -> None:
    """Queue one short UI message (translation key + level); never free-form text."""
    session_state[KEY_FLASH] = {"key": key, "level": level}


def take_flash(session_state: MutableMapping[str, Any]) -> dict[str, str] | None:
    """Read and clear the queued UI message."""
    flash = session_state.get(KEY_FLASH)
    session_state[KEY_FLASH] = None
    return flash


# --------------------------------------------------------------------------- #
# BYOK credential (session-only)
# --------------------------------------------------------------------------- #
def set_byok_credential(
    session_state: MutableMapping[str, Any], credential: str | None
) -> None:
    """Store a consumer credential for this session only (empty input clears it)."""
    value = (credential or "").strip()
    session_state[KEY_BYOK_CREDENTIAL] = value or None


def get_byok_credential(session_state: MutableMapping[str, Any]) -> str | None:
    """Return the session credential for internal use only.

    Callers must never render, log or serialize the returned value.
    """
    value = session_state.get(KEY_BYOK_CREDENTIAL)
    return value if isinstance(value, str) and value else None


def has_byok_credential(session_state: MutableMapping[str, Any]) -> bool:
    return get_byok_credential(session_state) is not None


def credential_mask() -> str:
    """The only representation of a stored key that may ever be displayed."""
    return CREDENTIAL_MASK


def clear_byok_credential(session_state: MutableMapping[str, Any]) -> None:
    """Remove the session credential (the explicit Clear Key control)."""
    session_state.pop(KEY_BYOK_CREDENTIAL, None)
    session_state[KEY_BYOK_CREDENTIAL] = None


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #
def consumer_provider_options() -> list[ProviderOption]:
    """Providers a consumer may select a key for: ``VERIFIED`` + ``ui_exposed``.

    An ``EXPERIMENTAL`` or non-exposed combination is never promoted here; the
    list is exactly the registry's consumer-eligible set and may legitimately be
    empty.
    """
    return [
        ProviderOption(
            provider_id=entry["provider_id"],
            display_name=entry["display_name"],
            model_id=entry["model_id"],
            certification_ref=entry.get("certification_ref"),
        )
        for entry in verified_combinations()
    ]


def demo_provider_status() -> DemoProviderStatus:
    """Inspect the server-side demo provider configuration without building it.

    The formal Consumer Demo accepts a model only when its **exact registered
    combination** is :attr:`CompatibilityStatus.VERIFIED`. An ``EXPERIMENTAL`` or
    ``UNSUPPORTED`` combination is not consumer-ready, so it is reported as
    unavailable with ``reason="not_verified"`` rather than silently accepted.

    Never reads or returns a credential value: only environment-variable
    *presence* is consulted.
    """
    provider = (os.environ.get("MODEL_PROVIDER") or "").strip().lower()
    if provider in ("", "none"):
        return DemoProviderStatus(available=False, reason="not_configured")
    descriptor = PROVIDER_REGISTRY.get(provider)
    if descriptor is None:
        return DemoProviderStatus(available=False, reason="not_configured")
    model_id = (os.environ.get("MODEL_ID") or "").strip()
    if not model_id:
        return DemoProviderStatus(available=False, reason="not_configured")
    combination = resolve_combination(provider, model_id)
    if combination is None:
        return DemoProviderStatus(available=False, reason="not_configured")
    # Consumer Demo gate: VERIFIED only. EXPERIMENTAL / UNSUPPORTED combinations
    # are recognised but deliberately not offered to consumers.
    if combination.compatibility_status is not CompatibilityStatus.VERIFIED:
        return DemoProviderStatus(available=False, reason="not_verified")
    if any(not (os.environ.get(name) or "").strip() for name in descriptor.runtime_env):
        return DemoProviderStatus(available=False, reason="not_configured")
    if any(not (os.environ.get(name) or "").strip() for name in descriptor.credential_scope):
        return DemoProviderStatus(available=False, reason="not_configured")
    return DemoProviderStatus(
        available=True,
        provider_id=provider,
        display_name=descriptor.display_name,
        reason=None,
    )


def resolve_competition_demo_model() -> ModelResolution:
    """Build the server-configured demo model through the accepted factory.

    The model object is built per action and deliberately **never cached**. A
    configured but non-``VERIFIED`` combination is refused here as well, so an
    EXPERIMENTAL model can never be reached from Competition Demo mode.
    """
    status = demo_provider_status()
    if not status.available:
        if status.reason == "not_verified":
            return ModelResolution(error_code=ERROR_PROVIDER_NOT_VERIFIED)
        return ModelResolution(error_code=ERROR_PROVIDER_NOT_CONFIGURED)
    try:
        model = build_model()
    except ValueError:
        return ModelResolution(error_code=ERROR_PROVIDER_NOT_CONFIGURED)
    except Exception:  # noqa: BLE001 - SDK/transport failures stay opaque to the consumer
        return ModelResolution(error_code=ERROR_PROVIDER_UNAVAILABLE)
    if model is None:
        return ModelResolution(error_code=ERROR_PROVIDER_NOT_CONFIGURED)
    return ModelResolution(
        model=model,
        provider_id=status.provider_id,
        display_name=status.display_name,
        credential_present=True,
    )


def resolve_byok_model(session_state: MutableMapping[str, Any]) -> ModelResolution:
    """Build a model from the session-only credential and a registry provider.

    Only ``VERIFIED`` + ``ui_exposed`` combinations are accepted, and the
    credential is passed request-scoped without touching ``os.environ``.
    """
    options = consumer_provider_options()
    if not options:
        return ModelResolution(error_code=ERROR_PROVIDER_NONE_AVAILABLE)
    selected = session_state.get(KEY_PROVIDER)
    option = next((item for item in options if item.provider_id == selected), None)
    if option is None:
        option = options[0]
    credential = get_byok_credential(session_state)
    if credential is None:
        return ModelResolution(
            provider_id=option.provider_id,
            display_name=option.display_name,
            error_code=ERROR_PROVIDER_MISSING_KEY,
        )
    config = RuntimeProviderConfig(
        provider_id=option.provider_id,
        model_id=option.model_id,
        credential=credential,
    )
    try:
        model = build_model_from_runtime_config(config)
    except ValueError:
        return ModelResolution(
            provider_id=option.provider_id,
            display_name=option.display_name,
            error_code=ERROR_PROVIDER_REJECTED,
            credential_present=True,
        )
    except Exception:  # noqa: BLE001
        return ModelResolution(
            provider_id=option.provider_id,
            display_name=option.display_name,
            error_code=ERROR_PROVIDER_UNAVAILABLE,
            credential_present=True,
        )
    return ModelResolution(
        model=model,
        provider_id=option.provider_id,
        display_name=option.display_name,
        credential_present=True,
    )


def resolve_model(
    session_state: MutableMapping[str, Any], mode: str | None = None
) -> ModelResolution:
    """Resolve the model for the active access mode (never raises)."""
    active = (mode or session_state.get(KEY_MODE) or MODE_DEMO)
    if active == MODE_BYOK:
        return resolve_byok_model(session_state)
    return resolve_competition_demo_model()
