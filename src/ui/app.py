"""Consumer Web UI page (Streamlit).

Flow: product description -> AI category suggestion -> human confirmation ->
missing information -> canonical analysis -> results -> what-if -> evidence.

This page only orchestrates. Every compliance decision comes from the accepted
deterministic engines (via :mod:`src.ui.pipeline`), and every rendered value comes
from :mod:`src.ui.presenters`.

Rendering rule for consistency: any action that changes canonical state stores it
in session state, queues a short message and immediately calls ``st.rerun()``, so
each render reflects exactly one consistent state (no half-updated page).
"""

from __future__ import annotations

from typing import Any, MutableMapping

import streamlit as st

from src.services.analysis import AnalysisResult
from src.services.what_if import WhatIfResult
from src.ui import components as ui
from src.ui import pipeline
from src.ui import state as ui_state
from src.ui.i18n import t
from src.ui.resources import get_analysis_service, get_repository

PAGE_TITLE = "ImportReady AI"


def _lang(session_state: MutableMapping[str, Any]) -> str:
    return ui_state.normalize_language(session_state.get(ui_state.KEY_LANGUAGE))


def _theme(session_state: MutableMapping[str, Any]) -> str:
    return ui_state.normalize_theme(session_state.get(ui_state.KEY_THEME))


def _load_analysis(session_state: MutableMapping[str, Any]) -> AnalysisResult | None:
    raw = session_state.get(ui_state.KEY_ANALYSIS)
    if not raw:
        return None
    try:
        return AnalysisResult.model_validate(raw)
    except Exception:  # noqa: BLE001 - a corrupt session payload never crashes the page
        session_state[ui_state.KEY_ANALYSIS] = None
        session_state[ui_state.KEY_ANALYSIS_META] = None
        return None


def _load_what_if(session_state: MutableMapping[str, Any]) -> WhatIfResult | None:
    raw = session_state.get(ui_state.KEY_WHAT_IF)
    if not raw:
        return None
    try:
        return WhatIfResult.model_validate(raw)
    except Exception:  # noqa: BLE001
        session_state[ui_state.KEY_WHAT_IF] = None
        return None


def _resolve_model(session_state: MutableMapping[str, Any]) -> tuple[Any | None, str | None]:
    """Resolve a model for one action; return ``(model, safe_error_code)``.

    In Competition Demo mode an unavailable server provider is a normal,
    already-explained state (not configured, or configured but not certified for
    consumer demo use) rather than an error banner.
    """
    resolution = ui_state.resolve_model(session_state)
    if resolution.ok:
        return resolution.model, None
    if session_state.get(ui_state.KEY_MODE) == ui_state.MODE_DEMO and resolution.error_code in (
        ui_state.ERROR_PROVIDER_NOT_CONFIGURED,
        ui_state.ERROR_PROVIDER_NOT_VERIFIED,
    ):
        return None, None
    return None, resolution.error_code


def _run_canonical_analysis(
    session_state: MutableMapping[str, Any],
    repository: Any,
    analysis_service: Any,
    category: str,
    model: Any | None,
) -> None:
    """Human-confirmed canonical analysis; stores the canonical result in session."""
    category_result = pipeline.category_result_for(repository, category)
    facts = pipeline.build_facts(session_state.get(ui_state.KEY_FACT_ANSWERS))
    outcome = pipeline.run_analysis(
        repository,
        analysis_service,
        category_result,
        facts=facts,
        model=model,
        description=session_state.get(ui_state.KEY_DESCRIPTION) or "",
    )
    session_state[ui_state.KEY_CONFIRMED_CATEGORY] = category
    session_state[ui_state.KEY_WHAT_IF] = None
    if outcome.analysis is None:
        session_state[ui_state.KEY_ANALYSIS] = None
        session_state[ui_state.KEY_ANALYSIS_META] = None
        ui_state.set_flash(
            session_state, outcome.error_code or ui_state.ERROR_ANALYSIS_FAILED, "error"
        )
        return
    session_state[ui_state.KEY_ANALYSIS] = outcome.analysis.to_dict()
    session_state[ui_state.KEY_ANALYSIS_META] = {
        "review_status": outcome.review_status,
        "agent_runtime": outcome.agent_runtime,
        "classification_runtime": outcome.classification_runtime,
        "error_code": outcome.error_code,
        "exit_code": outcome.exit_code,
        "request_id": outcome.request_id,
    }
    if outcome.error_code:
        # The AI explanation failed; the deterministic analysis is preserved and
        # explicitly reported as requiring human review.
        ui_state.set_flash(session_state, outcome.error_code, "warning")


def _handle_analyze(
    session_state: MutableMapping[str, Any],
    repository: Any,
) -> None:
    """Start a new product: reset state, then attempt an AI category suggestion."""
    description = (session_state.get(ui_state.KEY_DESCRIPTION) or "").strip()
    if not description:
        ui_state.set_flash(session_state, "describe_required", "warning")
        return

    # A new product invalidates the previous confirmation, analysis and scenario.
    session_state[ui_state.KEY_CONFIRMED_CATEGORY] = None
    session_state[ui_state.KEY_ANALYSIS] = None
    session_state[ui_state.KEY_ANALYSIS_META] = None
    session_state[ui_state.KEY_WHAT_IF] = None
    session_state[ui_state.KEY_WHAT_IF_INPUTS] = {}
    session_state[ui_state.KEY_FACT_ANSWERS] = {}
    session_state[ui_state.KEY_SUGGESTION] = None
    session_state[ui_state.KEY_SUGGESTION_ERROR] = None

    model, error_code = _resolve_model(session_state)
    outcome = pipeline.suggest_category(repository, description, model)
    if outcome.suggestion is not None:
        session_state[ui_state.KEY_SUGGESTION] = outcome.suggestion.model_dump(mode="json")
    if outcome.error_code == "sensitive_input":
        session_state[ui_state.KEY_SUGGESTION_ERROR] = "sensitive_input"
        ui_state.set_flash(session_state, "sensitive_input", "error")
    elif outcome.error_code:
        session_state[ui_state.KEY_SUGGESTION_ERROR] = outcome.error_code
    elif error_code:
        ui_state.set_flash(session_state, error_code, "error")


def _handle_update_facts(
    session_state: MutableMapping[str, Any],
    repository: Any,
    analysis_service: Any,
    answers: dict[str, Any],
) -> None:
    """Merge explicit user answers and re-run the canonical analysis."""
    stored = dict(session_state.get(ui_state.KEY_FACT_ANSWERS) or {})
    stored.update(answers)
    session_state[ui_state.KEY_FACT_ANSWERS] = stored
    category = session_state.get(ui_state.KEY_CONFIRMED_CATEGORY)
    if not category:
        return
    model, error_code = _resolve_model(session_state)
    _run_canonical_analysis(session_state, repository, analysis_service, category, model)
    if error_code:
        ui_state.set_flash(session_state, error_code, "error")


def _attribute_name(repository: Any, attribute_id: str) -> str | None:
    attribute = repository.get_attribute(attribute_id)
    return attribute.attribute_name if attribute is not None else None


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    session_state = st.session_state
    ui_state.initialize_state(session_state)
    # Presentation controls mirror the language/theme state keys; reconcile before
    # the widgets are created so the visible control can never disagree with the
    # rendered language or theme.
    ui_state.reconcile_control(
        session_state, ui_state.KEY_LANGUAGE_CONTROL, ui_state.KEY_LANGUAGE
    )
    ui_state.reconcile_control(session_state, ui_state.KEY_THEME_CONTROL, ui_state.KEY_THEME)

    ui.inject_theme(_theme(session_state))
    lang = _lang(session_state)

    ui.render_sidebar(session_state, lang)
    lang = _lang(session_state)
    ui.render_header(lang)
    ui.render_flash(ui_state.take_flash(session_state), lang)

    repository = get_repository()
    analysis_service = get_analysis_service()

    # ---------------------------------------------------------------- section 1
    if ui.render_product_section(session_state, lang):
        _handle_analyze(session_state, repository)
        st.rerun()

    # ---------------------------------------------------------------- section 2
    confirmed_category, confirm_clicked = ui.render_category_section(
        session_state, repository, lang
    )
    if confirm_clicked and confirmed_category:
        model, error_code = _resolve_model(session_state)
        _run_canonical_analysis(
            session_state, repository, analysis_service, confirmed_category, model
        )
        if error_code:
            ui_state.set_flash(session_state, error_code, "error")
        st.rerun()

    analysis = _load_analysis(session_state)
    meta = session_state.get(ui_state.KEY_ANALYSIS_META) or {}

    # ---------------------------------------------------------------- section 3
    if analysis is not None:
        answers, update_clicked = ui.render_missing_information(
            session_state, repository, analysis, lang
        )
        if update_clicked:
            _handle_update_facts(session_state, repository, analysis_service, answers)
            st.rerun()

    # ---------------------------------------------------------------- section 4
    ui.render_results(session_state, repository, analysis, meta, lang)

    if analysis is None:
        return

    category = session_state.get(ui_state.KEY_CONFIRMED_CATEGORY)
    category_result = (
        pipeline.category_result_for(repository, category) if category else None
    )
    if category_result is None or category_result.category_status.value != "RESOLVED":
        st.markdown(f"#### {t('what_if_section', lang)}")
        st.info(t("what_if_not_available", lang))
        return

    overrides, run_what_if, reset_what_if = ui.render_what_if(
        session_state, repository, analysis, category, lang
    )
    if reset_what_if:
        ui_state.reset_what_if(session_state)
        st.rerun()
    if run_what_if:
        if not overrides:
            ui_state.set_flash(session_state, "what_if_override_required", "warning")
            st.rerun()
        session_state[ui_state.KEY_WHAT_IF_INPUTS] = dict(overrides)
        facts = pipeline.build_facts(session_state.get(ui_state.KEY_FACT_ANSWERS))
        outcome = pipeline.run_what_if(
            analysis_service, repository, category_result, facts, overrides
        )
        if outcome.ok and outcome.result is not None:
            session_state[ui_state.KEY_WHAT_IF] = outcome.result.to_dict()
        else:
            session_state[ui_state.KEY_WHAT_IF] = None
            ui_state.set_flash(
                session_state, outcome.error_code or ui_state.ERROR_ANALYSIS_FAILED, "error"
            )
        st.rerun()

    what_if_result = _load_what_if(session_state)
    if what_if_result is not None:
        ui.render_what_if_result(
            what_if_result,
            lambda attribute_id: _attribute_name(repository, attribute_id),
            lang,
        )


if __name__ == "__main__":  # pragma: no cover - streamlit executes the module
    main()
