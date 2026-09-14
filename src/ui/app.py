"""Consumer Web UI page (Streamlit).

Customer flow (Consumer UX v2):

    one natural-language product box
      -> AI category suggestion (candidate only)
      -> "AI understood" review (suggested category + extracted candidates)
      -> explicit customer confirmation
      -> deterministic canonical analysis
      -> concise natural-language customer report
      -> optional scenario analysis (What-if, collapsed)
      -> optional Technical details (every canonical id/status/source, collapsed)

This page only orchestrates. Every compliance decision comes from the accepted
deterministic engines (via :mod:`src.ui.pipeline`); the narrative report is composed
deterministically by :mod:`src.ui.narrative`; the customer's text and the AI-extracted
candidates are handled by :mod:`src.ui.intake` and never become canonical facts before
the customer confirms them.

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
from src.ui import intake as intake_flow
from src.ui import pipeline
from src.ui import state as ui_state
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


def _category_status(session_state: MutableMapping[str, Any]) -> str | None:
    """Canonical category status of the confirmed category (never inferred)."""
    category = session_state.get(ui_state.KEY_CONFIRMED_CATEGORY)
    if not category:
        return None
    if category == "unsupported":
        return "UNSUPPORTED"
    if category == "uncertain":
        return "NEEDS_INFO"
    return "RESOLVED"


def _handle_intake_analyze(
    session_state: MutableMapping[str, Any],
    repository: Any,
) -> None:
    """Start a new product: reset derived state, then suggest + extract."""
    description = (session_state.get(ui_state.KEY_DESCRIPTION) or "").strip()
    if not description:
        ui_state.set_flash(session_state, "intake_no_text", "warning")
        return

    # A new Analyze action starts a new workflow: drop the previous product's category
    # widget state so the selector defaults to THIS suggestion (never auto-confirmed).
    ui_state.reset_category_control(session_state)

    # A new intake invalidates the previous confirmation, analysis and scenario.
    session_state[ui_state.KEY_CONFIRMED_CATEGORY] = None
    session_state[ui_state.KEY_ANALYSIS] = None
    session_state[ui_state.KEY_ANALYSIS_META] = None
    session_state[ui_state.KEY_WHAT_IF] = None
    session_state[ui_state.KEY_WHAT_IF_INPUTS] = {}
    session_state[ui_state.KEY_FACT_ANSWERS] = {}
    session_state[ui_state.KEY_SUGGESTION] = None
    session_state[ui_state.KEY_SUGGESTION_ERROR] = None
    intake_flow.clear_intake_extraction(session_state)

    # Sensitive-input guard runs before any model is constructed or called.
    if pipeline.contains_sensitive_input(description):
        session_state[ui_state.KEY_SUGGESTION_ERROR] = "sensitive_input"
        session_state[intake_flow.KEY_INTAKE_ERROR] = "sensitive_input"
        session_state[intake_flow.KEY_INTAKE_TEXT_USED] = description
        ui_state.set_flash(session_state, "intake_sensitive", "error")
        return

    model, error_code = _resolve_model(session_state)
    outcome = pipeline.suggest_category(repository, description, model)
    if outcome.suggestion is not None:
        session_state[ui_state.KEY_SUGGESTION] = outcome.suggestion.model_dump(mode="json")
    if outcome.error_code:
        session_state[ui_state.KEY_SUGGESTION_ERROR] = outcome.error_code
    if error_code:
        ui_state.set_flash(session_state, error_code, "error")

    # Extraction runs for the SUGGESTED category only as a convenience: the customer
    # must still confirm the category, and any later category change re-runs the
    # extraction with the human-selected vocabulary (never the suggestion's).
    suggested = outcome.suggestion.category if outcome.suggestion is not None else None
    intake_flow.run_extraction(
        session_state, repository, description, suggested, model
    )


def _handle_confirm_bundle(
    session_state: MutableMapping[str, Any],
    repository: Any,
    analysis_service: Any,
    category: str,
) -> None:
    """Explicit confirmation of the visible bundle for exactly this category.

    The intake layer refuses (and discards) a bundle that was extracted with another
    category's vocabulary, so no stale candidate can become a USER fact.
    """
    answers = intake_flow.confirm_bundle(session_state, repository, category)
    if not intake_flow.is_confirmed(session_state):
        ui_state.set_flash(session_state, "intake_bundle_stale", "warning")
        return
    model, error_code = _resolve_model(session_state)
    _run_canonical_analysis(session_state, repository, analysis_service, category, model)
    if error_code:
        ui_state.set_flash(session_state, error_code, "error")


def _handle_use_category(
    session_state: MutableMapping[str, Any],
    repository: Any,
    category: str,
) -> None:
    """Human confirmed/changed the category: read the details for THAT category.

    Re-runs the extraction against the human-selected category's approved
    vocabulary and drops any bundle (and analysis) derived from a different one.
    """
    session_state[ui_state.KEY_CONFIRMED_CATEGORY] = None
    session_state[ui_state.KEY_ANALYSIS] = None
    session_state[ui_state.KEY_ANALYSIS_META] = None
    session_state[ui_state.KEY_WHAT_IF] = None
    session_state[ui_state.KEY_WHAT_IF_INPUTS] = {}
    session_state[ui_state.KEY_FACT_ANSWERS] = {}
    intake_flow.clear_intake_extraction(session_state)
    model, error_code = _resolve_model(session_state)
    intake_flow.run_extraction(
        session_state,
        repository,
        session_state.get(ui_state.KEY_DESCRIPTION) or "",
        category,
        model,
    )
    if error_code:
        ui_state.set_flash(session_state, error_code, "error")


def _handle_update_facts(
    session_state: MutableMapping[str, Any],
    repository: Any,
    analysis_service: Any,
    answers: dict[str, Any],
) -> None:
    """Merge explicit answers from the advanced view and re-run the analysis."""
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


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    session_state = st.session_state
    ui_state.initialize_state(session_state)
    intake_flow.initialize_intake_state(session_state)
    # An edited description invalidates unconfirmed extraction and its analysis.
    if intake_flow.invalidate_if_stale(session_state):
        ui_state.set_flash(session_state, "intake_edit_hint", "info")
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

    # ------------------------------------------------------- step 1: intake
    if ui.render_intake_section(session_state, lang):
        _handle_intake_analyze(session_state, repository)
        st.rerun()

    analysis = _load_analysis(session_state)
    meta = session_state.get(ui_state.KEY_ANALYSIS_META) or {}
    confirmed_category = session_state.get(ui_state.KEY_CONFIRMED_CATEGORY)

    # ------------------------------------- step 2: AI understood + confirmation
    if not confirmed_category:
        selected, action = ui.render_ai_understood(session_state, repository, lang)
        if action == "edit":
            intake_flow.clear_intake_extraction(session_state)
            ui_state.set_flash(session_state, "intake_edit_hint", "info")
            st.rerun()
        if action == "use_category" and selected:
            # Human category boundary: the details are (re)read with this category.
            _handle_use_category(session_state, repository, selected)
            st.rerun()
        if action == "confirm" and selected:
            _handle_confirm_bundle(
                session_state, repository, analysis_service, selected
            )
            st.rerun()
        return

    # ------------------------------------------- step 3: customer report
    ui.render_customer_report(
        repository,
        analysis,
        lang,
        _category_status(session_state),
        commercial_notes=intake_flow.notes(session_state),
    )

    if analysis is None:
        return

    category_result = pipeline.category_result_for(repository, confirmed_category)
    resolved = category_result.category_status.value == "RESOLVED"

    # ------------------------------------------- step 4: optional scenario
    if resolved:
        what_if_result = _load_what_if(session_state)
        overrides, run_what_if, reset_what_if = ui.render_scenario_section(
            session_state, repository, analysis, confirmed_category, lang, what_if_result
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
                    session_state,
                    outcome.error_code or ui_state.ERROR_ANALYSIS_FAILED,
                    "error",
                )
            st.rerun()

    # ------------------------------------------- step 5: technical details
    answers, update_clicked = ui.render_technical_details(
        session_state, repository, analysis, meta, lang
    )
    if update_clicked:
        _handle_update_facts(session_state, repository, analysis_service, answers)
        st.rerun()


if __name__ == "__main__":  # pragma: no cover - streamlit executes the module
    main()
