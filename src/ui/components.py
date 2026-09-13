"""Streamlit rendering components for the consumer UI.

Presentation only: these functions read canonical results, render them, and
return simple UI intents (which button was pressed, what the user typed). They
never modify a canonical value. Custom HTML is limited to ImportReady-owned
blocks so no undocumented Streamlit DOM internals are relied upon.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

import streamlit as st

from src.models import ProductAttribute
from src.services.actions import ActionPlan, ActionType
from src.services.analysis import AnalysisResult
from src.services.classification import CategoryResult
from src.ui import pipeline, state as ui_state
from src.ui.i18n import LANGUAGE_LABELS, t
from src.ui.presenters import (
    actions_view,
    category_label,
    cost_view,
    error_message,
    evidence_view,
    fact_value_text,
    requirement_cards,
    requirement_context_label,
    risk_view,
    summary,
    technical_view,
)

# Distinct sentinel for the "I don't know" choice; an approved enum value can
# never contain a NUL character.
_UNKNOWN = "\x00unknown"

_KEY_INPUT = "ir_key_input"
_KEY_INPUT_CLEAR = "ir_key_input_clear"


# --------------------------------------------------------------------------- #
# Chrome
# --------------------------------------------------------------------------- #
def inject_styles() -> None:
    """Small amount of stable CSS for a calm, premium business look."""
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1150px; }
          .ir-hero { border: 1px solid #E3E8EF; border-radius: 14px; padding: 20px 24px;
                     background: #FBFCFD; margin-bottom: 6px; }
          .ir-hero h1 { font-size: 1.65rem; margin: 0 0 2px 0; color: #16273A;
                        letter-spacing: -0.01em; }
          .ir-hero p { margin: 0; color: #55637A; font-size: 0.95rem; }
          .ir-hero .ir-note { margin-top: 10px; color: #7A8699; font-size: 0.78rem; }
          .ir-summary-line { color: #22303F; font-size: 0.95rem; margin: 2px 0; }
          .ir-chip { display: inline-block; border: 1px solid #D8DEE8; border-radius: 999px;
                     padding: 1px 10px; margin-right: 6px; font-size: 0.75rem; color: #3C4A5C;
                     background: #FFFFFF; }
          .ir-meta { color: #6B7788; font-size: 0.78rem; }
          .ir-section-gap { height: 6px; }
          h3 { color: #16273A; letter-spacing: -0.01em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(lang: str) -> None:
    st.markdown(
        f"""
        <div class="ir-hero">
          <h1>{t("app_title", lang)}</h1>
          <p>{t("app_subtitle", lang)}</p>
          <div class="ir-note">{t("disclaimer", lang)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flash(flash: Mapping[str, str] | None, lang: str) -> None:
    if not flash:
        return
    key = flash.get("key")
    if not key:
        return
    level = flash.get("level", "info")
    message = error_message(key, lang) or t(key, lang)
    if level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    elif level == "success":
        st.success(message)
    else:
        st.info(message)


# --------------------------------------------------------------------------- #
# Sidebar: language, access mode, credentials, reset
# --------------------------------------------------------------------------- #
def render_sidebar(session_state: MutableMapping[str, Any], lang: str) -> None:
    with st.sidebar:
        st.markdown(f"#### {t('settings_title', lang)}")

        codes = ["en", "zh"]
        current = ui_state.normalize_language(session_state.get(ui_state.KEY_LANGUAGE))
        choice = st.radio(
            t("language_label", lang),
            options=codes,
            index=codes.index(current),
            format_func=lambda code: LANGUAGE_LABELS.get(code, code),
            horizontal=True,
            key="ir_language_widget",
        )
        if choice != current:
            session_state[ui_state.KEY_LANGUAGE] = choice
            st.rerun()

        st.divider()
        _render_access_panel(session_state, lang)

        st.divider()
        if st.button(t("start_over", lang), key="ir_start_over", use_container_width=True):
            ui_state.start_over(session_state)
            ui_state.set_flash(session_state, "started_over", "success")
            st.rerun()


def _render_access_panel(session_state: MutableMapping[str, Any], lang: str) -> None:
    modes = [ui_state.MODE_DEMO, ui_state.MODE_BYOK]
    labels = {ui_state.MODE_DEMO: t("mode_demo", lang), ui_state.MODE_BYOK: t("mode_byok", lang)}
    active = session_state.get(ui_state.KEY_MODE, ui_state.MODE_DEMO)
    if active not in modes:
        active = ui_state.MODE_DEMO

    selected = st.radio(
        t("mode_label", lang),
        options=modes,
        index=modes.index(active),
        format_func=lambda value: labels[value],
        key="ir_mode_widget",
    )
    session_state[ui_state.KEY_MODE] = selected

    if selected == ui_state.MODE_DEMO:
        status = ui_state.demo_provider_status()
        if status.available:
            st.success(t("demo_provider_available", lang))
        elif status.reason == "not_verified":
            # Configured, but the exact registered combination is not VERIFIED.
            st.warning(t("demo_provider_not_verified", lang))
            st.caption(t("demo_deterministic_note", lang))
        else:
            st.info(t("demo_provider_not_configured", lang))
            st.caption(t("demo_deterministic_note", lang))
        _render_credential_controls(session_state, lang, allow_input=False)
        return

    options = ui_state.consumer_provider_options()
    if not options:
        st.warning(t("provider_none_available", lang))
        _render_credential_controls(session_state, lang, allow_input=False)
        return

    provider_ids = [option.provider_id for option in options]
    current = session_state.get(ui_state.KEY_PROVIDER)
    index = provider_ids.index(current) if current in provider_ids else 0
    chosen = st.selectbox(
        t("provider_label", lang),
        options=provider_ids,
        index=index,
        format_func=lambda value: next(
            option.display_name for option in options if option.provider_id == value
        ),
        key="ir_provider_widget",
    )
    session_state[ui_state.KEY_PROVIDER] = chosen
    _render_credential_controls(session_state, lang, allow_input=True)


def _render_credential_controls(
    session_state: MutableMapping[str, Any],
    lang: str,
    allow_input: bool,
) -> None:
    """Session-only credential input and the explicit Clear Key control.

    The submitted key is copied into session state and then removed from the
    widget key, so the password field never retains it.
    """
    if session_state.pop(_KEY_INPUT_CLEAR, False):
        session_state.pop(_KEY_INPUT, None)

    if allow_input:
        with st.form("ir_byok_form", clear_on_submit=False):
            typed = st.text_input(
                t("api_key_label", lang),
                type="password",
                key=_KEY_INPUT,
            )
            submitted = st.form_submit_button(t("save_key", lang))

        if submitted:
            ui_state.set_byok_credential(session_state, typed)
            session_state[_KEY_INPUT_CLEAR] = True
            if ui_state.has_byok_credential(session_state):
                ui_state.set_flash(session_state, "api_key_stored", "success")
            else:
                ui_state.set_flash(session_state, "api_key_missing", "warning")
            st.rerun()

    if ui_state.has_byok_credential(session_state):
        st.caption(f"{t('api_key_stored', lang)} {ui_state.credential_mask()}")

    if st.button(
        t("clear_key", lang),
        key="ir_clear_key",
        disabled=not ui_state.has_byok_credential(session_state),
        use_container_width=True,
    ):
        ui_state.clear_byok_credential(session_state)
        session_state[_KEY_INPUT_CLEAR] = True
        ui_state.set_flash(session_state, "key_cleared", "success")
        st.rerun()

    st.caption(t("key_privacy_note", lang))


# --------------------------------------------------------------------------- #
# Section 1: product
# --------------------------------------------------------------------------- #
def render_product_section(session_state: MutableMapping[str, Any], lang: str) -> bool:
    st.markdown(f"### {t('product_section', lang)}")
    st.text_area(
        t("product_description", lang),
        key=ui_state.KEY_DESCRIPTION,
        height=120,
        placeholder=t("product_placeholder", lang),
    )
    return st.button(
        t("analyze_product", lang), key="ir_analyze", type="primary"
    )


# --------------------------------------------------------------------------- #
# Section 2: category confirmation
# --------------------------------------------------------------------------- #
def _suggestion_result(session_state: MutableMapping[str, Any]) -> CategoryResult | None:
    raw = session_state.get(ui_state.KEY_SUGGESTION)
    if not raw:
        return None
    try:
        return CategoryResult.model_validate(raw)
    except Exception:  # noqa: BLE001
        return None


def render_category_section(
    session_state: MutableMapping[str, Any],
    repository: Any,
    lang: str,
) -> tuple[str | None, bool]:
    """Render category confirmation; return ``(category, confirmed)``."""
    confirmed = session_state.get(ui_state.KEY_CONFIRMED_CATEGORY)
    if confirmed:
        st.markdown(f"### {t('category_section', lang)}")
        st.success(f"{t('category_confirmed', lang)}: {category_label(confirmed, lang)}")
        return confirmed, False

    if not session_state.get(ui_state.KEY_DESCRIPTION):
        return None, False

    choices = pipeline.consumer_category_choices(repository)
    if not choices:
        return None, False

    suggestion = _suggestion_result(session_state)
    suggestion_error = session_state.get(ui_state.KEY_SUGGESTION_ERROR)

    st.markdown(f"### {t('category_section', lang)}")

    suggested_category = suggestion.category if suggestion is not None else None
    if suggested_category:
        st.markdown(
            f"**{t('ai_suggested_category', lang)}:** {category_label(suggested_category, lang)}"
        )
        with st.expander(t("detail_show", lang)):
            st.write(
                {
                    "category": suggested_category,
                    "category_source": suggestion.category_source.value,
                    "category_status": suggestion.category_status.value,
                    "classifier_confidence": suggestion.classifier_confidence,
                }
            )
        st.caption(t("suggestion_candidate_note", lang))
    else:
        st.info(t("no_suggestion", lang))
        if suggestion_error == "classification_failed":
            st.warning(error_message("classification_failed", lang))
        elif suggestion_error == "sensitive_input":
            st.error(error_message("sensitive_input", lang))

    mode_labels = [t("confirm_category", lang), t("change_category", lang)]
    default_mode = 0 if suggested_category else 1
    mode = st.radio(
        t("category_choice", lang),
        options=[0, 1],
        index=default_mode,
        format_func=lambda value: mode_labels[value],
        horizontal=True,
        key=f"ir_confirm_mode_{suggested_category or 'none'}",
    )

    if mode == 0 and suggested_category:
        selected = suggested_category
        st.markdown(f"**{t('category_choice', lang)}:** {category_label(selected, lang)}")
    else:
        options = choices
        default_index = (
            options.index(suggested_category) if suggested_category in options else 0
        )
        selected = st.selectbox(
            t("category_choice", lang),
            options=options,
            index=default_index,
            format_func=lambda value: category_label(value, lang),
            key=f"ir_category_choice_{suggested_category or 'none'}",
        )

    confirmed_clicked = st.button(
        t("confirm_and_analyze", lang), key="ir_confirm_category", type="primary"
    )
    if confirmed_clicked and selected:
        return selected, True
    return None, False


# --------------------------------------------------------------------------- #
# Section 3: missing information
# --------------------------------------------------------------------------- #
def _canonical_question(
    plan: ActionPlan | None, attribute_id: str
) -> str | None:
    """Reuse the canonical MISSING_INFORMATION question text when available."""
    if plan is None:
        return None
    for item in plan.items:
        if (
            item.action_type is ActionType.MISSING_INFORMATION
            and item.attribute_id == attribute_id
            and item.canonical_text
        ):
            return item.canonical_text
    return None


def _missing_info_input(
    attribute: ProductAttribute,
    question: str,
    lang: str,
    repository: Any,
) -> Any:
    """Typed input for one approved attribute. ``None`` means 'no answer'."""
    label = question
    help_text = attribute.why_it_matters or None
    key = f"ir_fact_{attribute.attribute_id}"

    if attribute.data_type == "boolean":
        choice = st.radio(
            label,
            options=["yes", "no", "unknown"],
            index=2,
            format_func=lambda value: {
                "yes": t("answer_yes", lang),
                "no": t("answer_no", lang),
                "unknown": t("answer_unknown", lang),
            }[value],
            horizontal=True,
            key=key,
            help=help_text,
        )
        return {"yes": True, "no": False, "unknown": None}[choice]

    if attribute.data_type == "enum" and attribute.allowed_values:
        options = [_UNKNOWN] + list(attribute.allowed_values)
        choice = st.selectbox(
            label,
            options=options,
            index=0,
            format_func=lambda value: (
                t("answer_unknown", lang) if value == _UNKNOWN else value
            ),
            key=key,
            help=help_text,
        )
        return None if choice == _UNKNOWN else choice

    if attribute.data_type in ("number", "decimal"):
        # An empty numeric field means "no answer" (no fact is created).
        return st.number_input(label, value=None, step=0.1, key=key, help=help_text)

    if attribute.data_type == "integer":
        return st.number_input(label, value=None, step=1, key=key, help=help_text)

    if attribute.data_type == "text" or attribute.data_type == "structured_text":
        typed = st.text_input(label, key=key, help=help_text)
        return typed.strip() or None

    if attribute.data_type == "structured_list":
        typed = st.text_area(label, key=key, height=68, help=help_text)
        items = [line.strip() for line in typed.splitlines() if line.strip()]
        return items or None

    if attribute.data_type == "date":
        typed = st.text_input(label, key=key, help=help_text, placeholder="YYYY-MM-DD")
        typed = typed.strip()
        if not typed:
            return None
        # The accepted human-fact boundary performs the only type conversion.
        return pipeline.normalize_fact_value(repository, attribute.attribute_id, typed)

    st.caption(f"{label} — {attribute.data_type}")
    return None


def _render_missing_item(
    repository: Any,
    analysis: AnalysisResult,
    item: Any,
    lang: str,
) -> Any:
    attribute = repository.get_attribute(item.attribute_id)
    if attribute is None:
        return None
    question = _canonical_question(analysis.actions, item.attribute_id) or (
        attribute.attribute_name
    )
    return _missing_info_input(attribute, question, lang, repository)


def render_missing_information(
    session_state: MutableMapping[str, Any],
    repository: Any,
    analysis: AnalysisResult | None,
    lang: str,
) -> tuple[dict[str, Any], bool]:
    """Render the canonical missing-information questions; return answers + intent.

    Progressive disclosure only: no canonical missing item is dropped.

    * **Key information** shows the missing attributes of the rules that have a
      valid automated ``TriggerSpec`` for the confirmed category (derived from
      canonical applicability + approved spec metadata, never from prose).
    * **Additional information (N)** is a collapsed expander holding every other
      canonical missing attribute, still answerable.

    Answers from both groups follow the same existing ``FactOrigin.USER`` path.
    """
    if analysis is None:
        return {}, False

    missing = analysis.unknown.missing_information
    issues = analysis.applicability.input_issues if analysis.applicability else []

    if not missing and not issues:
        return {}, False

    st.markdown(f"### {t('missing_section', lang)}")
    if missing:
        st.caption(t("missing_intro", lang))

    if issues:
        st.warning(t("input_issues_note", lang))
        for issue in issues:
            st.markdown(
                f"- `{issue.attribute_id}` · `{issue.issue_code.value}` — {issue.detail}"
            )

    answers: dict[str, Any] = {}
    if not missing:
        return answers, False

    key_items, additional_items = pipeline.missing_information_split(repository, analysis)

    if key_items:
        st.markdown(f"**{t('key_information', lang)}**")
        for item in key_items:
            answers[item.attribute_id] = _render_missing_item(
                repository, analysis, item, lang
            )

    if additional_items:
        # Expanded automatically only when there is no key group, so nothing is
        # hidden from the customer on categories without trigger-based facts.
        with st.expander(
            t("additional_information", lang, count=len(additional_items)),
            expanded=not key_items,
        ):
            for item in additional_items:
                answers[item.attribute_id] = _render_missing_item(
                    repository, analysis, item, lang
                )

    updated = st.button(t("update_analysis", lang), key="ir_update_analysis", type="primary")
    return answers, updated


# --------------------------------------------------------------------------- #
# Section 4: results
# --------------------------------------------------------------------------- #
def render_results(
    session_state: MutableMapping[str, Any],
    repository: Any,
    analysis: AnalysisResult | None,
    meta: Mapping[str, Any] | None,
    lang: str,
) -> None:
    st.markdown(f"### {t('results_section', lang)}")
    if analysis is None:
        st.info(t("results_placeholder", lang))
        return

    review_status = (meta or {}).get("review_status")

    _render_summary(analysis, review_status, lang)
    _render_risk(analysis, lang)
    _render_requirements(analysis, lang)
    _render_actions(analysis, lang)
    _render_cost(analysis, lang)

    _render_evidence(analysis, lang)
    _render_technical(analysis, meta, lang)


def _render_summary(
    analysis: AnalysisResult, review_status: str | None, lang: str
) -> None:
    view = summary(analysis, review_status, lang)
    st.markdown(f"#### {t('summary_section', lang)}")
    with st.container(border=True):
        for line in view.lines:
            st.markdown(f"<div class='ir-summary-line'>{line}</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='ir-meta'>"
            f"{t('label_category', lang)}: <b>{view.category_label}</b> &nbsp;·&nbsp; "
            f"{t('label_review_status', lang)}: <b>{view.review_label}</b> &nbsp;·&nbsp; "
            f"{t('label_risk_level', lang)}: <b>{view.risk_label}</b>"
            "</div>",
            unsafe_allow_html=True,
        )


def _render_risk(analysis: AnalysisResult, lang: str) -> None:
    view = risk_view(analysis.risk, lang)
    st.markdown(f"#### {t('top_risks', lang)}")
    with st.container(border=True):
        st.markdown(f"**{t('label_risk_level', lang)}:** {view.level_label}")
        if view.empty_message:
            st.caption(view.empty_message)
        for item in view.items:
            _render_risk_item(item, lang)
        if view.remaining:
            # A real control: every remaining canonical RiskItem is rendered here.
            with st.expander(t("view_remaining", lang, count=len(view.remaining))):
                for item in view.remaining:
                    _render_risk_item(item, lang)
        if view.notes:
            with st.expander(t("risk_notes", lang)):
                for note in view.notes:
                    st.caption(note)


def _render_risk_item(item: Any, lang: str) -> None:
    st.markdown(f"- **{item.title}** — {item.level_label}")
    with st.expander(t("detail_show", lang)):
        st.write(
            {
                "rule_id": item.rule_id,
                "gap_id": item.gap_id,
                "risk_level": item.level,
                "reason_code": item.reason_code,
                "reason_codes_ordered": item.reason_codes_ordered,
                "applicability_status": item.applicability_status,
                "evidence_status": item.evidence_status,
                "rule_status": item.rule_status,
                "missing_attribute_ids": item.missing_attribute_ids,
            }
        )


def _render_requirements(analysis: AnalysisResult, lang: str) -> None:
    cards = requirement_cards(analysis, lang)
    st.markdown(f"#### {t('applicable_requirements', lang)}")
    if not cards:
        st.info(t("no_requirements", lang))
        st.caption(t("applicable_requirements_note", lang))
        return
    st.caption(t("applicable_requirements_note", lang))
    for card in cards:
        with st.container(border=True):
            st.markdown(f"**{card.requirement}**")
            st.markdown(
                f"<span class='ir-chip'>{card.applicability_status_label}</span>"
                f"<span class='ir-chip'>{t('authority', lang)}: {card.authority}</span>"
                f"<span class='ir-chip'>{card.requirement_type}</span>",
                unsafe_allow_html=True,
            )
            st.caption(requirement_context_label(card.applicability_status, lang))
            if card.clarification_question:
                st.caption(card.clarification_question)
            with st.expander(t("detail_show", lang)):
                st.write(
                    {
                        "rule_id": card.rule_id,
                        "applicability_status": card.applicability_status,
                        "reason_codes": card.reason_codes,
                        "evidence_status": card.evidence_status,
                        "rule_status": card.rule_status,
                        "source_ids": card.source_ids,
                        "missing_attribute_ids": card.missing_attribute_ids,
                        "jurisdiction": card.jurisdiction,
                    }
                )


def _render_actions(analysis: AnalysisResult, lang: str) -> None:
    view = actions_view(analysis.actions, lang)
    st.markdown(f"#### {t('recommended_actions', lang)}")
    if view.empty_message:
        st.info(view.empty_message)
        return
    # Every canonical action of a group is rendered inside its expander; the
    # expander is the only collapsing mechanism (no permanent top-N truncation).
    for group in view.groups:
        with st.expander(
            f"{group.label} ({group.total})", expanded=group.expanded_by_default
        ):
            for item in group.items:
                st.markdown(f"- **{item.title}**")
                if item.canonical_text:
                    st.caption(item.canonical_text)
                st.markdown(
                    f"<span class='ir-chip'>{item.priority_label}</span>"
                    f"<span class='ir-chip'>{item.action_type}</span>"
                    + (
                        f"<span class='ir-chip'>{t('rule_id', lang)}: {item.rule_id}</span>"
                        if item.rule_id
                        else ""
                    )
                    + (
                        f"<span class='ir-chip'>{item.attribute_id}</span>"
                        if item.attribute_id
                        else ""
                    )
                    + (
                        f"<span class='ir-chip'>{item.cost_id}</span>"
                        if item.cost_id
                        else ""
                    ),
                    unsafe_allow_html=True,
                )
    if view.notes:
        with st.expander(t("action_notes", lang)):
            for note in view.notes:
                st.caption(note)


def _render_cost(analysis: AnalysisResult, lang: str) -> None:
    view = cost_view(analysis.cost, lang)
    st.markdown(f"#### {t('cost_references', lang)}")
    with st.container(border=True):
        st.caption(view.intro)
        if view.empty_message:
            st.caption(view.empty_message)
        for item in view.items:
            st.markdown(f"**{item.cost_item}** — {item.status_label}")
            if item.amount_text:
                st.markdown(f"{t('cost_amount', lang)}: `{item.amount_text}`")
            else:
                st.caption(t("cost_no_amount", lang))
            with st.expander(t("detail_show", lang)):
                st.write(
                    {
                        "cost_id": item.cost_id,
                        "calculation_status": item.status,
                        "currency": item.currency,
                        "pricing_basis": item.pricing_basis,
                        "scope_included": item.scope_included,
                        "important_exclusions": item.important_exclusions,
                        "price_status": item.price_status,
                        "evidence_status": item.evidence_status,
                        "source_ids": item.source_ids,
                        "source_date": item.source_date,
                        "checked_date": item.checked_date,
                    }
                )
        if view.items or view.total_unavailable_reason:
            st.caption(t("cost_no_total_note", lang))
        if view.notes:
            with st.expander(t("cost_notes", lang)):
                for note in view.notes:
                    st.caption(note)


def _render_evidence(analysis: AnalysisResult, lang: str) -> None:
    view = evidence_view(analysis, lang)
    with st.expander(t("evidence_sources", lang)):
        st.caption(t("evidence_intro", lang))
        if view.empty_message:
            st.caption(view.empty_message)
        for item in view.items:
            st.markdown(f"**{item.title}** — {item.authority}")
            st.markdown(
                f"<span class='ir-chip'>{item.source_id}</span>"
                f"<span class='ir-chip'>{item.evidence_status}</span>"
                + (
                    f"<span class='ir-chip'>{item.source_tier}</span>"
                    if item.source_tier
                    else ""
                ),
                unsafe_allow_html=True,
            )
            if item.rule_ids:
                st.caption(f"{t('rule_id', lang)}: {', '.join(item.rule_ids)}")
            if item.last_checked:
                st.caption(f"{t('last_checked', lang)}: {item.last_checked}")
            if item.url:
                st.markdown(f"[{t('canonical_url', lang)}]({item.url})")
        if view.limitations:
            st.markdown(f"**{t('known_limitations', lang)}**")
            for limitation in view.limitations:
                st.caption(
                    f"{limitation['gap_id']} · {limitation['area']} — {limitation['issue']}"
                )


def _render_technical(
    analysis: AnalysisResult, meta: Mapping[str, Any] | None, lang: str
) -> None:
    with st.expander(t("technical_details", lang)):
        st.caption(t("technical_intro", lang))
        st.caption(t("technical_collapsed_hint", lang))
        st.write(technical_view(analysis, meta, lang))


# --------------------------------------------------------------------------- #
# What-if
# --------------------------------------------------------------------------- #
def render_what_if(
    session_state: MutableMapping[str, Any],
    repository: Any,
    analysis: AnalysisResult | None,
    category: str | None,
    lang: str,
) -> tuple[dict[str, Any], bool, bool]:
    """Render hypothetical controls; return ``(overrides, run, reset)``."""
    if analysis is None or not category:
        return {}, False, False

    attributes = pipeline.what_if_control_attributes(repository, category)
    st.markdown(f"#### {t('what_if_section', lang)}")
    with st.container(border=True):
        st.caption(t("what_if_intro", lang))
        if not attributes:
            st.caption(t("what_if_no_controls", lang))
            return {}, False, False

        facts = pipeline.build_facts(session_state.get(ui_state.KEY_FACT_ANSWERS))
        current = pipeline.current_fact_values(facts)

        overrides: dict[str, Any] = {}
        controlled: list[ProductAttribute] = []
        for attribute in attributes:
            if attribute.data_type == "boolean":
                options: list[Any] = [_UNKNOWN, True, False]
                rendered = st.radio(
                    attribute.attribute_name,
                    options=options,
                    index=0,
                    format_func=lambda value: (
                        t("no_change", lang)
                        if value == _UNKNOWN
                        else fact_value_text(value, lang)
                    ),
                    horizontal=True,
                    key=f"ir_wi_{attribute.attribute_id}",
                    help=attribute.why_it_matters or None,
                )
                st.caption(
                    f"{t('current_value', lang)}: {fact_value_text(current.get(attribute.attribute_id), lang)}"
                )
                controlled.append(attribute)
                if rendered != _UNKNOWN:
                    overrides[attribute.attribute_id] = rendered
                continue

            if attribute.data_type == "enum" and attribute.allowed_values:
                options = [_UNKNOWN] + list(attribute.allowed_values)
                rendered = st.selectbox(
                    attribute.attribute_name,
                    options=options,
                    index=0,
                    format_func=lambda value: (
                        t("no_change", lang) if value == _UNKNOWN else value
                    ),
                    key=f"ir_wi_{attribute.attribute_id}",
                    help=attribute.why_it_matters or None,
                )
                st.caption(
                    f"{t('current_value', lang)}: {fact_value_text(current.get(attribute.attribute_id), lang)}"
                )
                controlled.append(attribute)
                if rendered != _UNKNOWN:
                    overrides[attribute.attribute_id] = rendered
                continue

            st.caption(f"{attribute.attribute_name} — {attribute.data_type}")

        run = st.button(t("run_what_if", lang), key="ir_run_what_if", type="primary")
        reset = st.button(t("reset_what_if", lang), key="ir_reset_what_if")
        return overrides, run, reset


def render_what_if_result(result: Any, name_lookup: Any, lang: str) -> None:
    from src.ui.presenters import what_if_view

    view = what_if_view(result, lang, name_lookup)
    if not view.assessed:
        st.info(t("what_if_not_available", lang))
        return
    with st.container(border=True):
        st.markdown(
            f"<span class='ir-chip'>{t('hypothetical_scenario', lang)}</span>",
            unsafe_allow_html=True,
        )
        if view.overrides:
            for override in view.overrides:
                st.markdown(
                    f"- `{override.attribute_id}` — {t('current_value', lang)}: "
                    f"**{override.before_text}** → {t('what_if_value', lang)}: "
                    f"**{override.after_text}**"
                )
        if view.no_changes:
            st.caption(t("no_changes", lang))
        for title_key, changes in (
            ("changed_requirements", view.rule_changes),
            ("changed_risks", view.risk_changes),
            ("changed_actions", view.action_changes),
            ("changed_costs", view.cost_changes),
        ):
            if not changes:
                continue
            st.markdown(f"**{t(title_key, lang)}**")
            for change in changes:
                st.markdown(
                    f"- `{change.identifier}` · {change.change_type_label} — {change.summary}"
                )
                if change.details:
                    with st.expander(t("detail_show", lang)):
                        for label, value in change.details:
                            st.caption(f"{label}: {value}")
        if view.notes:
            with st.expander(t("what_if_notes", lang)):
                for note in view.notes:
                    st.caption(note)
