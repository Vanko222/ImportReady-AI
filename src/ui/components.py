"""Streamlit rendering components for the consumer UI.

Presentation only: these functions read canonical results, render them, and
return simple UI intents (which button was pressed, what the user typed). They
never modify a canonical value.

Visual rules:

* every color comes from a semantic theme token (``--ir-*``, see
  :mod:`src.ui.theme`) - no literal colors in this module;
* every user-visible string comes from the central translation system
  (:func:`src.ui.i18n.t`) - no hard-coded UI copy;
* canonical values (rule ids, source ids, enum members, regulatory prose) are
  rendered verbatim and HTML-escaped;
* native Streamlit elements are used wherever they fit; the small number of
  ImportReady-owned HTML blocks exist because they need theme tokens Streamlit
  does not expose for custom layout.
"""

from __future__ import annotations

import html
from typing import Any, Mapping, MutableMapping

import streamlit as st

from src.models import ProductAttribute
from src.services.actions import ActionPlan, ActionType
from src.services.analysis import AnalysisResult
from src.services.classification import CategoryResult
from src.ui import pipeline
from src.ui import state as ui_state
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
from src.ui.theme import apply_theme

# Distinct sentinel for the "I don't know" / "no change" controlled values; an
# approved enum member can never contain a NUL character.
_UNKNOWN = "\x00unknown"

_KEY_INPUT = "ir_key_input"
_KEY_INPUT_CLEAR = "ir_key_input_clear"

# Note kinds -> theme-token class.
_NOTE_KINDS = {
    "info": "ir-note ir-note--info",
    "success": "ir-note ir-note--success",
    "warning": "ir-note ir-note--warning",
    "error": "ir-note ir-note--error",
}


# --------------------------------------------------------------------------- #
# Small HTML helpers (the only place ImportReady markup is produced)
# --------------------------------------------------------------------------- #
def esc(value: Any) -> str:
    """Escape an arbitrary value for safe inclusion in our HTML blocks."""
    return html.escape("" if value is None else str(value))


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def note(kind: str, text: str) -> None:
    """Render one ImportReady note (theme-token surface, never a native alert)."""
    css_class = _NOTE_KINDS.get(kind, _NOTE_KINDS["info"])
    _html(f'<div class="{css_class}">{esc(text)}</div>')


def section(title: str) -> None:
    _html(f'<div class="ir-section">{esc(title)}</div>')


def subsection(title: str) -> None:
    _html(f'<div class="ir-subsection">{esc(title)}</div>')


def chips(*parts: str, accent: bool = False) -> str:
    css = "ir-chip ir-chip--accent" if accent else "ir-chip"
    return "".join(f'<span class="{css}">{esc(part)}</span>' for part in parts if part)


def inject_theme(theme: str) -> None:
    """Apply the selected theme style sheet (no file and no config writes)."""
    apply_theme(theme)


# --------------------------------------------------------------------------- #
# Chrome
# --------------------------------------------------------------------------- #
def render_header(lang: str) -> None:
    _html(
        '<div class="ir-hero">'
        f"<h1>{esc(t('app_title', lang))}</h1>"
        f'<p class="ir-tagline">{esc(t("app_subtitle", lang))} · '
        f"{esc(t('app_tagline', lang))}</p>"
        f'<p class="ir-disclaimer">{esc(t("disclaimer", lang))}</p>'
        "</div>"
    )


def render_flash(flash: Mapping[str, str] | None, lang: str) -> None:
    if not flash:
        return
    key = flash.get("key")
    if not key:
        return
    message = error_message(key, lang) or t(key, lang)
    note(flash.get("level", "info"), message)


# --------------------------------------------------------------------------- #
# Sidebar: presentation controls, then model & access, then reset
# --------------------------------------------------------------------------- #
def _on_language_change() -> None:
    ui_state.set_language(
        st.session_state, st.session_state.get(ui_state.KEY_LANGUAGE_CONTROL)
    )


def _on_theme_change() -> None:
    ui_state.set_theme(st.session_state, st.session_state.get(ui_state.KEY_THEME_CONTROL))


def render_sidebar(session_state: MutableMapping[str, Any], lang: str) -> None:
    with st.sidebar:
        language = ui_state.normalize_language(session_state.get(ui_state.KEY_LANGUAGE))
        st.segmented_control(
            t("language_label", lang),
            options=["en", "zh"],
            default=language,
            format_func=lambda code: LANGUAGE_LABELS.get(code, code),
            key=ui_state.KEY_LANGUAGE_CONTROL,
            on_change=_on_language_change,
            required=True,
            width="stretch",
        )

        theme = ui_state.normalize_theme(session_state.get(ui_state.KEY_THEME))
        st.segmented_control(
            t("appearance_label", lang),
            options=["light", "dark"],
            default=theme,
            format_func=lambda name: t(f"theme_{name}", lang),
            key=ui_state.KEY_THEME_CONTROL,
            on_change=_on_theme_change,
            required=True,
            width="stretch",
        )

        st.divider()
        subsection(t("settings_title", lang))
        _render_access_panel(session_state, lang)

        st.divider()
        if st.button(t("start_over", lang), key="ir_start_over", width="stretch"):
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
            note("success", t("demo_provider_available", lang))
        elif status.reason == "not_verified":
            # Configured, but the exact registered combination is not VERIFIED:
            # a normal setup state, presented calmly rather than as an error.
            note("info", t("demo_provider_not_verified", lang))
            st.caption(t("demo_deterministic_note", lang))
        else:
            note("info", t("demo_provider_not_configured", lang))
            st.caption(t("demo_deterministic_note", lang))
        _render_credential_controls(session_state, lang, allow_input=False)
        return

    options = ui_state.consumer_provider_options()
    if not options:
        note("info", t("provider_none_available", lang))
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
        width="stretch",
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
    section(t("product_section", lang))
    st.caption(t("product_helper", lang))
    st.text_area(
        t("product_description", lang),
        key=ui_state.KEY_DESCRIPTION,
        height=150,
        placeholder=t("product_placeholder", lang),
    )
    return st.button(
        t("analyze_product", lang), key="ir_analyze", type="primary", width="stretch"
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
        section(t("category_section", lang))
        note("success", f"{t('category_confirmed', lang)}: {category_label(confirmed, lang)}")
        return confirmed, False

    if not session_state.get(ui_state.KEY_DESCRIPTION):
        return None, False

    choices = pipeline.consumer_category_choices(repository)
    if not choices:
        return None, False

    suggestion = _suggestion_result(session_state)
    suggestion_error = session_state.get(ui_state.KEY_SUGGESTION_ERROR)

    section(t("category_section", lang))

    suggested_category = suggestion.category if suggestion is not None else None
    if suggested_category:
        _html(
            '<div class="ir-card">'
            f'<div class="ir-card-title">{esc(t("ai_suggested_category", lang))}</div>'
            f'<div class="ir-line">{esc(category_label(suggested_category, lang))}</div>'
            f'<div class="ir-meta">{esc(t("suggestion_candidate_note", lang))}</div>'
            "</div>"
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
    else:
        note("info", t("no_suggestion", lang))
        if suggestion_error == "classification_failed":
            note("warning", error_message("classification_failed", lang) or "")
        elif suggestion_error == "sensitive_input":
            note("error", error_message("sensitive_input", lang) or "")

    mode_labels = [t("confirm_category", lang), t("change_category", lang)]
    default_mode = 0 if suggested_category else 1
    mode = st.segmented_control(
        t("category_choice", lang),
        options=[0, 1],
        default=default_mode,
        format_func=lambda value: mode_labels[value],
        key=f"ir_confirm_mode_{suggested_category or 'none'}",
        required=True,
        width="stretch",
    )
    if mode is None:
        mode = default_mode

    if mode == 0 and suggested_category:
        selected = suggested_category
        _html(
            '<div class="ir-card ir-card--quiet">'
            f'<div class="ir-line"><strong>{esc(t("category_choice", lang))}</strong>: '
            f"{esc(category_label(selected, lang))}</div>"
            "</div>"
        )
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
# Section 3: missing information (progressive disclosure)
# --------------------------------------------------------------------------- #
def _canonical_question(plan: ActionPlan | None, attribute_id: str) -> str | None:
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
        choice = st.segmented_control(
            label,
            options=["yes", "no", "unknown"],
            default="unknown",
            format_func=lambda value: {
                "yes": t("answer_yes", lang),
                "no": t("answer_no", lang),
                "unknown": t("answer_unknown", lang),
            }[value],
            key=key,
            help=help_text,
            required=True,
            width="stretch",
        )
        return {"yes": True, "no": False, "unknown": None}[choice or "unknown"]

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

    section(t("missing_section", lang))
    if missing:
        st.caption(t("missing_intro", lang))

    if issues:
        note("warning", t("input_issues_note", lang))
        for issue in issues:
            _html(
                '<div class="ir-line">'
                f'<span class="ir-inline-code">{esc(issue.attribute_id)}</span> '
                f'<span class="ir-inline-code">{esc(issue.issue_code.value)}</span> — '
                f"{esc(issue.detail)}</div>"
            )

    answers: dict[str, Any] = {}
    if not missing:
        return answers, False

    key_items, additional_items = pipeline.missing_information_split(repository, analysis)

    if key_items:
        subsection(t("key_information", lang))
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

    updated = st.button(
        t("update_analysis", lang), key="ir_update_analysis", type="primary"
    )
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
    section(t("results_section", lang))
    if analysis is None:
        note("info", t("results_placeholder", lang))
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
    subsection(t("summary_section", lang))
    lines = "".join(f'<div class="ir-line">{esc(line)}</div>' for line in view.lines)
    meta = (
        f"{esc(t('label_category', lang))}: <strong>{esc(view.category_label)}</strong>"
        f" &nbsp;·&nbsp; {esc(t('label_review_status', lang))}: "
        f"<strong>{esc(view.review_label)}</strong>"
        f" &nbsp;·&nbsp; {esc(t('label_risk_level', lang))}: "
        f"<strong>{esc(view.risk_label)}</strong>"
    )
    _html(f'<div class="ir-card">{lines}<div class="ir-meta">{meta}</div></div>')


def _risk_item_html(item: Any) -> str:
    return (
        '<div class="ir-item">'
        f'<div class="ir-item-title">{esc(item.title)}</div>'
        f"<div>{chips(item.level_label)}</div>"
        "</div>"
    )


def _render_risk(analysis: AnalysisResult, lang: str) -> None:
    view = risk_view(analysis.risk, lang)
    subsection(t("top_risks", lang))
    body = [
        '<div class="ir-card">',
        f'<div class="ir-line"><strong>{esc(t("label_risk_level", lang))}</strong>: '
        f"{esc(view.level_label)}</div>",
    ]
    if view.empty_message:
        body.append(f'<div class="ir-meta">{esc(view.empty_message)}</div>')
    body.extend(_risk_item_html(item) for item in view.items)
    body.append("</div>")
    _html("".join(body))

    if view.remaining:
        # A real control: every remaining canonical RiskItem is rendered here.
        with st.expander(t("view_remaining", lang, count=len(view.remaining))):
            _html(
                '<div class="ir-card">'
                + "".join(_risk_item_html(item) for item in view.remaining)
                + "</div>"
            )

    all_items = list(view.items) + list(view.remaining)
    if all_items:
        with st.expander(t("detail_show", lang)):
            for item in all_items:
                _html(
                    f'<div class="ir-line"><strong>{esc(item.title)}</strong></div>'
                    f"<div>{chips(item.rule_id or item.gap_id or '', item.level, item.reason_code or '')}</div>"
                    f'<div class="ir-meta">{esc(item.applicability_status or "")} · '
                    f'{esc(item.evidence_status or "")} · {esc(item.rule_status or "")} · '
                    f'{esc(", ".join(item.missing_attribute_ids))}</div>'
                )
    if view.notes:
        with st.expander(t("risk_notes", lang)):
            for text in view.notes:
                st.caption(text)


def _render_requirements(analysis: AnalysisResult, lang: str) -> None:
    cards = requirement_cards(analysis, lang)
    subsection(t("applicable_requirements", lang))
    if not cards:
        note("info", t("no_requirements", lang))
        st.caption(t("applicable_requirements_note", lang))
        return
    st.caption(t("applicable_requirements_note", lang))
    for card in cards:
        rows = [
            '<div class="ir-card">',
            f'<div class="ir-card-title">{esc(card.requirement)}</div>',
            "<div>",
            chips(
                card.applicability_status_label,
                f"{t('authority', lang)}: {card.authority}",
                card.requirement_type,
                accent=True,
            ),
            "</div>",
            f'<div class="ir-meta">'
            f"{esc(requirement_context_label(card.applicability_status, lang))}</div>",
        ]
        if card.clarification_question:
            rows.append(f'<div class="ir-meta">{esc(card.clarification_question)}</div>')
        rows.append("</div>")
        _html("".join(rows))
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
    subsection(t("recommended_actions", lang))
    if view.empty_message:
        note("info", view.empty_message)
        return
    # Every canonical action of a group is rendered inside its expander; the
    # expander is the only collapsing mechanism (no permanent top-N truncation).
    for group in view.groups:
        with st.expander(
            f"{group.label} ({group.total})", expanded=group.expanded_by_default
        ):
            for item in group.items:
                _html(
                    '<div class="ir-item">'
                    f'<div class="ir-item-title">{esc(item.title)}</div>'
                    + (
                        f'<div class="ir-line">{esc(item.canonical_text)}</div>'
                        if item.canonical_text
                        else ""
                    )
                    + "<div>"
                    + chips(
                        item.priority_label,
                        item.action_type,
                        f"{t('rule_id', lang)}: {item.rule_id}" if item.rule_id else "",
                        item.attribute_id or "",
                        item.cost_id or "",
                    )
                    + "</div></div>"
                )
    if view.notes:
        with st.expander(t("action_notes", lang)):
            for text in view.notes:
                st.caption(text)


def _render_cost(analysis: AnalysisResult, lang: str) -> None:
    view = cost_view(analysis.cost, lang)
    subsection(t("cost_references", lang))
    st.caption(view.intro)
    if view.empty_message:
        note("info", view.empty_message)
    for item in view.items:
        amount = (
            f'<div class="ir-line"><strong>{esc(t("cost_amount", lang))}</strong>: '
            f'<span class="ir-inline-code">{esc(item.amount_text)}</span></div>'
            if item.amount_text
            else f'<div class="ir-meta">{esc(t("cost_no_amount", lang))}</div>'
        )
        _html(
            '<div class="ir-card">'
            f'<div class="ir-card-title">{esc(item.cost_item)}</div>'
            f"<div>{chips(item.status_label)}</div>"
            f"{amount}</div>"
        )
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
            for text in view.notes:
                st.caption(text)


def _render_evidence(analysis: AnalysisResult, lang: str) -> None:
    view = evidence_view(analysis, lang)
    with st.expander(t("evidence_sources", lang)):
        st.caption(t("evidence_intro", lang))
        if view.empty_message:
            st.caption(view.empty_message)
        for item in view.items:
            rows = [
                '<div class="ir-item">',
                f'<div class="ir-item-title">{esc(item.title)}</div>',
                f'<div class="ir-meta">{esc(item.authority)}</div>',
                "<div>",
                chips(item.source_id, item.evidence_status, item.source_tier or ""),
                "</div>",
            ]
            if item.rule_ids:
                rows.append(
                    f'<div class="ir-meta">{esc(t("rule_id", lang))}: '
                    f'{esc(", ".join(item.rule_ids))}</div>'
                )
            if item.last_checked:
                rows.append(
                    f'<div class="ir-meta">{esc(t("last_checked", lang))}: '
                    f"{esc(item.last_checked)}</div>"
                )
            if item.url:
                rows.append(
                    f'<div class="ir-meta"><a href="{esc(item.url)}" target="_blank" '
                    f'rel="noopener noreferrer">{esc(t("canonical_url", lang))}</a></div>'
                )
            rows.append("</div>")
            _html("".join(rows))
        if view.limitations:
            subsection(t("known_limitations", lang))
            for limitation in view.limitations:
                _html(
                    '<div class="ir-line">'
                    f'<span class="ir-inline-code">{esc(limitation["gap_id"])}</span> '
                    f'{esc(limitation["area"])} — {esc(limitation["issue"])}</div>'
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
    subsection(t("what_if_section", lang))
    st.caption(t("what_if_intro", lang))
    if not attributes:
        note("info", t("what_if_no_controls", lang))
        return {}, False, False

    facts = pipeline.build_facts(session_state.get(ui_state.KEY_FACT_ANSWERS))
    current = pipeline.current_fact_values(facts)

    overrides: dict[str, Any] = {}
    for attribute in attributes:
        if attribute.data_type == "boolean":
            rendered = st.segmented_control(
                attribute.attribute_name,
                options=[_UNKNOWN, True, False],
                default=_UNKNOWN,
                format_func=lambda value: (
                    t("no_change", lang)
                    if value == _UNKNOWN
                    else fact_value_text(value, lang)
                ),
                key=f"ir_wi_{attribute.attribute_id}",
                help=attribute.why_it_matters or None,
                required=True,
                width="stretch",
            )
            st.caption(
                f"{t('current_value', lang)}: "
                f"{fact_value_text(current.get(attribute.attribute_id), lang)}"
            )
            if rendered is not None and rendered != _UNKNOWN:
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
                f"{t('current_value', lang)}: "
                f"{fact_value_text(current.get(attribute.attribute_id), lang)}"
            )
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
        note("info", t("what_if_not_available", lang))
        return

    rows = [f"<div>{chips(t('hypothetical_scenario', lang), accent=True)}</div>"]
    for override in view.overrides:
        rows.append(
            '<div class="ir-item">'
            f'<span class="ir-inline-code">{esc(override.attribute_id)}</span> — '
            f"{esc(t('current_value', lang))}: <strong>{esc(override.before_text)}</strong>"
            f" → {esc(t('what_if_value', lang))}: "
            f"<strong>{esc(override.after_text)}</strong>"
            "</div>"
        )
    if view.no_changes:
        rows.append(f'<div class="ir-meta">{esc(t("no_changes", lang))}</div>')
    for title_key, changes in (
        ("changed_requirements", view.rule_changes),
        ("changed_risks", view.risk_changes),
        ("changed_actions", view.action_changes),
        ("changed_costs", view.cost_changes),
    ):
        if not changes:
            continue
        rows.append(f'<div class="ir-subsection">{esc(t(title_key, lang))}</div>')
        for change in changes:
            rows.append(
                '<div class="ir-item">'
                f'<span class="ir-inline-code">{esc(change.identifier)}</span> '
                f"{chips(change.change_type_label)}"
                f'<div class="ir-line">{esc(change.summary)}</div>'
                "</div>"
            )
    _html(f'<div class="ir-card">{"".join(rows)}</div>')

    details = [
        (change.identifier, change.details)
        for change in (
            list(view.rule_changes)
            + list(view.risk_changes)
            + list(view.action_changes)
            + list(view.cost_changes)
        )
        if change.details
    ]
    if details:
        with st.expander(t("detail_show", lang)):
            for identifier, change_details in details:
                _html(
                    '<div class="ir-line"><span class="ir-inline-code">'
                    f"{esc(identifier)}</span></div>"
                )
                for label, value in change_details:
                    st.caption(f"{label}: {value}")
    if view.notes:
        with st.expander(t("what_if_notes", lang)):
            for text in view.notes:
                st.caption(text)
