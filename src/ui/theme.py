"""Light / Dark theme design tokens and the centralized CSS for the consumer UI.

Why CSS here at all: Streamlit 1.63 exposes no programmatic theme switch (the
active theme is client-side and read-only via ``st.context.theme``), and
``.streamlit/config.toml`` cannot be rewritten at runtime. The task therefore
requires session-state-driven theming implemented with UI styling, so this module
owns the single centralized style sheet.

Design rules kept deliberately narrow and stable:

* Components never invent colors: every color comes from the semantic token set
  below, exposed as ``--ir-*`` CSS custom properties.
* Selectors use Streamlit's public ``data-testid`` hooks, ARIA roles, plain HTML
  element names, and the documented ``.st-key-<key>`` class hook - never generated
  emotion class names, and never deep descendant chains.
* If a future Streamlit version renames a hook, the affected native widget falls
  back to the native theme instead of breaking the page.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

DEFAULT_THEME = "light"
THEMES: tuple[str, ...] = ("light", "dark")

#: Semantic token set (task design tokens). Both themes define every key.
TOKENS: dict[str, dict[str, str]] = {
    "light": {
        "background": "#F6F8FA",
        "surface": "#FFFFFF",
        "surface_alt": "#EDF1F6",
        "surface_elevated": "#FFFFFF",
        "text_primary": "#16273A",
        "text_secondary": "#3F4F63",
        "text_muted": "#5E6B7E",
        "border": "#D3DCE7",
        "border_strong": "#B7C5D6",
        "accent": "#24507C",
        "accent_hover": "#1B3E62",
        "accent_soft": "#E9F0F8",
        # Selected segmented-control surface: clearly distinct from the page and
        # from the unselected control, without becoming saturated.
        "accent_selected": "#D5E3F5",
        "accent_text": "#FFFFFF",
        "success": "#1E7A4B",
        "success_soft": "#E7F4EC",
        "warning": "#8A5A06",
        "warning_soft": "#FCF3E3",
        "danger": "#A33A32",
        "danger_soft": "#FBEBE9",
        "info": "#24507C",
        "info_soft": "#EAF1F9",
        "input_background": "#FFFFFF",
        "sidebar_background": "#F1F4F8",
    },
    "dark": {
        "background": "#101722",
        "surface": "#17202D",
        "surface_alt": "#1D2634",
        "surface_elevated": "#1B2533",
        "text_primary": "#E8EDF4",
        "text_secondary": "#BAC5D3",
        "text_muted": "#97A5B7",
        "border": "#2B3646",
        "border_strong": "#3A475A",
        "accent": "#2E6EA8",
        "accent_hover": "#3C82BF",
        "accent_soft": "#1B2B3D",
        # Kept identical to the previous dark selected surface so the dark theme
        # renders exactly as reviewed (the token exists only for parity).
        "accent_selected": "#1B2B3D",
        "accent_text": "#FFFFFF",
        "success": "#3F9E72",
        "success_soft": "#152A21",
        "warning": "#C79A4B",
        "warning_soft": "#2A2318",
        "danger": "#D07A72",
        "danger_soft": "#2C1E1E",
        "info": "#5E9BD1",
        "info_soft": "#152636",
        "input_background": "#131C28",
        "sidebar_background": "#0C131C",
    },
}

#: Spacing / radius / width tokens (shared by both themes).
LAYOUT: dict[str, str] = {
    "content_width": "1120px",
    "radius": "12px",
    "radius_small": "9px",
}


def normalize_theme(value: Any) -> str:
    """Return a supported theme name, falling back to the default (light)."""
    name = str(value or "").strip().lower()
    return name if name in THEMES else DEFAULT_THEME


def theme_tokens(theme: str) -> dict[str, str]:
    """Semantic tokens for one theme (always a complete mapping)."""
    return dict(TOKENS[normalize_theme(theme)])


def token_names() -> tuple[str, ...]:
    """Every semantic token name both themes must define."""
    return tuple(TOKENS[DEFAULT_THEME].keys())


def _variable_block(theme: str) -> str:
    tokens = theme_tokens(theme)
    declarations = "\n".join(
        f"  --ir-{name.replace('_', '-')}: {value};" for name, value in tokens.items()
    )
    layout = "\n".join(
        f"  --ir-{name.replace('_', '-')}: {value};" for name, value in LAYOUT.items()
    )
    return f":root {{\n{declarations}\n{layout}\n}}"


# The style sheet is static apart from the token block; every color below is a
# token reference so both themes stay consistent by construction.
_STYLESHEET = """
/* ---------------------------------------------------------------- app shell */
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--ir-background);
  color: var(--ir-text-primary);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stMainBlockContainer"],
.block-container {
  max-width: var(--ir-content-width);
  margin: 0 auto;
  padding: 2.1rem 2.4rem 3.5rem 2.4rem;
}
[data-testid="stSidebar"] {
  background: var(--ir-sidebar-background);
  border-right: 1px solid var(--ir-border);
}
[data-testid="stSidebarContent"] { padding: 0.4rem 0.35rem 1.5rem 0.35rem; }
[data-testid="stSidebarUserContent"] { padding-bottom: 1.2rem; }

/* --------------------------------------------------------------- typography */
[data-testid="stMarkdownContainer"] :is(h1, h2, h3, h4, h5, h6, p, li, strong) {
  color: var(--ir-text-primary);
}
[data-testid="stCaptionContainer"] :is(p, span) { color: var(--ir-text-muted); }
[data-testid="stWidgetLabel"] :is(label, p) {
  color: var(--ir-text-primary);
  font-size: 0.86rem;
  font-weight: 500;
}

/* ------------------------------------------------------------------- inputs */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
  background: var(--ir-input-background);
  color: var(--ir-text-primary);
  border: 1px solid var(--ir-border);
  border-radius: var(--ir-radius-small);
}
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder { color: var(--ir-text-muted); }
[data-testid="stSelectbox"] div[role="combobox"] {
  background: var(--ir-input-background);
  color: var(--ir-text-primary);
  border: 1px solid var(--ir-border);
  border-radius: var(--ir-radius-small);
}
[role="listbox"], [role="option"] {
  background: var(--ir-surface);
  color: var(--ir-text-primary);
}
[role="option"]:hover { background: var(--ir-surface-alt); }

/* ------------------------------------------------------------------ buttons */
[data-testid="stButton"] button,
[data-testid="stFormSubmitButton"] button {
  background: var(--ir-surface);
  color: var(--ir-text-primary);
  border: 1px solid var(--ir-border);
  border-radius: var(--ir-radius-small);
  font-weight: 500;
}
[data-testid="stButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover {
  border-color: var(--ir-accent);
  color: var(--ir-accent);
}
/* Streamlit renders a button label inside a markdown container, so the global
   markdown text colour would otherwise win over the button's own foreground.
   These rules keep every button label on the button's colour; the primary rules
   below then pin the high-contrast foreground explicitly. */
[data-testid="stButton"] button :is(p, div, span),
[data-testid="stFormSubmitButton"] button :is(p, div, span) { color: inherit; }
[data-testid="stButton"] button[kind="primary"] :is(p, div, span) {
  color: var(--ir-accent-text);
}
[data-testid="stButtonGroup"] button {
  background: var(--ir-surface);
  color: var(--ir-text-secondary);
  border-color: var(--ir-border);
}
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"] {
  background: var(--ir-accent-selected);
  color: var(--ir-accent);
  border-color: var(--ir-accent);
  font-weight: 600;
}

/* Primary actions (key-scoped, the documented .st-key hook). White-on-blue in
   both themes: the accent token is dark enough for AA text contrast. */
.st-key-ir_analyze button,
.st-key-ir_confirm_category button,
.st-key-ir_update_analysis button,
.st-key-ir_run_what_if button {
  background: var(--ir-accent);
  color: var(--ir-accent-text);
  border-color: var(--ir-accent);
  font-weight: 600;
}
.st-key-ir_analyze button :is(p, div, span),
.st-key-ir_confirm_category button :is(p, div, span),
.st-key-ir_update_analysis button :is(p, div, span),
.st-key-ir_run_what_if button :is(p, div, span) {
  color: var(--ir-accent-text);
}
.st-key-ir_analyze button:hover,
.st-key-ir_confirm_category button:hover,
.st-key-ir_update_analysis button:hover,
.st-key-ir_run_what_if button:hover,
.st-key-ir_analyze button:focus,
.st-key-ir_confirm_category button:focus,
.st-key-ir_update_analysis button:focus,
.st-key-ir_run_what_if button:focus,
.st-key-ir_analyze button:focus-visible,
.st-key-ir_confirm_category button:focus-visible,
.st-key-ir_update_analysis button:focus-visible,
.st-key-ir_run_what_if button:focus-visible {
  background: var(--ir-accent-hover);
  border-color: var(--ir-accent-hover);
  color: var(--ir-accent-text);
}
.st-key-ir_analyze button:hover :is(p, div, span),
.st-key-ir_confirm_category button:hover :is(p, div, span),
.st-key-ir_update_analysis button:hover :is(p, div, span),
.st-key-ir_run_what_if button:hover :is(p, div, span),
.st-key-ir_analyze button:focus :is(p, div, span),
.st-key-ir_confirm_category button:focus :is(p, div, span),
.st-key-ir_update_analysis button:focus :is(p, div, span),
.st-key-ir_run_what_if button:focus :is(p, div, span) {
  color: var(--ir-accent-text);
}

/* ---------------------------------------------------------------- expanders */
[data-testid="stExpander"] details {
  background: var(--ir-surface);
  border: 1px solid var(--ir-border);
  border-radius: var(--ir-radius-small);
}
[data-testid="stExpander"] summary { color: var(--ir-text-primary); font-weight: 500; }

/* ------------------------------------------------------- ImportReady blocks */
.ir-hero {
  border: 1px solid var(--ir-border);
  border-radius: var(--ir-radius);
  background: var(--ir-surface);
  padding: 1.5rem 1.75rem 1.35rem 1.75rem;
  margin-bottom: 1.4rem;
}
.ir-hero h1 {
  margin: 0;
  font-size: 1.6rem;
  line-height: 1.2;
  letter-spacing: -0.015em;
  color: var(--ir-text-primary);
}
.ir-hero .ir-tagline {
  margin: 0.5rem 0 0 0;
  font-size: 0.95rem;
  color: var(--ir-text-secondary);
}
.ir-hero .ir-disclaimer {
  margin: 0.85rem 0 0 0;
  padding-top: 0.7rem;
  border-top: 1px solid var(--ir-border);
  font-size: 0.76rem;
  color: var(--ir-text-muted);
}
.ir-section {
  margin: 1.9rem 0 0.75rem 0;
  font-size: 1.06rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--ir-text-primary);
}
.ir-subsection {
  margin: 0.9rem 0 0.35rem 0;
  font-size: 0.82rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--ir-text-muted);
}
.ir-card {
  background: var(--ir-surface);
  border: 1px solid var(--ir-border);
  border-radius: var(--ir-radius);
  padding: 1.05rem 1.2rem;
  margin-bottom: 0.7rem;
}
.ir-card--quiet { background: var(--ir-surface-alt); }
.ir-card-title {
  margin: 0 0 0.55rem 0;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--ir-text-primary);
}
.ir-line { margin: 0.18rem 0; color: var(--ir-text-primary); font-size: 0.92rem; }
.ir-meta { margin-top: 0.7rem; color: var(--ir-text-muted); font-size: 0.79rem; }
.ir-item { padding: 0.55rem 0; border-top: 1px solid var(--ir-border); }
.ir-item:first-of-type { border-top: none; padding-top: 0.15rem; }
.ir-item-title { font-weight: 600; color: var(--ir-text-primary); font-size: 0.93rem; }
.ir-chip {
  display: inline-block;
  border: 1px solid var(--ir-border-strong);
  border-radius: 999px;
  padding: 0.08rem 0.6rem;
  margin: 0.28rem 0.35rem 0 0;
  font-size: 0.72rem;
  line-height: 1.5;
  color: var(--ir-text-secondary);
  background: var(--ir-surface-alt);
  white-space: nowrap;
}
.ir-chip--accent { border-color: var(--ir-accent); color: var(--ir-accent); background: var(--ir-accent-soft); }
.ir-note {
  border: 1px solid var(--ir-border);
  border-left: 3px solid var(--ir-border-strong);
  border-radius: var(--ir-radius-small);
  background: var(--ir-surface-alt);
  padding: 0.7rem 0.9rem;
  margin: 0.35rem 0 0.6rem 0;
  font-size: 0.88rem;
  color: var(--ir-text-secondary);
}
.ir-note--info { border-left-color: var(--ir-info); background: var(--ir-info-soft); color: var(--ir-info); }
.ir-note--success { border-left-color: var(--ir-success); background: var(--ir-success-soft); color: var(--ir-success); }
.ir-note--warning { border-left-color: var(--ir-warning); background: var(--ir-warning-soft); color: var(--ir-warning); }
.ir-note--error { border-left-color: var(--ir-danger); background: var(--ir-danger-soft); color: var(--ir-danger); }
.ir-inline-code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.78rem;
  color: var(--ir-text-secondary);
  background: var(--ir-surface-alt);
  border: 1px solid var(--ir-border);
  border-radius: 5px;
  padding: 0.02rem 0.32rem;
}
"""


# Light-only definition pass. The dark theme passed human visual review and must
# render exactly as reviewed, so these refinements are emitted only for light mode.
_LIGHT_REFINEMENTS = """
/* Light theme: a 1px inset definition ring makes the selected segmented control
   unambiguous without saturation, a drop shadow or a layout shift. */
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"] {
  box-shadow: inset 0 0 0 1px var(--ir-accent);
}
"""


def theme_css(theme: str) -> str:
    """Complete centralized style sheet for one theme."""
    name = normalize_theme(theme)
    refinements = _LIGHT_REFINEMENTS if name == "light" else ""
    return f"<style>\n{_variable_block(name)}\n{_STYLESHEET}\n{refinements}\n</style>"


def apply_theme(theme: str) -> None:
    """Inject the selected theme's style sheet (no file and no config writes)."""
    st.html(theme_css(theme))
