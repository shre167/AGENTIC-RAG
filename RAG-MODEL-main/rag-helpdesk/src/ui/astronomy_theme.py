from __future__ import annotations

import html
from typing import Any

import streamlit as st


# ============================================================================
# SAFE HTML RENDERING
# ============================================================================
#
# Streamlit's markdown renderer treats lines indented by four or more
# spaces as code blocks, which is what previously caused raw <div> markup
# to appear literally in the UI. st.html() does not run the markdown
# parser at all, so it is the primary path. The markdown fallback strips
# leading indentation per line so that older Streamlit builds behave the
# same way.


def _flatten_markup(markup: str) -> str:
    """Remove per-line indentation without collapsing newlines."""

    lines = str(markup).splitlines()

    return "\n".join(line.lstrip() for line in lines)


def render_html(markup: str) -> None:
    """
    Render trusted theme markup safely in Streamlit.

    All dynamic values passed into the render_* helpers below are escaped
    with html.escape() before reaching this function.
    """

    if not markup or not str(markup).strip():
        return

    if hasattr(st, "html"):
        st.html(markup)
        return

    st.markdown(
        _flatten_markup(markup),
        unsafe_allow_html=True,
    )


def inject_global_css() -> str:
    """Return CSS for the professional deep-space astronomy theme."""

    return """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --space-bg: #070b14;
    --space-bg-alt: #0c1220;
    --space-panel: rgba(14, 20, 36, 0.82);
    --space-border: rgba(120, 150, 210, 0.18);

    --space-text: #e8edf7;
    --space-muted: #94a3b8;

    --space-accent: #6b9fd4;
    --space-cyan: #5ec4d4;

    --space-success: #4ade80;
    --space-warning: #fbbf24;
}


/* =========================================================
   GLOBAL
   ========================================================= */

html,
body,
[class*="css"] {
    font-family:
        "Inter",
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.stApp {
    background:
        radial-gradient(
            ellipse 120% 80% at 50% -20%,
            rgba(45, 70, 130, 0.22) 0%,
            transparent 55%
        ),
        radial-gradient(
            circle at 85% 15%,
            rgba(90, 60, 140, 0.12) 0%,
            transparent 35%
        ),
        radial-gradient(
            circle at 10% 80%,
            rgba(40, 90, 130, 0.10) 0%,
            transparent 40%
        ),
        linear-gradient(
            180deg,
            #070b14 0%,
            #0a0f1a 50%,
            #070b14 100%
        );

    color: var(--space-text);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;

    pointer-events: none;
    z-index: 0;

    background-image:
        radial-gradient(
            1px 1px at 20px 30px,
            rgba(255,255,255,0.35),
            transparent
        ),
        radial-gradient(
            1px 1px at 80px 120px,
            rgba(255,255,255,0.25),
            transparent
        ),
        radial-gradient(
            1px 1px at 160px 60px,
            rgba(255,255,255,0.20),
            transparent
        ),
        radial-gradient(
            1px 1px at 240px 180px,
            rgba(255,255,255,0.30),
            transparent
        ),
        radial-gradient(
            1px 1px at 320px 40px,
            rgba(255,255,255,0.18),
            transparent
        );

    background-size: 650px 250px;
    opacity: 0.45;

    animation: starDrift 120s linear infinite;
}

@keyframes starDrift {
    from {
        transform: translateY(0);
    }

    to {
        transform: translateY(-250px);
    }
}

.block-container {
    max-width: 1240px !important;

    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}

/* Hide only Streamlit's own chrome: the footer, the colored top
   decoration bar, and the "running" status widget. #MainMenu and the
   Deploy button are hidden individually rather than hiding the whole
   toolbar, because the sidebar's reopen button (stExpandSidebarButton)
   lives inside that same toolbar container when the sidebar is
   collapsed — hiding the toolbar outright hides the reopen button too,
   leaving no way to bring the sidebar back. */

#MainMenu,
[data-testid="stAppDeployButton"],
footer,
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            rgba(8, 12, 24, 0.98),
            rgba(10, 16, 30, 0.96)
        );

    border-right:
        1px solid var(--space-border);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.25rem !important;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.35rem 0 1.5rem;
}

.sidebar-brand-icon {
    width: 38px;
    height: 38px;

    border-radius: 11px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: rgba(107,159,212,0.12);

    border:
        1px solid rgba(107,159,212,0.25);

    color: #8dc5ff;

    font-size: 1.1rem;
}

.sidebar-brand-title {
    font-size: 0.82rem;
    letter-spacing: 0.18em;
    font-weight: 700;
    color: #edf4ff;
}

.sidebar-brand-subtitle {
    margin-top: 0.2rem;
    font-size: 0.55rem;
    letter-spacing: 0.12em;
    color: #64748b;
}

.sidebar-section-label {
    font-size: 0.58rem;
    letter-spacing: 0.16em;
    color: #64748b;
    font-weight: 700;
    margin-bottom: 0.7rem;
}

.sidebar-core,
.sidebar-system {
    background: rgba(255,255,255,0.025);

    border:
        1px solid rgba(120,150,210,0.14);

    border-radius: 13px;

    padding: 0.85rem;

    margin-bottom: 1rem;
}

.sidebar-stat-grid {
    display: grid;

    grid-template-columns: 1fr 1fr;

    gap: 0.5rem;

    margin-bottom: 0.8rem;
}

.sidebar-stat {
    background:
        rgba(255,255,255,0.035);

    border-radius: 9px;

    padding: 0.65rem;
}

.sidebar-stat-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #edf4ff;
}

.sidebar-stat-label {
    margin-top: 0.15rem;
    font-size: 0.52rem;
    letter-spacing: 0.1em;
    color: #64748b;
}

.sidebar-status-row,
.sidebar-system-row {
    display: flex;

    align-items: center;

    justify-content: space-between;

    padding: 0.4rem 0;

    color: #94a3b8;

    font-size: 0.72rem;
}

.status-active {
    color: #4ade80;
}

.status-muted {
    color: #64748b;

    font-size: 0.6rem;

    letter-spacing: 0.08em;
}

.sidebar-system-row strong {
    color: #cbd5e1;

    font-size: 0.63rem;

    font-weight: 500;

    max-width: 130px;

    overflow: hidden;

    text-overflow: ellipsis;

    white-space: nowrap;
}


/* =========================================================
   PAGE HEADER
   ========================================================= */

.page-header {
    padding: 2.5rem 0 1.6rem;
}

.page-header.compact {
    padding-top: 1rem;
}

.page-eyebrow {
    color: var(--space-accent);

    font-size: 0.62rem;

    letter-spacing: 0.2em;

    font-weight: 700;

    margin-bottom: 0.7rem;
}

.page-header h1 {
    font-size: 2.35rem;

    line-height: 1.05;

    letter-spacing: -0.045em;

    font-weight: 650;

    margin: 0;

    color: #edf4ff;
}

.page-header p {
    max-width: 680px;

    color: #8291a8;

    font-size: 0.9rem;

    line-height: 1.6;

    margin-top: 0.75rem;
}

.page-divider {
    height: 1px;

    background:
        linear-gradient(
            90deg,
            rgba(107,159,212,0.32),
            rgba(107,159,212,0.05),
            transparent
        );

    margin:
        0.8rem
        0
        1.6rem;
}


/* =========================================================
   HERO STATUS
   ========================================================= */

.hero-status-row {
    display: grid;

    grid-template-columns: 1fr 1fr;

    gap: 0.8rem;

    margin:
        0.8rem
        0
        1.5rem;
}

.hero-status-item {
    background:
        rgba(255,255,255,0.025);

    border:
        1px solid rgba(120,150,210,0.14);

    border-radius: 13px;

    padding: 1rem 1.1rem;
}

.hero-status-label {
    color: #64748b;

    font-size: 0.55rem;

    letter-spacing: 0.14em;

    font-weight: 700;
}

.hero-status-value {
    margin-top: 0.45rem;

    color: #4ade80;

    font-size: 0.8rem;

    font-weight: 700;

    letter-spacing: 0.08em;
}

.hero-status-value.inactive {
    color: #64748b;
}


/* =========================================================
   RETRIEVAL MODE
   ========================================================= */

div[role="radiogroup"] {
    gap: 0.5rem !important;
    margin-top: 0.8rem;
}

div[role="radiogroup"] > label {
    background:
        rgba(255,255,255,0.025);

    border:
        1px solid rgba(120,150,210,0.16);

    border-radius: 10px !important;

    padding:
        0.65rem
        1.2rem !important;

    transition:
        all 0.18s ease;
}

div[role="radiogroup"] > label:hover {
    background:
        rgba(107,159,212,0.08);

    border-color:
        rgba(107,159,212,0.35);
}

div[role="radiogroup"] > label:has(input:checked) {
    background:
        rgba(107,159,212,0.15);

    border-color:
        rgba(107,159,212,0.55);

    box-shadow:
        0 0 18px rgba(107,159,212,0.08);
}

div[role="radiogroup"] > label p {
    font-size: 0.72rem !important;

    font-weight: 650 !important;

    letter-spacing: 0.05em;

    color: #dce7f7 !important;
}

.retrieval-mode-description {
    display: flex;

    align-items: center;

    gap: 0.5rem;

    margin-top: 0.65rem;

    color: #718096;

    font-size: 0.72rem;
}

.mode-live-dot {
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: var(--space-success);

    box-shadow:
        0 0 8px rgba(74,222,128,0.55);
}


/* =========================================================
   SUGGESTIONS
   ========================================================= */

.suggestion-grid {
    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 0.8rem;

    margin:
        1.4rem
        0
        2rem;
}

.suggestion-card {
    background:
        rgba(255,255,255,0.025);

    border:
        1px solid rgba(120,150,210,0.13);

    border-radius: 13px;

    padding: 1rem;

    transition:
        all 0.2s ease;
}

.suggestion-card:hover {
    transform:
        translateY(-3px);

    border-color:
        rgba(107,159,212,0.4);

    background:
        rgba(107,159,212,0.055);
}

.suggestion-icon {
    color: #7fb4e8;

    font-size: 1.15rem;

    margin-bottom: 0.7rem;
}

.suggestion-title {
    color: #dce7f7;

    font-size: 0.82rem;

    font-weight: 650;
}

.suggestion-text {
    color: #64748b;

    font-size: 0.68rem;

    line-height: 1.5;

    margin-top: 0.3rem;
}


/* =========================================================
   CHAT
   ========================================================= */

[data-testid="stChatMessage"] {
    background: transparent !important;

    border: none !important;

    padding: 0.4rem 0 !important;
}

[data-testid="stChatMessageContent"] {
    background:
        var(--space-panel) !important;

    border:
        1px solid var(--space-border);

    border-radius:
        12px !important;

    padding:
        1rem !important;

    color:
        var(--space-text) !important;
}


/* =========================================================
   CHAT INPUT
   ========================================================= */

[data-testid="stChatInput"] {
    background:
        rgba(14,20,36,0.95);

    border:
        1px solid rgba(120,150,210,0.22);

    border-radius:
        14px;

    box-shadow:
        0 12px 35px rgba(0,0,0,0.18);
}


/* =========================================================
   SOURCES
   ========================================================= */

.sources-card {
    margin-top: 0.9rem;

    background:
        rgba(255,255,255,0.02);

    border:
        1px solid var(--space-border);

    border-radius: 10px;

    padding:
        0.85rem
        1rem;
}

.sources-title {
    font-size: 0.68rem;

    letter-spacing: 0.14em;

    text-transform: uppercase;

    color: var(--space-accent);

    margin-bottom: 0.55rem;

    font-weight: 600;
}

.source-item {
    display: flex;

    align-items: center;

    gap: 0.5rem;

    color: var(--space-muted);

    font-size: 0.82rem;

    padding: 0.25rem 0;
}

.source-dot {
    color: var(--space-cyan);

    font-size: 0.55rem;
}


/* =========================================================
   CONTEXT
   ========================================================= */

.context-chunk {
    background:
        rgba(255,255,255,0.02);

    border-left:
        2px solid var(--space-accent);

    padding:
        0.75rem
        0.9rem;

    margin:
        0.6rem
        0;

    border-radius:
        0 8px 8px 0;
}

.context-chunk-title {
    font-size: 0.82rem;

    font-weight: 600;

    color: #dce7f7;

    margin-bottom: 0.4rem;
}

.context-chunk-text {
    font-size: 0.8rem;

    color: var(--space-muted);

    line-height: 1.6;

    white-space: pre-wrap;
}

.context-chunk-score {
    font-size: 0.67rem;

    color: #64748b;

    margin-top: 0.5rem;
}


/* =========================================================
   RETRIEVAL LAB
   ========================================================= */

.trace-query-card {
    background:
        linear-gradient(
            135deg,
            rgba(107,159,212,0.09),
            rgba(255,255,255,0.02)
        );

    border:
        1px solid rgba(107,159,212,0.22);

    border-radius: 15px;

    padding:
        1.15rem
        1.25rem;

    margin:
        1rem
        0
        1.4rem;
}

.trace-query-label {
    color: #64748b;

    font-size: 0.55rem;

    letter-spacing: 0.16em;

    font-weight: 700;
}

.trace-query {
    color: #edf4ff;

    font-size: 1rem;

    font-weight: 550;

    margin-top: 0.5rem;

    line-height: 1.5;
}

.lab-section-title {
    margin:
        1.8rem
        0
        0.7rem;

    color: #b9c8dc;

    font-size: 0.68rem;

    letter-spacing: 0.14em;

    text-transform: uppercase;

    font-weight: 700;
}

.pipeline-node {
    min-height: 105px;

    border-radius: 13px;

    padding: 0.9rem;

    background:
        rgba(255,255,255,0.025);

    border:
        1px solid rgba(120,150,210,0.12);
}

.pipeline-node.active {
    border-color:
        rgba(107,159,212,0.35);

    background:
        rgba(107,159,212,0.055);
}

.pipeline-node.inactive {
    opacity: 0.45;
}

.pipeline-node-icon {
    color: #7fb4e8;

    font-size: 1.1rem;
}

.pipeline-node-title {
    color: #dce7f7;

    font-size: 0.72rem;

    font-weight: 650;

    margin-top: 0.5rem;
}

.pipeline-node-status {
    color: #64748b;

    font-size: 0.52rem;

    letter-spacing: 0.1em;

    margin-top: 0.35rem;
}


/* =========================================================
   FORMULA
   ========================================================= */

.formula-card {
    background:
        rgba(107,159,212,0.05);

    border:
        1px solid rgba(107,159,212,0.2);

    border-radius: 13px;

    padding:
        1rem
        1.15rem;

    margin-bottom: 0.8rem;
}

.formula-title {
    color: #8dc5ff;

    font-size: 0.62rem;

    letter-spacing: 0.12em;

    text-transform: uppercase;

    font-weight: 700;
}

.formula {
    color: #edf4ff;

    font-family:
        Consolas,
        monospace;

    font-size: 0.9rem;

    margin-top: 0.55rem;
}

.formula-note {
    color: #64748b;

    font-size: 0.66rem;

    margin-top: 0.45rem;
}


/* =========================================================
   EVIDENCE
   ========================================================= */

.evidence-card {
    background:
        rgba(255,255,255,0.025);

    border:
        1px solid rgba(120,150,210,0.14);

    border-radius: 12px;

    padding: 0.9rem;
}

.evidence-label {
    color: #64748b;

    font-size: 0.52rem;

    letter-spacing: 0.12em;

    font-weight: 700;
}

.evidence-value {
    color: #8dc5ff;

    font-size: 1.05rem;

    font-weight: 650;

    margin-top: 0.35rem;
}


/* =========================================================
   INGESTION
   ========================================================= */

.architecture-node {
    text-align: center;

    background:
        rgba(255,255,255,0.025);

    border:
        1px solid rgba(120,150,210,0.14);

    border-radius: 13px;

    padding:
        1rem
        0.5rem;

    min-height: 135px;
}

.architecture-number {
    color: #4f6380;

    font-size: 0.5rem;

    letter-spacing: 0.1em;
}

.architecture-icon {
    color: #7fb4e8;

    font-size: 1.3rem;

    margin: 0.5rem 0;
}

.architecture-name {
    color: #dce7f7;

    font-size: 0.65rem;

    font-weight: 700;

    letter-spacing: 0.06em;
}

.architecture-detail {
    color: #64748b;

    font-size: 0.55rem;

    margin-top: 0.3rem;
}

.component-row {
    display: grid;

    grid-template-columns:
        1.25fr
        2.2fr
        0.7fr;

    gap: 1rem;

    align-items: center;

    padding:
        0.8rem
        0.95rem;

    margin-bottom: 0.4rem;

    border:
        1px solid rgba(120,150,210,0.1);

    border-radius: 10px;

    background:
        rgba(255,255,255,0.02);
}

.component-name {
    color: #dce7f7;

    font-size: 0.72rem;

    font-weight: 600;
}

.component-description {
    color: #64748b;

    font-size: 0.65rem;
}

.component-active,
.component-muted {
    text-align: right;

    font-size: 0.55rem;

    letter-spacing: 0.1em;

    font-weight: 700;
}

.component-active {
    color: var(--space-success);
}

.component-muted {
    color: #64748b;
}

.ingestion-file-preview {
    display: flex;

    align-items: center;

    gap: 0.8rem;

    background:
        rgba(107,159,212,0.05);

    border:
        1px solid rgba(107,159,212,0.2);

    border-radius: 12px;

    padding: 0.9rem;

    margin-bottom: 0.8rem;
}

.ingestion-file-icon {
    color: #7fb4e8;

    font-size: 1.3rem;
}

.ingestion-file-name {
    color: #dce7f7;

    font-size: 0.75rem;

    font-weight: 600;
}

.ingestion-file-meta {
    color: #64748b;

    font-size: 0.6rem;

    margin-top: 0.2rem;
}


/* =========================================================
   CHUNK EXPLORER
   ========================================================= */

.chunk-card {
    background:
        rgba(255,255,255,0.025);

    border:
        1px solid rgba(120,150,210,0.14);

    border-radius: 12px;

    padding: 0.9rem 1rem;

    margin-bottom: 0.65rem;
}

.chunk-header {
    display: flex;

    justify-content: space-between;

    align-items: center;

    gap: 1rem;

    margin-bottom: 0.55rem;
}

.chunk-source {
    color: #dce7f7;

    font-size: 0.72rem;

    font-weight: 650;
}

.chunk-id {
    color: #64748b;

    font-size: 0.58rem;

    font-family: Consolas, monospace;
}

.chunk-text {
    color: #94a3b8;

    font-size: 0.72rem;

    line-height: 1.65;

    white-space: pre-wrap;
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    background:
        rgba(255,255,255,0.02);

    border:
        1px dashed rgba(107,159,212,0.30);

    border-radius: 12px;

    padding: 0.6rem;
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
}


/* =========================================================
   METRICS
   ========================================================= */

[data-testid="stMetric"] {
    background:
        rgba(255,255,255,0.025);

    border:
        1px solid rgba(120,150,210,0.12);

    border-radius: 12px;

    padding: 0.85rem;
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;

    font-size: 0.6rem !important;
}

[data-testid="stMetricValue"] {
    color: #e8edf7 !important;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    border:
        1px solid rgba(107,159,212,0.22);

    border-radius: 9px;

    background:
        rgba(107,159,212,0.07);

    color: #cddbf0;

    font-weight: 600;

    transition:
        all 0.18s ease;
}

.stButton > button:hover {
    border-color:
        rgba(107,159,212,0.5);

    background:
        rgba(107,159,212,0.13);

    color: #edf4ff;
}


/* =========================================================
   DATAFRAMES / TABLES
   ========================================================= */

[data-testid="stDataFrame"] {
    border:
        1px solid rgba(120,150,210,0.14);

    border-radius: 10px;

    overflow: hidden;
}


/* =========================================================
   ALERTS
   ========================================================= */

[data-testid="stAlert"] {
    border-radius: 10px;
}


/* =========================================================
   RESPONSIVE
   ========================================================= */

@media (max-width: 900px) {

    .suggestion-grid {
        grid-template-columns: 1fr;
    }

    .hero-status-row {
        grid-template-columns: 1fr;
    }

    .component-row {
        grid-template-columns: 1fr;

        gap: 0.4rem;
    }

    .component-active,
    .component-muted {
        text-align: left;
    }

    .architecture-node {
        margin-bottom: 0.6rem;
    }
}

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    .page-header h1 {
        font-size: 1.8rem;
    }

    div[role="radiogroup"] {
        flex-wrap: wrap;
    }
}


/* =========================================================
   MISSION CONTROL — SEGMENTED RETRIEVAL CONTROL
   =========================================================
   Styles st.segmented_control (and the pills fallback) into a
   proper segmented control: no radio circles, no red dot. */

[data-testid="stSegmentedControl"] [role="radiogroup"],
[data-testid="stButtonGroup"] [role="radiogroup"],
[data-testid="stSegmentedControl"] [role="group"],
[data-testid="stButtonGroup"] [role="group"] {
    display: flex !important;

    gap: 0 !important;

    width: 100%;

    background: rgba(8, 13, 26, 0.9);

    border: 1px solid var(--space-border);

    border-radius: 12px;

    padding: 4px;

    overflow: hidden;
}

[data-testid="stSegmentedControl"] button,
[data-testid="stButtonGroup"] button {
    flex: 1 1 0 !important;

    border: 1px solid transparent !important;

    border-radius: 9px !important;

    background: transparent !important;

    color: var(--space-muted) !important;

    font-family: 'Space Grotesk', 'Inter', sans-serif !important;

    font-size: 0.74rem !important;

    font-weight: 600 !important;

    letter-spacing: 0.1em !important;

    text-transform: uppercase;

    padding: 0.62rem 0.5rem !important;

    box-shadow: none !important;

    transition:
        background 0.18s ease,
        color 0.18s ease,
        border-color 0.18s ease;
}

[data-testid="stSegmentedControl"] button:hover,
[data-testid="stButtonGroup"] button:hover {
    color: var(--space-text) !important;

    background: rgba(107, 159, 212, 0.07) !important;
}

[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[aria-pressed="true"],
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"],
[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"],
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
[data-testid="stButtonGroup"] button[kind="pillsActive"] {
    background:
        linear-gradient(
            180deg,
            rgba(94, 196, 212, 0.16),
            rgba(107, 159, 212, 0.1)
        ) !important;

    color: #eaf4ff !important;

    border-color: rgba(94, 196, 212, 0.45) !important;

    box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.05) inset,
        0 2px 10px rgba(6, 12, 24, 0.55) !important;
}

/* Hide any residual radio circle if the fallback path is used. */

[data-testid="stRadio"] [role="radiogroup"] {
    display: flex;

    gap: 0.4rem;
}

[data-testid="stRadio"] label > div:first-child {
    display: none !important;
}

.control-label {
    color: #7c8ba3;

    font-family: 'Space Grotesk', 'Inter', sans-serif;

    font-size: 0.62rem;

    font-weight: 700;

    letter-spacing: 0.22em;

    text-transform: uppercase;

    margin: 0.2rem 0 0.45rem;
}

.mode-description {
    color: var(--space-muted);

    font-size: 0.72rem;

    margin: 0.5rem 0 0.2rem;

    padding-left: 0.1rem;
}


/* =========================================================
   SIDEBAR NAVIGATION
   ========================================================= */

.nav-group-label {
    color: #5c6a82;

    font-family: 'Space Grotesk', 'Inter', sans-serif;

    font-size: 0.56rem;

    font-weight: 700;

    letter-spacing: 0.2em;

    text-transform: uppercase;

    margin: 0.85rem 0 0.3rem 0.1rem;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;

    justify-content: flex-start !important;

    text-align: left !important;

    background: transparent !important;

    border: 1px solid transparent !important;

    border-radius: 9px !important;

    color: var(--space-muted) !important;

    font-size: 0.78rem !important;

    font-weight: 500 !important;

    padding: 0.44rem 0.7rem !important;

    box-shadow: none !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(107, 159, 212, 0.07) !important;

    color: var(--space-text) !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(94, 196, 212, 0.1) !important;

    border-color: rgba(94, 196, 212, 0.35) !important;

    color: #eaf4ff !important;

    font-weight: 600 !important;
}

.sidebar-divider {
    height: 1px;

    margin: 0.9rem 0;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(120, 150, 210, 0.22),
            transparent
        );
}


/* =========================================================
   MISSION CONTROL — STATUS CARDS
   ========================================================= */

.status-card {
    background: var(--space-panel);

    border: 1px solid var(--space-border);

    border-radius: 12px;

    padding: 0.85rem 0.95rem;

    height: 100%;

    position: relative;

    overflow: hidden;
}

.status-card::before {
    content: "";

    position: absolute;

    inset: 0 0 auto 0;

    height: 1px;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(94, 196, 212, 0.35),
            transparent
        );
}

.status-card-label {
    color: #6b7a92;

    font-family: 'Space Grotesk', 'Inter', sans-serif;

    font-size: 0.58rem;

    font-weight: 700;

    letter-spacing: 0.18em;

    text-transform: uppercase;
}

.status-card-value {
    color: var(--space-text);

    font-family: 'IBM Plex Mono', monospace;

    font-size: 1.45rem;

    font-weight: 600;

    line-height: 1.5;

    margin-top: 0.3rem;

    word-break: break-word;
}

.status-card-value.small {
    font-size: 0.95rem;

    line-height: 1.7;
}

.status-card-meta {
    color: #64748b;

    font-size: 0.62rem;

    letter-spacing: 0.06em;

    text-transform: uppercase;

    margin-top: 0.18rem;
}

.status-dot {
    display: inline-block;

    width: 7px;

    height: 7px;

    border-radius: 50%;

    margin-right: 0.45rem;

    vertical-align: middle;
}

.status-dot.online {
    background: var(--space-success);

    box-shadow: 0 0 6px rgba(74, 222, 128, 0.5);
}

.status-dot.offline {
    background: #64748b;
}

.status-dot.unknown {
    background: var(--space-warning);
}


/* =========================================================
   SYSTEM ARCHITECTURE DIAGRAM
   ========================================================= */

.arch-plane {
    border: 1px solid var(--space-border);

    border-radius: 14px;

    padding: 1rem 1.1rem 1.15rem;

    margin-bottom: 0.6rem;

    background:
        radial-gradient(
            120% 140% at 50% 0%,
            rgba(94, 196, 212, 0.05),
            transparent 60%
        ),
        rgba(10, 15, 28, 0.6);
}

.arch-plane-title {
    color: #9dc2e8;

    font-family: 'Space Grotesk', 'Inter', sans-serif;

    font-size: 0.66rem;

    font-weight: 700;

    letter-spacing: 0.2em;

    text-transform: uppercase;
}

.arch-plane-subtitle {
    color: #64748b;

    font-size: 0.68rem;

    margin: 0.2rem 0 0.35rem;
}

.arch-node {
    border: 1px solid var(--space-border);

    border-left: 2px solid var(--space-accent);

    border-radius: 10px;

    background: rgba(255, 255, 255, 0.022);

    padding: 0.6rem 0.7rem;

    height: 100%;
}

.arch-node.storage {
    border-left-color: #7c8cf8;
}

.arch-node.retrieval {
    border-left-color: var(--space-cyan);
}

.arch-node.processing {
    border-left-color: var(--space-accent);
}

.arch-node.decision {
    border-left-color: var(--space-warning);
}

.arch-node.generation {
    border-left-color: var(--space-success);
}

.arch-node-icon {
    font-size: 0.9rem;

    line-height: 1.4;
}

.arch-node-name {
    color: var(--space-text);

    font-family: 'Space Grotesk', 'Inter', sans-serif;

    font-size: 0.72rem;

    font-weight: 650;

    letter-spacing: 0.04em;

    margin-top: 0.15rem;
}

.arch-node-tech {
    color: #64748b;

    font-family: 'IBM Plex Mono', monospace;

    font-size: 0.58rem;

    letter-spacing: 0.04em;

    margin-top: 0.12rem;
}

.arch-arrow {
    color: #3f5372;

    font-size: 0.85rem;

    text-align: center;

    line-height: 1.1;

    padding: 0.18rem 0;
}

.arch-connector {
    height: 22px;

    margin: 0.1rem 0;

    background:
        linear-gradient(
            180deg,
            rgba(94, 196, 212, 0.45),
            rgba(94, 196, 212, 0.05)
        );

    width: 1px;

    margin-left: auto;

    margin-right: auto;
}

.arch-legend {
    display: flex;

    flex-wrap: wrap;

    gap: 1.1rem;

    padding: 0.7rem 0.9rem;

    border: 1px solid var(--space-border);

    border-radius: 10px;

    background: rgba(255, 255, 255, 0.015);
}

.arch-legend-item {
    display: flex;

    align-items: center;

    gap: 0.45rem;

    color: var(--space-muted);

    font-size: 0.66rem;
}

.arch-legend-swatch {
    width: 9px;

    height: 9px;

    border-radius: 2px;
}

@media (max-width: 820px) {

    .status-card-value {
        font-size: 1.15rem;
    }

    .arch-legend {
        gap: 0.6rem;
    }
}


/* =========================================================
   OBSERVABILITY ADDITIONS
   ========================================================= */

.kv-row {
    display: flex;

    justify-content: space-between;

    align-items: baseline;

    gap: 1rem;

    padding: 0.42rem 0;

    border-bottom:
        1px solid rgba(120,150,210,0.08);
}

.kv-row:last-child {
    border-bottom: none;
}

.kv-key {
    color: #7c8ba3;

    font-size: 0.7rem;
}

.kv-value {
    color: #dce7f7;

    font-size: 0.78rem;

    font-weight: 550;

    text-align: right;

    word-break: break-word;
}

.kv-value.muted {
    color: #64748b;

    font-weight: 400;
}

.evidence-badge {
    display: inline-flex;

    align-items: center;

    gap: 0.5rem;

    border-radius: 999px;

    padding: 0.3rem 0.85rem;

    font-size: 0.7rem;

    font-weight: 600;

    letter-spacing: 0.03em;

    border: 1px solid transparent;
}

.evidence-badge.strong {
    color: #86efac;

    background: rgba(74,222,128,0.1);

    border-color: rgba(74,222,128,0.28);
}

.evidence-badge.moderate {
    color: #93c5fd;

    background: rgba(107,159,212,0.1);

    border-color: rgba(107,159,212,0.3);
}

.evidence-badge.weak {
    color: #fcd34d;

    background: rgba(251,191,36,0.09);

    border-color: rgba(251,191,36,0.28);
}

.evidence-badge.none {
    color: #94a3b8;

    background: rgba(148,163,184,0.08);

    border-color: rgba(148,163,184,0.22);
}

.mode-chip {
    display: inline-flex;

    align-items: center;

    gap: 0.45rem;

    color: #8dc5ff;

    background: rgba(107,159,212,0.09);

    border: 1px solid rgba(107,159,212,0.26);

    border-radius: 999px;

    padding: 0.28rem 0.8rem;

    font-size: 0.68rem;

    font-weight: 600;
}

.sim-tag {
    display: inline-block;

    color: #8f9bb3;

    background: rgba(148,163,184,0.08);

    border: 1px dashed rgba(148,163,184,0.3);

    border-radius: 6px;

    padding: 0.16rem 0.55rem;

    font-size: 0.6rem;

    letter-spacing: 0.08em;
}

.stage-rail {
    display: flex;

    flex-direction: column;

    gap: 0.3rem;
}

.stage-item {
    display: flex;

    align-items: center;

    gap: 0.7rem;

    padding: 0.5rem 0.75rem;

    border-radius: 10px;

    border: 1px solid rgba(120,150,210,0.1);

    background: rgba(255,255,255,0.02);
}

.stage-item.done {
    border-color: rgba(74,222,128,0.25);

    background: rgba(74,222,128,0.045);
}

.stage-item.pending {
    opacity: 0.45;
}

.stage-marker {
    color: #7fb4e8;

    font-size: 0.85rem;

    width: 1.1rem;

    text-align: center;
}

.stage-item.done .stage-marker {
    color: var(--space-success);
}

.stage-label {
    color: #dce7f7;

    font-size: 0.74rem;

    font-weight: 550;
}

.stage-detail {
    color: #64748b;

    font-size: 0.66rem;

    margin-left: auto;

    text-align: right;
}

.stage-arrow {
    color: #3d4c63;

    font-size: 0.7rem;

    text-align: center;

    line-height: 1;
}

.chunk-map {
    display: flex;

    flex-wrap: wrap;

    gap: 4px;

    margin: 0.5rem 0 0.2rem;
}

.chunk-cell {
    flex: 1 1 34px;

    min-width: 34px;

    height: 26px;

    border-radius: 5px;

    background: rgba(107,159,212,0.13);

    border: 1px solid rgba(107,159,212,0.2);

    color: #9dc2e8;

    font-size: 0.56rem;

    display: flex;

    align-items: center;

    justify-content: center;
}

.chunk-cell.selected {
    background: rgba(107,159,212,0.42);

    border-color: rgba(141,197,255,0.7);

    color: #f2f7ff;

    font-weight: 700;
}

.overlap-legend {
    display: flex;

    flex-wrap: wrap;

    gap: 1rem;

    color: #64748b;

    font-size: 0.66rem;

    margin-top: 0.5rem;
}

@media (max-width: 820px) {

    .kv-row {
        flex-direction: column;

        gap: 0.15rem;
    }

    .kv-value {
        text-align: left;
    }

    .stage-detail {
        margin-left: 0;
    }
}

</style>
"""


def render_landing_page(
    *,
    index_ready: bool,
    rag_ready: bool,
) -> str:

    kb_status = "ACTIVE" if index_ready else "BUILD REQUIRED"
    rag_status = "ACTIVE" if rag_ready else "STANDBY"

    kb_class = "" if index_ready else " inactive"
    rag_class = "" if rag_ready else " inactive"

    return f"""
<div class="page-header">

    <div class="page-eyebrow">
        ASTRONOMY KNOWLEDGE SYSTEM
    </div>

    <h1>Ask the universe.</h1>

    <p>
        Explore astronomy through a transparent
        retrieval-augmented knowledge system.
    </p>

</div>

<div class="page-divider"></div>

<div class="hero-status-row">

    <div class="hero-status-item">

        <div class="hero-status-label">
            KNOWLEDGE BASE
        </div>

        <div class="hero-status-value{kb_class}">
            {kb_status}
        </div>

    </div>

    <div class="hero-status-item">

        <div class="hero-status-label">
            RAG ENGINE
        </div>

        <div class="hero-status-value{rag_class}">
            {rag_status}
        </div>

    </div>

</div>
"""


def render_sources_card(
    sources: list[str],
    num_retrieved: int | None = None,
) -> str:

    if not sources:
        return ""

    items = "".join(
        f"""
<div class="source-item">
    <span class="source-dot">●</span>
    {html.escape(str(src))}
</div>
"""
        for src in sources
    )

    meta = ""

    if num_retrieved is not None:
        meta = f"""
<div style="
    margin-top:0.7rem;
    color:#64748b;
    font-size:0.68rem;
">
    Retrieved chunks: {html.escape(str(num_retrieved))}
</div>
"""

    return f"""
<div class="sources-card">

    <div class="sources-title">
        Sources
    </div>

    {items}

    {meta}

</div>
"""


def render_context_chunks(
    chunks: list[dict[str, Any]],
) -> str:

    parts: list[str] = []

    for chunk in chunks:

        filename = html.escape(
            str(chunk.get("filename", "Unknown source"))
        )

        text = html.escape(
            str(chunk.get("text", ""))
        )

        heading = chunk.get("section_heading")

        heading_html = ""

        if heading:
            heading_html = (
                f" — {html.escape(str(heading))}"
            )

        score_html = ""

        score = chunk.get("score")

        if score is not None:

            try:
                score_float = float(score)

                score_html = f"""
<div class="context-chunk-score">
    Relevance score: {score_float:.4f}
</div>
"""

            except (TypeError, ValueError):
                pass

        parts.append(
            f"""
<div class="context-chunk">

    <div class="context-chunk-title">
        {filename}{heading_html}
    </div>

    <div class="context-chunk-text">
        {text}
    </div>

    {score_html}

</div>
"""
        )

    return "".join(parts)


def render_user_message(
    content: str,
) -> str:

    return f"""
<div class="context-chunk">

    <div class="context-chunk-text">
        {html.escape(content)}
    </div>

</div>
"""


def render_assistant_message(
    content: str,
) -> str:

    # IMPORTANT:
    # Assistant content is escaped so accidental HTML/Markdown-like
    # content cannot break the surrounding UI.
    safe_content = html.escape(str(content)).replace("\n", "<br>")

    return f"""
<div class="context-chunk">

    <div class="context-chunk-text">
        {safe_content}
    </div>

</div>
"""


def render_panel_card(
    title: str,
    inner_html: str,
) -> str:

    return f"""
<div class="evidence-card">

    <div class="evidence-label">
        {html.escape(title)}
    </div>

    <div style="margin-top:0.7rem;">
        {inner_html}
    </div>

</div>
"""


def render_stat_grid(
    items: list[tuple[str, str]],
) -> str:

    cells = "".join(
        f"""
<div class="sidebar-stat">

    <div class="sidebar-stat-value">
        {html.escape(str(value))}
    </div>

    <div class="sidebar-stat-label">
        {html.escape(str(label))}
    </div>

</div>
"""
        for value, label in items
    )

    return f"""
<div class="sidebar-stat-grid">
    {cells}
</div>
"""


def render_status_row(
    label: str,
    active: bool,
    *,
    warning: bool = False,
) -> str:

    if warning:
        status = "● WARNING"

    elif active:
        status = "● ACTIVE"

    else:
        status = "○ OFFLINE"

    status_class = (
        "status-active"
        if active
        else "status-muted"
    )

    return f"""
<div class="sidebar-status-row">

    <span>
        {html.escape(label)}
    </span>

    <span class="{status_class}">
        {status}
    </span>

</div>
"""

# ============================================================================
# OBSERVABILITY COMPONENTS
# ============================================================================
#
# Every one of these helpers escapes its dynamic input. They return markup
# strings that are intended to be passed to render_html().


def render_page_header(
    eyebrow: str,
    title: str,
    subtitle: str = "",
) -> str:
    """Header block used at the top of each page."""

    subtitle_html = ""

    if subtitle:
        subtitle_html = (
            f"<p>{html.escape(str(subtitle))}</p>"
        )

    return f"""
<div class="page-header">

    <div class="page-eyebrow">
        {html.escape(str(eyebrow))}
    </div>

    <h1>{html.escape(str(title))}</h1>

    {subtitle_html}

</div>

<div class="page-divider"></div>
"""


def render_section_title(
    text: str,
) -> str:
    """Small uppercase section heading used inside the lab pages."""

    return f"""
<div class="lab-section-title">
    {html.escape(str(text))}
</div>
"""


def render_kv_rows(
    items: list[tuple[str, Any]],
) -> str:
    """
    Render aligned key/value rows.

    A value of None renders as a muted "N/A" so that missing runtime data
    is visibly missing rather than silently filled in.
    """

    rows: list[str] = []

    for key, value in items:

        if value is None or value == "":
            display = "N/A"
            value_class = "kv-value muted"

        else:
            display = str(value)
            value_class = "kv-value"

        rows.append(
            f"""
<div class="kv-row">

    <span class="kv-key">
        {html.escape(str(key))}
    </span>

    <span class="{value_class}">
        {html.escape(display)}
    </span>

</div>
"""
        )

    return "".join(rows)


def render_evidence_badge(
    level_label: str,
    level_key: str = "none",
) -> str:
    """Render the evidence level as a coloured pill."""

    key = str(level_key).strip().lower()

    tone = {
        "strong": "strong",
        "high": "strong",
        "moderate": "moderate",
        "medium": "moderate",
        "weak": "weak",
        "low": "weak",
    }.get(key, "none")

    return f"""
<span class="evidence-badge {tone}">
    Evidence: {html.escape(str(level_label))}
</span>
"""


def render_mode_chip(
    label: str,
) -> str:
    """Render the active retrieval mode as a chip."""

    return f"""
<span class="mode-chip">
    <span class="mode-live-dot"></span>
    {html.escape(str(label))}
</span>
"""


def render_simulation_tag(
    text: str = "Pipeline visualization",
) -> str:
    """
    Mark a block as a conceptual visualization.

    Used so that educational diagrams are never mistaken for live
    backend telemetry.
    """

    return f"""
<span class="sim-tag">
    {html.escape(str(text))}
</span>
"""


def render_trace_query_card(
    label: str,
    text: str,
) -> str:
    """Highlight the query a trace belongs to."""

    return f"""
<div class="trace-query-card">

    <div class="trace-query-label">
        {html.escape(str(label))}
    </div>

    <div class="trace-query">
        {html.escape(str(text))}
    </div>

</div>
"""


def render_pipeline_node(
    icon: str,
    title: str,
    status: str,
    state: str = "active",
) -> str:
    """
    Render one ingestion pipeline node.

    state: "active", "inactive" or "" (neutral).
    """

    state_class = ""

    if state in {"active", "inactive"}:
        state_class = f" {state}"

    return f"""
<div class="pipeline-node{state_class}">

    <div class="pipeline-node-icon">
        {html.escape(str(icon))}
    </div>

    <div class="pipeline-node-title">
        {html.escape(str(title))}
    </div>

    <div class="pipeline-node-status">
        {html.escape(str(status))}
    </div>

</div>
"""


def render_architecture_node(
    number: str,
    icon: str,
    name: str,
    detail: str = "",
) -> str:
    """Render one node of an architecture diagram."""

    return f"""
<div class="architecture-node">

    <div class="architecture-number">
        {html.escape(str(number))}
    </div>

    <div class="architecture-icon">
        {html.escape(str(icon))}
    </div>

    <div class="architecture-name">
        {html.escape(str(name))}
    </div>

    <div class="architecture-detail">
        {html.escape(str(detail))}
    </div>

</div>
"""


def render_stage_rail(
    stages: list[tuple[str, str, Any]],
) -> str:
    """
    Render a vertical stage rail.

    Each stage is (state, label, detail) where state is one of
    "done", "pending" or "".
    """

    parts: list[str] = []

    for index, (state, label, detail) in enumerate(stages):

        if index:
            parts.append(
                '<div class="stage-arrow">↓</div>'
            )

        state = str(state or "").strip().lower()

        state_class = ""

        if state in {"done", "pending"}:
            state_class = f" {state}"

        marker = "✓" if state == "done" else "●"

        detail_html = ""

        if detail is not None and str(detail) != "":
            detail_html = (
                f'<span class="stage-detail">'
                f"{html.escape(str(detail))}"
                f"</span>"
            )

        parts.append(
            f"""
<div class="stage-item{state_class}">

    <span class="stage-marker">
        {marker}
    </span>

    <span class="stage-label">
        {html.escape(str(label))}
    </span>

    {detail_html}

</div>
"""
        )

    return (
        '<div class="stage-rail">'
        + "".join(parts)
        + "</div>"
    )


def render_formula_card(
    title: str,
    formula: str,
    note: str = "",
) -> str:
    """Render a formula with an optional explanatory note."""

    note_html = ""

    if note:
        note_html = f"""
<div class="formula-note">
    {html.escape(str(note))}
</div>
"""

    return f"""
<div class="formula-card">

    <div class="formula-title">
        {html.escape(str(title))}
    </div>

    <div class="formula">
        {html.escape(str(formula))}
    </div>

    {note_html}

</div>
"""


def render_chunk_map(
    total: int,
    selected_index: int | None = None,
    max_cells: int = 120,
) -> str:
    """
    Render a compact map of a document's chunks.

    selected_index is zero-based. Only the first max_cells chunks are
    drawn so that very large documents do not flood the page.
    """

    try:
        total = int(total)
    except (TypeError, ValueError):
        return ""

    if total <= 0:
        return ""

    shown = min(total, max_cells)

    cells: list[str] = []

    for index in range(shown):

        selected = (
            selected_index is not None
            and index == selected_index
        )

        cell_class = (
            "chunk-cell selected"
            if selected
            else "chunk-cell"
        )

        cells.append(
            f'<div class="{cell_class}">{index}</div>'
        )

    overflow = ""

    if total > shown:
        overflow = (
            f'<div class="overlap-legend">'
            f"Showing the first {shown} of {total} chunks."
            f"</div>"
        )

    return (
        '<div class="chunk-map">'
        + "".join(cells)
        + "</div>"
        + overflow
    )


def render_chunk_card(
    filename: str,
    chunk_id: str,
    text: str,
    footer: str = "",
) -> str:
    """Render a single chunk as a card."""

    footer_html = ""

    if footer:
        footer_html = f"""
<div class="context-chunk-score">
    {html.escape(str(footer))}
</div>
"""

    return f"""
<div class="chunk-card">

    <div class="chunk-header">

        <span class="chunk-source">
            {html.escape(str(filename))}
        </span>

        <span class="chunk-id">
            {html.escape(str(chunk_id))}
        </span>

    </div>

    <div class="chunk-text">
        {html.escape(str(text))}
    </div>

    {footer_html}

</div>
"""


# ============================================================================
# MISSION CONTROL + ARCHITECTURE COMPONENTS
# ============================================================================

ARCH_CATEGORIES = {
    "storage": ("#7c8cf8", "Knowledge Storage"),
    "retrieval": ("#5ec4d4", "Retrieval"),
    "processing": ("#6b9fd4", "Processing"),
    "decision": ("#fbbf24", "Decision / Evidence"),
    "generation": ("#4ade80", "Generation"),
}


def render_control_label(text: str) -> str:
    """Small uppercase label used above a control."""

    return f"""
<div class="control-label">
    {html.escape(str(text))}
</div>
"""


def render_mode_description(text: str) -> str:
    """Description line shown under the retrieval toggle."""

    return f"""
<div class="mode-description">
    {html.escape(str(text))}
</div>
"""


def render_nav_group_label(text: str) -> str:
    """Sidebar navigation group heading."""

    return f"""
<div class="nav-group-label">
    {html.escape(str(text))}
</div>
"""


def render_sidebar_divider() -> str:
    """Thin gradient divider for the sidebar."""

    return '<div class="sidebar-divider"></div>'


def render_status_card(
    label: str,
    value: str,
    meta: str = "",
    state: str = "",
    small: bool = False,
) -> str:
    """
    Mission Control status card.

    state: "online", "offline", "unknown" or "" (no indicator dot).
    Passing value=None renders N/A, so an unknown runtime value is
    visibly unknown rather than invented.
    """

    if value is None or str(value).strip() == "":
        value_text = "N/A"
    else:
        value_text = str(value)

    dot = ""

    if state in ("online", "offline", "unknown"):
        dot = f'<span class="status-dot {state}"></span>'

    meta_html = ""

    if meta:
        meta_html = (
            f'<div class="status-card-meta">'
            f"{html.escape(str(meta))}"
            f"</div>"
        )

    value_class = "status-card-value small" if small else "status-card-value"

    return f"""
<div class="status-card">

    <div class="status-card-label">
        {html.escape(str(label))}
    </div>

    <div class="{value_class}">
        {dot}{html.escape(value_text)}
    </div>

    {meta_html}

</div>
"""


def render_arch_node(
    icon: str,
    name: str,
    tech: str = "",
    category: str = "processing",
) -> str:
    """One node of the system architecture diagram."""

    if category not in ARCH_CATEGORIES:
        category = "processing"

    tech_html = ""

    if tech:
        tech_html = (
            f'<div class="arch-node-tech">'
            f"{html.escape(str(tech))}"
            f"</div>"
        )

    return f"""
<div class="arch-node {category}">

    <div class="arch-node-icon">
        {html.escape(str(icon))}
    </div>

    <div class="arch-node-name">
        {html.escape(str(name))}
    </div>

    {tech_html}

</div>
"""


def render_arch_arrow(glyph: str = "▼") -> str:
    """Directional arrow between architecture nodes."""

    return f'<div class="arch-arrow">{html.escape(str(glyph))}</div>'


def render_arch_plane_open(title: str, subtitle: str = "") -> str:
    """
    Open an architecture plane container.

    Must be paired with render_arch_plane_close(). Kept as two calls so
    Streamlit columns can be placed between them.
    """

    subtitle_html = ""

    if subtitle:
        subtitle_html = (
            f'<div class="arch-plane-subtitle">'
            f"{html.escape(str(subtitle))}"
            f"</div>"
        )

    return f"""
<div class="arch-plane">

    <div class="arch-plane-title">
        {html.escape(str(title))}
    </div>

    {subtitle_html}

</div>
"""


def render_arch_legend() -> str:
    """Legend mapping node colors to component categories."""

    items = []

    for color, label in ARCH_CATEGORIES.values():
        items.append(
            f"""
<div class="arch-legend-item">

    <span class="arch-legend-swatch"
          style="background: {color};"></span>

    {html.escape(label)}

</div>
"""
        )

    return (
        '<div class="arch-legend">'
        + "".join(items)
        + "</div>"
    )


# ============================================================================
# OBSERVATION, EVALUATION & CITATION THEMED UI COMPONENTS
# ============================================================================

def render_retrieval_health_card(health_score: float, summary: str = "") -> str:
    """Renders the top-level Retrieval Health Score card."""
    score = round(float(health_score), 1)
    if score >= 80.0:
        color = "#4ade80"
        border_color = "rgba(74, 222, 128, 0.35)"
        status_label = "ROBUST"
        glow = "rgba(74, 222, 128, 0.15)"
    elif score >= 55.0:
        color = "#5ec4d4"
        border_color = "rgba(94, 196, 212, 0.35)"
        status_label = "MODERATE"
        glow = "rgba(94, 196, 212, 0.15)"
    else:
        color = "#fbbf24"
        border_color = "rgba(251, 191, 36, 0.35)"
        status_label = "LIMITED"
        glow = "rgba(251, 191, 36, 0.15)"

    summary_escaped = html.escape(summary) if summary else ""

    return f"""
<div style="background: radial-gradient(circle at 10% 20%, {glow}, rgba(14, 20, 36, 0.85));
            border: 1px solid {border_color}; border-radius: 12px; padding: 1.2rem 1.4rem; margin: 0.8rem 0 1.2rem;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="color: #64748b; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase;">
                RETRIEVAL HEALTH SCORE · DETERMINISTIC RUNTIME EVALUATION
            </div>
            <div style="color: #e8edf7; font-size: 0.95rem; margin-top: 0.3rem; line-height: 1.4;">
                {summary_escaped}
            </div>
        </div>
        <div style="text-align: right; margin-left: 1.5rem;">
            <div style="font-size: 2.2rem; font-weight: 800; color: {color}; line-height: 1;">
                {score}<span style="font-size: 1.1rem; color: #64748b; font-weight: 500;">/100</span>
            </div>
            <div style="color: {color}; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; margin-top: 0.3rem;">
                {status_label}
            </div>
        </div>
    </div>
</div>
"""


def render_evaluation_criterion_card(criterion_key: str, data: dict) -> str:
    """Renders one criterion card with score, reason, definition, and supporting chunks."""
    score = round(float(data.get("score", 0.0)), 1)
    label = html.escape(str(data.get("label", criterion_key)))
    definition = html.escape(str(data.get("definition", "")))
    reason = html.escape(str(data.get("reason", "")))

    if score >= 75.0:
        bar_color = "#4ade80"
        badge_bg = "rgba(74, 222, 128, 0.15)"
        badge_border = "rgba(74, 222, 128, 0.35)"
    elif score >= 50.0:
        bar_color = "#5ec4d4"
        badge_bg = "rgba(94, 196, 212, 0.15)"
        badge_border = "rgba(94, 196, 212, 0.35)"
    else:
        bar_color = "#fbbf24"
        badge_bg = "rgba(251, 191, 36, 0.15)"
        badge_border = "rgba(251, 191, 36, 0.35)"

    return f"""
<div style="background: rgba(14, 20, 36, 0.7); border: 1px solid rgba(120, 150, 210, 0.16);
            border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;">
    <div style="display: flex; justify-content: space-between; align-items: baseline;">
        <span style="font-weight: 600; color: #e8edf7; font-size: 0.92rem;">{label}</span>
        <span style="background: {badge_bg}; border: 1px solid {badge_border}; color: {bar_color};
                     padding: 0.15rem 0.6rem; border-radius: 6px; font-weight: 700; font-size: 0.85rem;">
            {score}/100
        </span>
    </div>
    <div style="background: rgba(255, 255, 255, 0.06); border-radius: 4px; height: 5px; width: 100%; margin: 0.5rem 0;">
        <div style="background: {bar_color}; width: {min(100.0, score)}%; height: 5px; border-radius: 4px;"></div>
    </div>
    <div style="color: #94a3b8; font-size: 0.82rem; margin-top: 0.35rem; line-height: 1.4;">
        {reason}
    </div>
    <div style="color: #64748b; font-size: 0.72rem; margin-top: 0.25rem;">
        ◈ {definition}
    </div>
</div>
"""


def render_confidence_card(confidence: dict) -> str:
    """Renders the Evidence-backed Confidence card with explainable reasons."""
    level = html.escape(str(confidence.get("level", "MEDIUM")))
    score = round(float(confidence.get("score", 50.0)), 1)
    reasons = confidence.get("reasons", [])

    if level == "HIGH":
        color = "#4ade80"
        border_color = "rgba(74, 222, 128, 0.35)"
        bg_glow = "rgba(74, 222, 128, 0.12)"
    elif level == "MEDIUM":
        color = "#5ec4d4"
        border_color = "rgba(94, 196, 212, 0.35)"
        bg_glow = "rgba(94, 196, 212, 0.12)"
    else:
        color = "#fbbf24"
        border_color = "rgba(251, 191, 36, 0.35)"
        bg_glow = "rgba(251, 191, 36, 0.12)"

    reason_items = "".join(
        f'<div style="color: #cbd5e1; font-size: 0.82rem; margin-top: 0.3rem; line-height: 1.35;">{html.escape(r)}</div>'
        for r in reasons
    )

    return f"""
<div style="background: radial-gradient(circle at 5% 5%, {bg_glow}, rgba(14, 20, 36, 0.82));
            border: 1px solid {border_color}; border-radius: 10px; padding: 1rem 1.2rem; margin: 0.6rem 0;">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(120,150,210,0.12); padding-bottom: 0.5rem; margin-bottom: 0.5rem;">
        <span style="color: #64748b; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;">
            EVIDENCE-BACKED CONFIDENCE
        </span>
        <span style="font-weight: 800; font-size: 0.95rem; color: {color}; letter-spacing: 0.08em;">
            {level} · {score}/100
        </span>
    </div>
    <div>
        {reason_items}
    </div>
</div>
"""


def render_citation_card(citation: dict) -> str:
    """Renders a claim citation with passage excerpt and verification badge."""
    marker = html.escape(str(citation.get("marker", "")))
    claim = html.escape(str(citation.get("claim", "")))
    filename = html.escape(str(citation.get("filename") or citation.get("source_label") or "Document"))
    chunk_id = html.escape(str(citation.get("chunk_id", "")))
    passage = html.escape(str(citation.get("passage", "")))
    origin = html.escape(str(citation.get("retrieval_source", "")))
    status = citation.get("status", "supported")
    reason = html.escape(str(citation.get("verification_reason", "")))
    kb_ver = html.escape(str(citation.get("kb_version", "")))
    
    page_badge = ""
    if citation.get("page_range"):
        page_badge = f" · pp. {html.escape(str(citation.get('page_range')))}"
    elif citation.get("page_number"):
        page_badge = f" · p. {html.escape(str(citation.get('page_number')))}"

    if status == "supported":
        badge = '<span style="background: rgba(74,222,128,0.15); border: 1px solid rgba(74,222,128,0.3); color: #4ade80; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">✓ Supported by retrieved evidence</span>'
        border_col = "rgba(74,222,128,0.25)"
    elif status == "partially_supported":
        badge = '<span style="background: rgba(94,196,212,0.15); border: 1px solid rgba(94,196,212,0.3); color: #5ec4d4; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">ℹ Partially Supported</span>'
        border_col = "rgba(94,196,212,0.25)"
    elif status == "parser_failure":
        badge = '<span style="background: rgba(251,191,36,0.15); border: 1px solid rgba(251,191,36,0.3); color: #fbbf24; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">⚡ Citation Parser Issue</span>'
        border_col = "rgba(251,191,36,0.35)"
    else:
        badge = '<span style="background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3); color: #f87171; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">⚠ Unsupported Answer Claim</span>'
        border_col = "rgba(239,68,68,0.35)"

    return f"""
<div style="background: rgba(14, 20, 36, 0.75); border-left: 3px solid {border_col}; border-top: 1px solid rgba(120,150,210,0.14); border-right: 1px solid rgba(120,150,210,0.14); border-bottom: 1px solid rgba(120,150,210,0.14); border-radius: 8px; padding: 0.85rem 1.1rem; margin-bottom: 0.75rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
        <span style="font-weight: 700; color: #5ec4d4; font-size: 0.88rem;">{marker} {filename}<span style="color: #64748b; font-weight: normal; font-size: 0.8rem;">{page_badge} · Chunk {chunk_id} · KB v{kb_ver}</span></span>
        {badge}
    </div>
    <div style="color: #e2e8f0; font-size: 0.84rem; margin-bottom: 0.45rem; font-style: italic;">
        Claim: "{claim}"
    </div>
    <div style="background: rgba(7, 11, 20, 0.7); border: 1px solid rgba(120,150,210,0.12); border-radius: 6px; padding: 0.6rem 0.8rem; font-family: monospace; font-size: 0.8rem; color: #94a3b8; line-height: 1.45;">
        {passage}
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.4rem; font-size: 0.72rem; color: #64748b;">
        <span>Origin: {origin}</span>
        <span>{reason}</span>
    </div>
</div>
"""


def render_citation_coverage_badge(coverage: dict) -> str:
    """Renders the citation coverage badge."""
    pct = round(float(coverage.get("coverage_percentage", 100.0)), 1)
    has_unsupported = coverage.get("has_unsupported", False)
    total = coverage.get("total_claims", 0)
    supported = coverage.get("supported_claims", 0)
    unsupported = coverage.get("unsupported_claims", 0)
    parser_fails = coverage.get("parser_failure_claims", 0)

    if not has_unsupported and parser_fails == 0 and pct >= 90.0:
        color = "#4ade80"
        bg = "rgba(74,222,128,0.15)"
        border = "rgba(74,222,128,0.3)"
        text = f"✓ {pct:.0f}% Citation Coverage ({supported}/{total} claims grounded)"
    elif unsupported > 0 and parser_fails > 0:
        color = "#f87171"
        bg = "rgba(239,68,68,0.15)"
        border = "rgba(239,68,68,0.35)"
        text = f"⚠ {pct:.0f}% Coverage ({unsupported} unsupported claim(s), {parser_fails} parser issue(s))"
    elif parser_fails > 0:
        color = "#fbbf24"
        bg = "rgba(251,191,36,0.15)"
        border = "rgba(251,191,36,0.35)"
        text = f"⚡ {pct:.0f}% Coverage ({parser_fails} citation-parser issue(s))"
    else:
        color = "#f87171"
        bg = "rgba(239,68,68,0.15)"
        border = "rgba(239,68,68,0.35)"
        text = f"⚠ {pct:.0f}% Citation Coverage ({unsupported} unsupported claim(s))"

    return f"""
<div style="display: inline-block; background: {bg}; border: 1px solid {border}; color: {color};
            padding: 0.25rem 0.75rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; margin: 0.3rem 0;">
    {text}
</div>
"""


def render_kb_state_card(kb_state: dict) -> str:
    """Renders Knowledge Base state at query time."""
    version = html.escape(str(kb_state.get("version", 1)))
    doc_count = html.escape(str(kb_state.get("document_count", 0)))
    chroma_chunks = html.escape(str(kb_state.get("chroma_chunks", 0)))
    bm25_chunks = html.escape(str(kb_state.get("bm25_chunks", 0)))
    consistent = kb_state.get("indexes_consistent", True)
    msg = html.escape(str(kb_state.get("consistency_message", "")))

    tag_col = "#4ade80" if consistent else "#fbbf24"
    tag_bg = "rgba(74,222,128,0.15)" if consistent else "rgba(251,191,36,0.15)"

    return f"""
<div style="background: rgba(14, 20, 36, 0.75); border: 1px solid rgba(120, 150, 210, 0.16);
            border-radius: 10px; padding: 0.9rem 1.1rem; margin: 0.6rem 0;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <span style="font-weight: 700; color: #5ec4d4; font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase;">
            KNOWLEDGE BASE STATE · QUERY REVISION v{version}
        </span>
        <span style="background: {tag_bg}; color: {tag_col}; padding: 0.15rem 0.55rem; border-radius: 5px; font-size: 0.72rem; font-weight: 600;">
            {"✓ SYNCHRONIZED" if consistent else "⚠ MISMATCH"}
        </span>
    </div>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; text-align: center; margin: 0.5rem 0;">
        <div style="background: rgba(255,255,255,0.03); padding: 0.4rem; border-radius: 6px;">
            <div style="color: #64748b; font-size: 0.65rem;">Documents</div>
            <div style="color: #e2e8f0; font-weight: 700; font-size: 0.95rem;">{doc_count}</div>
        </div>
        <div style="background: rgba(255,255,255,0.03); padding: 0.4rem; border-radius: 6px;">
            <div style="color: #64748b; font-size: 0.65rem;">Chroma Chunks</div>
            <div style="color: #e2e8f0; font-weight: 700; font-size: 0.95rem;">{chroma_chunks}</div>
        </div>
        <div style="background: rgba(255,255,255,0.03); padding: 0.4rem; border-radius: 6px;">
            <div style="color: #64748b; font-size: 0.65rem;">BM25 Chunks</div>
            <div style="color: #e2e8f0; font-weight: 700; font-size: 0.95rem;">{bm25_chunks}</div>
        </div>
        <div style="background: rgba(255,255,255,0.03); padding: 0.4rem; border-radius: 6px;">
            <div style="color: #64748b; font-size: 0.65rem;">KB Revision</div>
            <div style="color: #5ec4d4; font-weight: 700; font-size: 0.95rem;">v{version}</div>
        </div>
    </div>
    <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 0.4rem;">
        {msg}
    </div>
</div>
"""