import streamlit as st
import os

# --- Design Tokens ---
PRIMARY_COLOR = "#0F172A"  # Deep Slate
ACCENT_COLOR = "#4F46E5"   # Indigo
BG_COLOR = "#F8FAFC"       # Slate 50
TEXT_COLOR = "#1E293B"     # Slate 800
SECONDARY_TEXT = "#64748B" # Slate 500

GLOBAL_CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Reset & Base */
    html, body, [data-testid="stApp"] {{
        background-color: {BG_COLOR} !important;
        color: {TEXT_COLOR} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    /* Hide default Streamlit elements */
    [data-testid="stSidebarNav"], [data-testid="stSidebar"], [data-testid="collapsedControl"] {{
        display: none !important;
    }}
    [data-testid="stHeader"] {{
        background: transparent !important;
        height: 0 !important;
    }}
    footer {{
        visibility: hidden !important;
    }}

    /* Layout Structure */
    [data-testid="stAppViewContainer"] {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        width: 100vw !important;
    }}

    [data-testid="stAppViewContainer"] > .main {{
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
    }}

    /* block-container: 80% for header flexibility */
    .main .block-container, 
    [data-testid="block-container"],
    div.stMainBlockContainer {{
        max-width: 80% !important;
        width: 80% !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        padding-top: 3rem !important;
        padding-bottom: 5rem !important;
    }}

    /* 
       HEADER CENTERING FIX 
       Targeting the columns container in the header
    */
    [data-testid="stVerticalBlock"] > div:first-child [data-testid="stHorizontalBlock"] {{
        justify-content: center !important;
        gap: 1rem !important;
    }}

    /* 
       BODY WIDTH: 50% of total screen 
       Since parent is 80%, 50/80 = 62.5%
    */
    [data-testid="stVerticalBlock"] > div:nth-child(n+3) {{
        max-width: 62.5% !important;
        width: 62.5% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }}

    /* Smaller Body Font Size */
    html {{
        font-size: 15px !important;
    }}

    /* Premium Cards & Components */
    .stButton > button {{
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }}

    .stButton > button[kind="primary"] {{
        background-color: {ACCENT_COLOR} !important;
    }}

    /* Target the text inside stPageLink labels more specifically */
    [data-testid="stPageLink"] svg {{
        display: none !important;
    }}
    [data-testid="stPageLink"] p {{
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        color: {PRIMARY_COLOR} !important;
        margin: 0 !important;
        white-space: nowrap !important;
    }}

    [data-testid="stPageLink"] a {{
        background: transparent !important;
        border: none !important;
        display: flex !important;
        justify-content: center !important;
        text-align: center !important;
        padding: 0.5rem 0.2rem !important;
    }}
    [data-testid="stPageLink"] a:hover {{
        color: {ACCENT_COLOR} !important;
    }}

    /* Page Title Styling */
    .page-title-container {{
        margin-bottom: 2rem !important;
    }}
    .page-title-container h1 {{
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        letter-spacing: -0.02em !important;
        color: {PRIMARY_COLOR} !important;
        text-align: center !important;
    }}
    .page-title-container p {{
        color: {SECONDARY_TEXT} !important;
        font-size: 1rem !important;
        text-align: center !important;
    }}

    /* Policy Card Refinement */
    .policy-card {{
        background: white !important;
        border-radius: 12px !important;
        padding: 20px !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
    }}
</style>
"""

NAV_ITEMS = [
    ("Home", "app.py"),
    ("Chunking", "pages/2_Chunking.py"),
    ("Prompt", "pages/3_Prompt.py"),
    ("Retriever", "pages/4_Retriever.py"),
    ("Evaluation", "pages/5_Evaluation.py"),
    ("Logs", "pages/6_Logs.py"),
]

def inject_ui():
    """Injects the global CSS and sets page config."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    
    # Navigation Header
    header_container = st.container()
    with header_container:
        # Using a centered container for the header links
        # We use fewer columns or a specific ratio to help centering
        cols = st.columns([1, 1, 1, 1, 1, 1])
        for i, (label, path) in enumerate(NAV_ITEMS):
            with cols[i]:
                st.page_link(path, label=label, use_container_width=True)
    
    st.markdown("---")

def render_page_title(title: str, subtitle: str = ""):
    st.markdown(f"""
        <div class="page-title-container">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)
