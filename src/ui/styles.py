"""
Custom CSS & Design System for ArXiv Research Assistant.
Recreates the exact visual aesthetic of the reference UI specification:
- Deep navy sidebar (#0f172a / #0b132b) with cobalt active pill highlight (#2563eb)
- Crisp light grey background (#f8fafc)
- Clean card elevations, pills, badges, and modern typography matching all 9 panels
"""

def get_custom_css() -> str:
    """Returns CSS styles to inject into the Streamlit app."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #1e293b;
    }

    /* Overall App Background */
    .stApp {
        background-color: #f8fafc !important;
    }

    /* Left Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b132b !important;
        border-right: 1px solid #1e293b !important;
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p {
        color: #94a3b8 !important;
    }

    /* Sidebar Logo & Branding */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 12px 20px 12px;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 20px;
    }
    .sidebar-brand-icon {
        width: 38px;
        height: 38px;
        background: linear-gradient(135deg, #1d4ed8, #3b82f6);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff !important;
        font-size: 20px;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4);
    }
    .sidebar-brand-title {
        color: #f8fafc !important;
        font-weight: 700;
        font-size: 15px;
        line-height: 1.25;
        letter-spacing: -0.2px;
    }
    .sidebar-brand-subtitle {
        color: #64748b !important;
        font-size: 11px;
        margin-top: 2px;
    }

    /* ========================================================
       COMPLETELY REMOVE RADIO BUTTON CIRCLE / CHECKBOX PATTERNS
       ======================================================== */
    /* Hide the 'Navigation' widget label above sidebar items */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > label,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* 1. Hide the radio circle elements (Streamlit emotion classes e1mpz0hj4 & e1mpz0hj5) */
    div[data-testid="stRadio"] div[class*="e1mpz0hj4"],
    div[data-testid="stRadio"] div[class*="e1mpz0hj5"],
    section[data-testid="stSidebar"] div[class*="e1mpz0hj4"],
    section[data-testid="stSidebar"] div[class*="e1mpz0hj5"],
    section[data-testid="stSidebar"] .e1mpz0hj4,
    section[data-testid="stSidebar"] .e1mpz0hj5,
    /* 2. Structural hierarchy: Hide circle div inside the option container */
    div[data-testid="stRadio"] [data-testid="stRadioOption"] > div > div:first-child,
    div[data-testid="stRadio"] [data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"]),
    div[data-testid="stRadio"] label > div > div:first-child,
    div[data-testid="stRadio"] label > div > div:not([data-testid="stMarkdownContainer"]),
    div[data-testid="stRadio"] [data-testid="stRadioOption"] div:has(> div[class*="e1mpz0hj5"]),
    div[data-testid="stRadio"] [data-testid="stRadioOption"] div[aria-hidden="true"],
    /* 3. Hide radio input and its parent span wrapper */
    div[data-testid="stRadio"] [data-testid="stRadioOption"] span:has(> input[type="radio"]),
    div[data-testid="stRadio"] label > span:first-child,
    div[data-testid="stRadio"] input[type="radio"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label > span:first-child,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] svg {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        min-width: 0 !important;
        min-height: 0 !important;
        max-width: 0 !important;
        max-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
        visibility: hidden !important;
        border: none !important;
        background: transparent !important;
        position: absolute !important;
        pointer-events: none !important;
    }

    /* Style the radio option as clean SaaS pill navigation item */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        gap: 4px !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioOption"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        padding: 9px 14px !important;
        border-radius: 8px !important;
        transition: all 0.15s ease !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        color: #94a3b8 !important;
        background-color: transparent !important;
        margin: 1px 0 !important;
        border: none !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioOption"]:hover,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background-color: #1e293b !important;
        color: #f8fafc !important;
    }
    /* Active navigation item pill */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioOption"]:has(input:checked),
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-selected],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background-color: #2563eb !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.35) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected] *,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioOption"]:has(input:checked) *,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-selected] *,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) * {
        color: #ffffff !important;
    }

    /* Hero Banner on Dashboard (Panel 1) */
    .hero-banner {
        background: linear-gradient(180deg, #f0f4ff 0%, #ffffff 100%);
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 40px 32px 36px 32px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .hero-icon-badge {
        width: 54px;
        height: 54px;
        background: #ede9fe;
        color: #7c3aed;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        margin-bottom: 12px;
    }
    .hero-title {
        color: #0f172a;
        font-size: 27px;
        font-weight: 700;
        letter-spacing: -0.4px;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #64748b;
        font-size: 14.5px;
        margin-bottom: 24px;
    }

    /* Prompt Chips (Panel 1 & Panel 9) */
    .prompt-chip {
        background: #ffffff;
        color: #334155;
        border: 1px solid #e2e8f0;
        border-radius: 9999px;
        padding: 6px 14px;
        font-size: 12.5px;
        font-weight: 500;
        display: inline-block;
        margin: 0 4px 8px 4px;
        cursor: pointer;
        transition: all 0.15s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .prompt-chip:hover {
        border-color: #93c5fd;
        background: #eff6ff;
        color: #1d4ed8;
    }

    /* Quick Action Feature Cards (4 cards in row) */
    .feature-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px 20px;
        transition: all 0.2s ease;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .feature-card:hover {
        border-color: #93c5fd;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.08);
        transform: translateY(-2px);
    }
    .feature-icon-box {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        margin-bottom: 16px;
    }
    .box-blue { background: #eff6ff; color: #2563eb; }
    .box-green { background: #ecfdf5; color: #059669; }
    .box-purple { background: #faf5ff; color: #9333ea; }
    .box-rose { background: #fff1f2; color: #e11d48; }

    .feature-title {
        color: #0f172a;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .feature-desc {
        color: #64748b;
        font-size: 12.5px;
        line-height: 1.45;
    }

    /* Paper Search Result Cards (Panel 3) */
    .paper-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        transition: all 0.15s ease;
    }
    .paper-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
    }
    .paper-title {
        color: #0f172a;
        font-size: 17px;
        font-weight: 600;
        letter-spacing: -0.2px;
        margin-bottom: 4px;
    }
    .paper-authors {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 10px;
    }
    .badge-category {
        background: #eff6ff;
        color: #2563eb;
        border: 1px solid #bfdbfe;
        font-size: 11px;
        font-weight: 500;
        padding: 3px 9px;
        border-radius: 9999px;
    }
    .badge-year {
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
        font-size: 11px;
        padding: 3px 9px;
        border-radius: 9999px;
    }
    .badge-relevance {
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 9999px;
    }
    .paper-badges {
        display: flex;
        gap: 6px;
        margin-top: 6px;
        margin-bottom: 8px;
        align-items: center;
        flex-wrap: wrap;
    }
    .paper-snippet, .paper-abstract-snippet {
        color: #334155;
        font-size: 13.5px;
        line-height: 1.5;
        margin-top: 10px;
        margin-bottom: 14px;
    }

    /* Research Chat (Panel 2) */
    .user-message-bubble {
        background: #2563eb;
        color: #ffffff !important;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 20px;
        margin-left: auto;
        max-width: 80%;
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 6px;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.2);
    }
    .user-timestamp {
        font-size: 11px;
        color: #94a3b8;
        text-align: right;
        margin-bottom: 16px;
        margin-right: 4px;
    }
    .assistant-message-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        margin-right: auto;
        max-width: 95%;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        line-height: 1.6;
        font-size: 14px;
    }
    .assistant-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 14px;
    }
    .assistant-avatar {
        width: 30px;
        height: 30px;
        background: #0f172a;
        color: #ffffff;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
    }

    /* Citations Block */
    .sources-container {
        margin-top: 16px;
        padding-top: 14px;
        border-top: 1px solid #f1f5f9;
    }
    .sources-title {
        font-size: 12.5px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .source-item {
        font-size: 13px;
        color: #334155;
        margin-bottom: 6px;
    }
    .source-item a {
        color: #2563eb;
        text-decoration: none;
        font-weight: 500;
    }
    .source-item a:hover {
        text-decoration: underline;
    }

    /* Structured Summary Section Cards (Panel 5) */
    .summary-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .summary-card-header {
        font-size: 15px;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 8px;
    }

    /* Empty and Error State Cards (Panel 9) */
    .state-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 44px 32px;
        text-align: center;
        margin: 28px auto;
        max-width: 500px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .state-icon {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        margin-bottom: 14px;
    }
    .state-icon-empty { background: #eff6ff; color: #2563eb; }
    .state-icon-error { background: #fef2f2; color: #dc2626; }
    .state-title {
        font-size: 17px;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 6px;
    }
    .state-desc {
        font-size: 13.5px;
        color: #64748b;
        line-height: 1.5;
        margin-bottom: 18px;
    }

    /* Streamlit Tabs custom aesthetic */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        font-size: 13.5px;
        font-weight: 500;
        color: #64748b;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #2563eb !important;
    }

    /* Text Input Boxes - Thin Framed Styling for clear visibility */
    div[data-testid="stTextInput"] > div {
        border: 1px solid #94a3b8 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    div[data-testid="stTextInput"] > div:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
    }
    div[data-testid="stTextInput"] input {
        background-color: transparent !important;
        color: #0f172a !important;
        font-size: 14px !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #64748b !important;
    }

    /* Streamlit Button Overrides */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 500;
        font-size: 13px;
        transition: all 0.15s ease;
    }
    div.stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        border-color: #2563eb !important;
        color: #ffffff !important;
        box-shadow: 0 1px 2px rgba(37, 99, 235, 0.2);
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #ffffff !important;
        border-color: #cbd5e1 !important;
        color: #334155 !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #f8fafc !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }

    /* ========================================================
       BOTTOM BAR & CHAT INPUT STYLING
       Bottom Container: Crisp White (#ffffff)
       Input Box: Sleek Dark Mode (#0f172a / #1e293b)
       Minimized bottom spacing
       ======================================================== */
    footer {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* 1. Force the bottom container background from black to clean white */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    .stChatFloatingInputContainer,
    div[class*="stChatFloatingInputContainer"],
    div[class*="stBottom"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
        border-top: 1px solid #e2e8f0 !important;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.03) !important;
        padding-top: 4px !important;
        padding-bottom: 4px !important;
    }

    [data-testid="stBottomBlockContainer"] {
        padding-top: 4px !important;
        padding-bottom: 6px !important;
    }

    /* 2. Chat Input Container */
    [data-testid="stChatInput"] {
        background-color: transparent !important;
        background: transparent !important;
        padding-top: 2px !important;
        padding-bottom: 2px !important;
        margin-bottom: 0 !important;
    }

    /* 3. The Input Box itself - Kept in Dark Mode */
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] div[data-baseweb="textarea"],
    [data-testid="stChatInput"] div[data-baseweb="base-input"] {
        background-color: #0f172a !important;
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }

    [data-testid="stChatInput"] > div:focus-within,
    [data-testid="stChatInput"] div[data-baseweb="base-input"]:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
    }

    /* 4. Textarea inside the Dark Input Box */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        background-color: transparent !important;
        background: transparent !important;
        color: #f8fafc !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
        caret-color: #38bdf8 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        color: #94a3b8 !important;
        opacity: 0.9 !important;
    }

    /* 5. Submit / Send Arrow Button (Replaces square with crisp '➔' arrow) */
    [data-testid="stChatInput"] button {
        background-color: #1e293b !important;
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        width: 32px !important;
        height: 32px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        position: relative !important;
        transition: all 0.15s ease !important;
    }

    [data-testid="stChatInput"] button:hover {
        background-color: #2563eb !important;
        background: #2563eb !important;
        border-color: #2563eb !important;
        color: #ffffff !important;
        box-shadow: 0 0 8px rgba(37, 99, 235, 0.4) !important;
    }

    [data-testid="stChatInput"] button:not(:disabled) {
        background-color: #2563eb !important;
        background: #2563eb !important;
        border-color: #3b82f6 !important;
        color: #ffffff !important;
    }

    /* Hide any default square or SVG icon */
    [data-testid="stChatInput"] button svg {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
    }

    /* Render distinct forward sending arrow (--> / ➔) */
    [data-testid="stChatInput"] button::after {
        content: "➔" !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        color: inherit !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        line-height: 1 !important;
    }
    </style>
    """
