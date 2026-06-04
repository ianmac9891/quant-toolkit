"""Streamlit entry point — navigation wrapper. Run with: streamlit run app.py"""

import streamlit as st

st.set_page_config(
    page_title="Quant Toolkit",
    layout="wide",
    initial_sidebar_state="collapsed",   # collapsed by default on mobile
)

# ── PWA head injection (iOS "Add to Home Screen" + Android manifest) ──────────
# Streamlit has no official head API; we inject via a script that runs client-side
# and appends meta/link tags to <head>. The guard prevents double-injection on
# Streamlit's hot-reloads.
st.markdown(
    """
    <script>
    (function injectPWA() {
        var head = document.querySelector('head');
        if (!head) { setTimeout(injectPWA, 40); return; }
        if (head.querySelector('meta[name="apple-mobile-web-app-capable"]')) return;

        var APP_NAME = 'Quant Toolkit';

        function meta(name, content) {
            var m = document.createElement('meta');
            m.name = name; m.content = content;
            head.appendChild(m);
        }
        function link(rel, href, extra) {
            var l = document.createElement('link');
            l.rel = rel; l.href = href;
            if (extra) Object.assign(l, extra);
            head.appendChild(l);
        }

        // iOS standalone mode
        meta('apple-mobile-web-app-capable',          'yes');
        meta('apple-mobile-web-app-status-bar-style', 'black');
        meta('apple-mobile-web-app-title',            APP_NAME);

        // Android / Chrome theme
        meta('mobile-web-app-capable', 'yes');
        meta('theme-color',            '#0E1117');

        // iOS home-screen icon (180 × 180 PNG, served from Streamlit static)
        link('apple-touch-icon',  '/app/static/icon-180.png', {sizes: '180x180'});

        // Web-app manifest (used by Android/Chrome install prompt)
        link('manifest', '/app/static/manifest.json');

        // Viewport: viewport-fit=cover fills the notch area on modern iPhones
        var vp = head.querySelector('meta[name="viewport"]');
        if (vp) vp.content = 'width=device-width, initial-scale=1, shrink-to-fit=no, viewport-fit=cover';

        // ── Force and hold the document title ──────────────────────────────
        // Streamlit's base HTML is hard-coded as <title>Streamlit</title>.
        // iOS reads document.title at the moment the user taps Share, so we
        // set it here and use a MutationObserver to re-apply it whenever
        // Streamlit's React runtime tries to change it back.
        document.title = APP_NAME;
        var titleEl = document.querySelector('title');
        if (titleEl) {
            new MutationObserver(function() {
                if (document.title !== APP_NAME) document.title = APP_NAME;
            }).observe(titleEl, { childList: true, characterData: true, subtree: true });
        }
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# ── Mobile-responsive CSS ─────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Safe-area insets for iPhone notch / home-indicator ── */
    .main .block-container {
        padding-bottom: max(2rem, env(safe-area-inset-bottom));
        padding-left:   max(1rem, env(safe-area-inset-left));
        padding-right:  max(1rem, env(safe-area-inset-right));
    }

    /* ── Mobile overrides (≤ 767 px) ── */
    @media (max-width: 767px) {

        /* Tighter top padding — the sidebar toggle is always visible */
        .main .block-container {
            padding-top:   0.75rem !important;
            padding-left:  0.75rem !important;
            padding-right: 0.75rem !important;
            max-width:     100vw  !important;
        }

        /* Stack st.columns() vertically instead of side-by-side */
        [data-testid="column"] {
            width:     100% !important;
            flex:      0 0 100% !important;
            min-width: 100% !important;
        }

        /* Scrollable data tables / editors */
        [data-testid="stDataFrame"],
        [data-testid="stDataEditor"],
        .stPlotlyChart {
            overflow-x: auto !important;
            max-width:  100% !important;
        }

        /* Prevent iOS from zooming into inputs (requires font-size ≥ 16 px) */
        input, select, textarea,
        [data-testid="stTextInput"]  input,
        [data-testid="stNumberInput"] input {
            font-size: 16px !important;
        }

        /* Larger tap targets for buttons */
        .stButton > button {
            min-height: 44px !important;
            font-size:  15px !important;
            padding:    0.5rem 1.25rem !important;
        }

        /* Make sidebar toggle easier to tap */
        [data-testid="collapsedControl"] {
            width:  44px !important;
            height: 44px !important;
        }

        /* Metric cards: full width and spaced */
        [data-testid="stMetric"] {
            background:    var(--secondary-background-color);
            border-radius: 8px;
            padding:       0.65rem 0.75rem !important;
            margin-bottom: 0.4rem;
        }

        /* Hide the wide-mode gap that appears on narrow screens */
        .stApp > header { display: none; }

        /* No horizontal scroll on the root */
        .stApp, .main { overflow-x: hidden !important; }
    }

    /* ── Plotly modebar: hide on touch screens (toolbar unusable on mobile) ── */
    @media (hover: none) {
        .modebar-container { display: none !important; }
        .stSlider [role="slider"] {
            width:  28px !important;
            height: 28px !important;
        }
    }

    /* ── Plotly axis tick labels: keep readable on narrow screens ── */
    @media (max-width: 767px) {
        .xtick text, .ytick text {
            font-size: 11px !important;
        }
        /* Let chart containers scroll horizontally rather than overflow */
        .stPlotlyChart > div {
            overflow-x: auto !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar header ────────────────────────────────────────────────────────────
st.sidebar.markdown("**Quant Toolkit**")
st.sidebar.caption("Quantitative analysis tools")

# ── Navigation ────────────────────────────────────────────────────────────────
pg = st.navigation(
    {
        "": [
            st.Page("home.py", title="Home", icon=":material/home:"),
        ],
        "Analysis": [
            st.Page("pages/stock_analysis.py",  title="Stock Analysis",        icon=":material/show_chart:"),
            st.Page("pages/vol_forecast.py",    title="Vol Forecast",          icon=":material/trending_up:"),
            st.Page("pages/event_study.py",     title="Event Study",           icon=":material/query_stats:"),
            st.Page("pages/options_planner.py", title="Options Trade Planner", icon=":material/candlestick_chart:"),
        ],
        "Portfolio": [
            st.Page("pages/portfolio_optimizer.py", title="Portfolio Optimizer", icon=":material/tune:"),
            st.Page("pages/risk_model.py",           title="Risk Model",          icon=":material/shield:"),
            st.Page("pages/backtester.py",           title="Backtester",          icon=":material/replay:"),
        ],
        "Research": [
            st.Page("pages/screener.py",    title="Screener",    icon=":material/search:"),
            st.Page("pages/anomaly_lab.py", title="Anomaly Lab", icon=":material/science:"),
        ],
    }
)
pg.run()
