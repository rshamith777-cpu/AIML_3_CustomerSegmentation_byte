"""
=============================================================================
RETAIL DYNAMICS // Swiss Graphic Customer Intelligence & 3D Galaxy
Arithmatrix Virtual Internship Program (AVIP 2026) · Task 3 Deliverable
Style: International Typographic Style (Swiss Style) · Light Editorial
Unsupervised Learning: K-Means++ Clustering (k=5)
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & SWISS INTERNATIONAL TYPOGRAPHIC THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RETAIL // DYNAMICS — Swiss Customer Intelligence",
    page_icon="🟥",
    layout="wide",
    initial_sidebar_state="expanded"
)

SWISS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --swiss-bg: #FAFAFA;
    --swiss-surface: #FFFFFF;
    --swiss-ink: #09090B;
    --swiss-muted: #71717A;
    --swiss-subtle: #A1A1AA;
    --swiss-border: #E4E4E7;
    --swiss-border-strong: #18181B;
    
    /* Swiss International Accent Palette */
    --swiss-red: #FF2D20;
    --swiss-cobalt: #0047FF;
    --swiss-emerald: #059669;
    --swiss-amber: #D97706;
    --swiss-slate: #64748B;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: var(--swiss-ink) !important;
}

.stApp {
    background-color: var(--swiss-bg) !important;
}

/* Sidebar Styling: Clean Architectural Paper */
[data-testid="stSidebar"] {
    background-color: #F4F4F5 !important;
    border-right: 1px solid var(--swiss-border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--swiss-ink) !important;
}

/* Swiss Header */
.swiss-header-container {
    padding: 1.2rem 0 1.6rem 0;
    border-bottom: 2px solid var(--swiss-border-strong);
    margin-bottom: 1.8rem;
}
.swiss-super {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--swiss-red);
    margin-bottom: 0.35rem;
}
.swiss-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1.0;
    color: var(--swiss-ink);
    margin: 0;
    text-transform: uppercase;
}
.swiss-title span {
    color: var(--swiss-cobalt);
}
.swiss-desc {
    font-size: 0.95rem;
    color: var(--swiss-muted);
    margin-top: 0.4rem;
    max-width: 900px;
}

/* Swiss Badges & Structural Pills */
.swiss-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 0.25rem 0.55rem;
    background: #FFFFFF;
    border: 1px solid var(--swiss-border-strong);
    color: var(--swiss-ink);
    border-radius: 0px;
    margin-right: 0.4rem;
    margin-bottom: 0.3rem;
    text-transform: uppercase;
}
.swiss-badge-red {
    background: var(--swiss-red);
    border-color: var(--swiss-red);
    color: #FFFFFF !important;
}
.swiss-badge-cobalt {
    background: var(--swiss-cobalt);
    border-color: var(--swiss-cobalt);
    color: #FFFFFF !important;
}

/* Swiss Bento Card Containers */
.swiss-card {
    background: var(--swiss-surface);
    border: 1px solid var(--swiss-border);
    border-radius: 0px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.swiss-card:hover {
    border-color: var(--swiss-border-strong);
}

.swiss-section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--swiss-muted);
    border-bottom: 1px solid var(--swiss-border);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* Persona Poster Card (Editorial Hero) */
.swiss-persona-hero {
    background: #FFFFFF;
    border: 2px solid var(--swiss-border-strong);
    padding: 1.8rem;
    margin-bottom: 1.8rem;
    position: relative;
}
.swiss-persona-hero::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 8px;
    height: 100%;
}

.persona-archetype-name {
    font-family: 'Inter', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

/* Metric Display Cards */
.swiss-metric-tile {
    background: #FFFFFF;
    border: 1px solid var(--swiss-border);
    padding: 1rem;
    border-radius: 0px;
    text-align: left;
}
.swiss-metric-tile:hover {
    border-color: var(--swiss-border-strong);
}
.swiss-metric-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--swiss-ink);
    line-height: 1.0;
}
.swiss-metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    color: var(--swiss-muted);
    margin-top: 0.4rem;
    letter-spacing: 0.05em;
}

/* Swiss Grid Presets Buttons */
.stButton > button {
    border-radius: 0px !important;
    border: 1px solid var(--swiss-border-strong) !important;
    background: #FFFFFF !important;
    color: var(--swiss-ink) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: var(--swiss-ink) !important;
    color: #FFFFFF !important;
}

/* Sliders */
.stSlider label, .stSelectbox label, .stNumberInput label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    color: var(--swiss-ink) !important;
}

/* Tabs: Architectural Swiss Tab Strip */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    border-bottom: 2px solid var(--swiss-border-strong);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--swiss-muted);
    background-color: transparent;
    border-radius: 0px;
    padding: 10px 22px;
    border: none;
}
.stTabs [aria-selected="true"] {
    color: var(--swiss-ink) !important;
    background-color: #FFFFFF !important;
    border-bottom: 3px solid var(--swiss-red) !important;
    font-weight: 700;
}
</style>
"""
st.markdown(SWISS_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA & MODEL ARTIFACT CACHING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "kmeans_model.joblib")
    scaler_path = os.path.join(base_dir, "models", "scaler.joblib")
    data_path = os.path.join(base_dir, "data", "Mall_Customers.csv")

    km = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df_raw = pd.read_csv(data_path)

    # Assign clusters to existing dataset for 3D visualization
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X_scaled = scaler.transform(df_raw[features])
    df_raw['Cluster'] = km.predict(X_scaled)

    # Centroids in real physical units
    centroids_real = scaler.inverse_transform(km.cluster_centers_)
    df_centroids = pd.DataFrame(
        centroids_real,
        columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    )
    df_centroids['Cluster'] = range(len(df_centroids))

    return km, scaler, df_raw, df_centroids

km, scaler, df_raw, df_centroids = load_ml_assets()


# -----------------------------------------------------------------------------
# 3. SWISS ARCHETYPE PROFILES & COLOR SCHEME
# -----------------------------------------------------------------------------
SWISS_METADATA = {
    0: {
        "name": "THE FRUGAL SHOPPER",
        "tag": "SEGMENT 00 · VALUE CONSCIOUS",
        "color": "#64748B",  # Slate
        "accent": "#475569",
        "summary": "Low Income · Low Spending · Mature Cohort (Mean Age: 46.3 yrs)",
        "behavior": "Conservative purchasers with limited disposable income who transact strictly on price markdowns, seasonal clearance, and essential volume deals. High voucher responsiveness.",
        "tactics": [
            "Clearance event mailers and price drop SMS alerts",
            "Bulk-pack staple goods with guaranteed lowest per-unit cost",
            "Zero-markup staple goods promotions driving baseline mall foot traffic",
            "Digital discount vouchers instantly scannable at point of sale"
        ],
        "roi_archetype_value": 45.0,
        "default_conversion": 4.5
    },
    1: {
        "name": "THE IMPULSIVE TRENDSETTER",
        "tag": "SEGMENT 01 · EARLY ADOPTER",
        "color": "#D97706",  # Swiss Amber
        "accent": "#B45309",
        "summary": "Low-to-Mid Income · High Spending · Youth Demographics (Mean Age: 25.2 yrs)",
        "behavior": "Youthful demographic exhibiting low earnings but maximal spending velocity. Driven by viral social media momentum, aesthetic drops, dining, and impulse acquisitions.",
        "tactics": [
            "TikTok and Instagram drop announcements with real-time countdown clocks",
            "Gen-Z micro-influencer collaborations and experiential pop-up lounges",
            "Buy-Now-Pay-Later (BNPL / Klarna) frictionless checkout options",
            "Limited-edition merchandise capsules with strict scarcity triggers"
        ],
        "roi_archetype_value": 95.0,
        "default_conversion": 11.0
    },
    2: {
        "name": "THE VIP POWER SHOPPER",
        "tag": "SEGMENT 02 · ENTERPRISE REVENUE",
        "color": "#0047FF",  # Swiss International Cobalt Blue
        "accent": "#0030B8",
        "summary": "High Income · High Spending · Prime Career (Mean Age: 32.9 yrs)",
        "behavior": "The foundational revenue pillar of the retail ecosystem. Affluent, brand-loyal, and maintaining peak transaction frequency. Demands bespoke white-glove hospitality.",
        "tactics": [
            "Private salon shopping appointments with dedicated retail stylists",
            "Exclusive invite-only previews for luxury seasonal collections",
            "Priority valet parking and private hospitality lounge privileges",
            "Unadvertised tiered loyalty perks and personalized anniversary gifts"
        ],
        "roi_archetype_value": 420.0,
        "default_conversion": 16.5
    },
    3: {
        "name": "THE CAREFUL WEALTHY",
        "tag": "SEGMENT 03 · EXPANSION FRONTIER",
        "color": "#FF2D20",  # Swiss International Red
        "accent": "#DC2626",
        "summary": "High Income · Low Spending · Mid-Career (Mean Age: 39.9 yrs)",
        "behavior": "Affluent professionals with substantial discretionary capital who exercise strict financial discipline. They reject superficial hype and require tangible proof of quality.",
        "tactics": [
            "Lifetime warranty protection and craftsmanship heritage campaigns",
            "Detailed technical specifications and durability benchmark documentation",
            "High-end artisanal goods, precision mechanics, and heirloom asset marketing",
            "Low-pressure consultations emphasizing long-term value retention"
        ],
        "roi_archetype_value": 280.0,
        "default_conversion": 6.8
    },
    4: {
        "name": "THE STANDARD SHOPPER",
        "tag": "SEGMENT 04 · BEDROCK COHORT",
        "color": "#059669",  # Swiss Emerald Green
        "accent": "#047857",
        "summary": "Moderate Income · Moderate Spending · Mature (Mean Age: 55.6 yrs)",
        "behavior": "The dependable median population. Consistent household needs, balanced expenditure cycles, and reliable multi-category seasonal shopping routines.",
        "tactics": [
            "Predictable seasonal promotional cycles (Spring, Back-to-School, Holidays)",
            "Family combo packages and household multi-category bundles",
            "Transparent cashback point-based rewards programs",
            "Omnichannel inventory pickup and straightforward returns processing"
        ],
        "roi_archetype_value": 110.0,
        "default_conversion": 8.2
    }
}


# -----------------------------------------------------------------------------
# 4. SIDEBAR // TELEMETRY & SWISS GRID INPUTS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 0.8rem; border-bottom: 2px solid #18181B; margin-bottom: 1.2rem;">
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; font-weight: 700; color: #FF2D20; letter-spacing: 0.1em;">
            AVIP 2026 // TASK 03
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 1.6rem; font-weight: 900; letter-spacing: -0.03em; color: #09090B;">
            SWISS // ML
        </div>
        <div style="font-size: 0.78rem; color: #71717A;">
            International Spatial Clustering
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <span class="swiss-badge swiss-badge-red">K = 5</span>
        <span class="swiss-badge">N = 200</span>
        <span class="swiss-badge swiss-badge-cobalt">SIL: 0.4166</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 01 // INPUT PARAMETERS")
    st.caption("Inject coordinate values into standardized Euclidean feature space.")

    # Archetype Presets
    st.markdown("**QUICK PRESET INGESTION:**")
    p1, p2 = st.columns(2)
    with p1:
        if st.button("💎 VIP POWER", use_container_width=True):
            st.session_state['age_input'] = 33
            st.session_state['income_input'] = 86
            st.session_state['spend_input'] = 82
        if st.button("⚡ IMPULSIVE", use_container_width=True):
            st.session_state['age_input'] = 25
            st.session_state['income_input'] = 41
            st.session_state['spend_input'] = 62
        if st.button("🛡️ STANDARD", use_container_width=True):
            st.session_state['age_input'] = 56
            st.session_state['income_input'] = 54
            st.session_state['spend_input'] = 49
    with p2:
        if st.button("🎯 CAREFUL", use_container_width=True):
            st.session_state['age_input'] = 40
            st.session_state['income_input'] = 86
            st.session_state['spend_input'] = 19
        if st.button("🏷️ FRUGAL", use_container_width=True):
            st.session_state['age_input'] = 46
            st.session_state['income_input'] = 27
            st.session_state['spend_input'] = 18
        if st.button("↺ RESET", use_container_width=True):
            st.session_state['age_input'] = 33
            st.session_state['income_input'] = 86
            st.session_state['spend_input'] = 82

    # State variables
    if 'age_input' not in st.session_state:
        st.session_state['age_input'] = 33
    if 'income_input' not in st.session_state:
        st.session_state['income_input'] = 86
    if 'spend_input' not in st.session_state:
        st.session_state['spend_input'] = 82

    age = st.slider("Customer Age (Years)", min_value=18, max_value=70,
                    value=st.session_state['age_input'], key="slider_age")
    income = st.slider("Annual Income (k$)", min_value=15, max_value=140,
                       value=st.session_state['income_input'], key="slider_income")
    spending = st.slider("Spending Score (1-100)", min_value=1, max_value=99,
                         value=st.session_state['spend_input'], key="slider_spend")

    st.markdown("---")
    st.markdown("### 02 // PARADIGM NOTE")
    st.markdown("""
    <div style="font-size: 0.78rem; color: #52525B; line-height: 1.5;">
        <b>Unsupervised Learning:</b> No labels exist in clustering. Evaluation is strictly mathematical 
        (Silhouette 0.4166) and structural. No classification accuracy is computed.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. REAL-TIME CLUSTERING INFERENCE & EUCLIDEAN METRICS
# -----------------------------------------------------------------------------
input_df = pd.DataFrame([[age, income, spending]],
                        columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
input_scaled = scaler.transform(input_df)
predicted_cluster = int(km.predict(input_scaled)[0])
meta = SWISS_METADATA[predicted_cluster]

# Compute Euclidean distances to all 5 centroids in StandardScaler space
cluster_centers_scaled = km.cluster_centers_
distances = np.linalg.norm(cluster_centers_scaled - input_scaled, axis=1)
dist_df = pd.DataFrame({
    'Cluster': range(5),
    'Segment': [SWISS_METADATA[i]['name'] for i in range(5)],
    'Distance': np.round(distances, 3),
    'Confidence': np.round(100.0 / (1.0 + distances), 1)
}).sort_values('Distance')


# -----------------------------------------------------------------------------
# 6. MAIN CONTENT // SWISS HEADER & HERO POSTER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="swiss-header-container">
    <div class="swiss-super">K-MEANS++ SPATIAL CLUSTERING · K=5 · AVIP 2026</div>
    <div class="swiss-title">RETAIL // <span>DYNAMICS</span></div>
    <div class="swiss-desc">
        International Typographic customer segmentation suite. Quantitative 3D spatial partitioning of 200 benchmark mall shoppers via StandardScaler and Euclidean distance optimization.
    </div>
</div>
""", unsafe_allow_html=True)

# Editorial Swiss Persona Poster
hero_html = f"""
<div class="swiss-persona-hero" style="border-left: 10px solid {meta['color']};">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; font-weight: 700; color: {meta['color']}; letter-spacing: 0.1em;">
                PARTITION ASSIGNMENT // CLUSTER 0{predicted_cluster}
            </div>
            <div class="persona-archetype-name" style="color: #09090B;">
                {meta['name']}
            </div>
            <div style="margin-top: 0.2rem;">
                <span class="swiss-badge" style="background: {meta['color']}; color: #FFFFFF !important; border-color: {meta['color']};">
                    {meta['tag']}
                </span>
                <span class="swiss-badge">
                    N = {df_raw['Cluster'].value_counts()[predicted_cluster]} SHOPPERS ({df_raw['Cluster'].value_counts()[predicted_cluster]/2.0:.1f}%)
                </span>
            </div>
            <p style="color: #52525B; font-size: 0.95rem; margin-top: 0.8rem; max-width: 820px; line-height: 1.6;">
                {meta['behavior']}
            </p>
        </div>
        <div style="border: 2px solid #18181B; padding: 1rem 1.4rem; background: #FFFFFF; text-align: right; min-width: 180px; margin-top: 0.6rem;">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; font-weight: 700; color: #71717A; letter-spacing: 0.08em; text-transform: uppercase;">
                AFFINITY SCORE
            </div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 2.4rem; font-weight: 800; color: {meta['color']}; line-height: 1.0; margin: 0.3rem 0;">
                {100.0 / (1.0 + distances[predicted_cluster]):.1f}%
            </div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #71717A;">
                L2 DIST: {distances[predicted_cluster]:.3f}
            </div>
        </div>
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. MAIN INTERACTIVE TABS (SWISS NUMBERED HEADERS)
# -----------------------------------------------------------------------------
tab_3d, tab_playbook, tab_roi, tab_metrics = st.tabs([
    "01 // 3D Spatial Galaxy",
    "02 // Strategic Playbooks",
    "03 // Campaign ROI Simulator",
    "04 // Evaluation Benchmarks"
])

# -----------------------------------------------------------------------------
# TAB 1: 3D SPATIAL GALAXY (HIGH-CONTRAST DEEP SPACE VIEWPORT)
# -----------------------------------------------------------------------------
with tab_3d:
    # Projection View Selector
    v_col1, v_col2 = st.columns([2.2, 1.8])
    with v_col1:
        view_mode = st.radio(
            "SPATIAL PROJECTION VIEW:",
            ["🌌 3D Deep Space Galaxy", "📐 2D Income vs. Spending Plane", "👤 2D Age vs. Spending Plane"],
            horizontal=True
        )
    with v_col2:
        if "3D" in view_mode:
            st.markdown("""
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #71717A; text-align: right; padding-top: 0.8rem;">
                INTERACTION: <b>CLICK + DRAG</b> to orbit · <b>SCROLL</b> to zoom · <b>HOVER</b> for telemetry
            </div>
            """, unsafe_allow_html=True)

    col_vis, col_side = st.columns([2.8, 1.2])

    # Vibrant galaxy palette engineered for deep space contrast
    GALAXY_COLORS = {
        0: "#94A3B8",  # Cool Silver Slate
        1: "#F59E0B",  # Luminous Amber
        2: "#00E5FF",  # Laser Cyan / Swiss Cobalt
        3: "#FF334B",  # Vibrant Swiss Vermilion
        4: "#10B981"   # Spring Emerald
    }

    with col_vis:
        if HAS_PLOTLY:
            if "3D" in view_mode:
                # Architectural Swiss darkroom viewport container header
                st.markdown("""
                <div style="background: #0B0D11; border: 2px solid #18181B; border-bottom: none; padding: 0.65rem 1rem; display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.08em;">
                        🌌 DEEP FIELD SPATIAL VIEWPORT // 200 DATA NODES + 5 CENTROID CORES
                    </div>
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #00E5FF; font-weight: 600;">
                        ● STEREO DEPTH ON
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fig = go.Figure()

                # Plot existing 200 mall customers with glowing high-contrast nodes
                for c_id in range(5):
                    subset = df_raw[df_raw['Cluster'] == c_id]
                    c_meta = SWISS_METADATA[c_id]
                    node_color = GALAXY_COLORS[c_id]
                    fig.add_trace(go.Scatter3d(
                        x=subset['Annual Income (k$)'],
                        y=subset['Spending Score (1-100)'],
                        z=subset['Age'],
                        mode='markers',
                        name=f"C0{c_id} · {c_meta['name'][:14]}",
                        marker=dict(
                            size=5.5,
                            color=node_color,
                            opacity=0.88,
                            line=dict(width=0.8, color='#0B0D11')
                        ),
                        hovertemplate=(
                            f"<b>{c_meta['name']} (C0{c_id})</b><br>"
                            "Annual Income: $%{x}k<br>"
                            "Spending Score: %{y}/100<br>"
                            "Customer Age: %{z} yrs<extra></extra>"
                        )
                    ))

                # Plot Cluster Centroids as gleaming white/gold stellar diamonds
                fig.add_trace(go.Scatter3d(
                    x=df_centroids['Annual Income (k$)'],
                    y=df_centroids['Spending Score (1-100)'],
                    z=df_centroids['Age'],
                    mode='markers+text',
                    name='CENTROIDS (REAL UNITS)',
                    text=[f"CORE 0{i}" for i in range(5)],
                    textposition="top center",
                    textfont=dict(family="IBM Plex Mono", size=10, color="#FFFFFF"),
                    marker=dict(
                        size=11,
                        symbol='diamond',
                        color='#FFFFFF',
                        line=dict(width=2.5, color='#F59E0B')
                    ),
                    hovertemplate="<b>CENTROID CORE C0%{pointNumber}</b><br>Income: $%{x:.1f}k<br>Spending: %{y:.1f}<br>Age: %{z:.1f} yrs<extra></extra>"
                ))

                # Project current active target in 3D Euclidean space
                fig.add_trace(go.Scatter3d(
                    x=[income],
                    y=[spending],
                    z=[age],
                    mode='markers+text',
                    name='⚡ ACTIVE TARGET',
                    text=['ACTIVE TARGET'],
                    textposition="bottom center",
                    textfont=dict(family="IBM Plex Mono", size=12, color="#00E5FF"),
                    marker=dict(
                        size=15,
                        symbol='cross',
                        color="#00E5FF",
                        line=dict(width=3, color='#FFFFFF')
                    ),
                    hovertemplate=(
                        "<b>⚡ ACTIVE TARGET</b><br>"
                        f"Assigned: {meta['name']} (C0{predicted_cluster})<br>"
                        f"Income: ${income}k<br>"
                        f"Spending: {spending}/100<br>"
                        f"Age: {age} yrs<extra></extra>"
                    )
                ))

                fig.update_layout(
                    scene=dict(
                        xaxis=dict(
                            title=dict(text="ANNUAL INCOME (k$)", font=dict(family="IBM Plex Mono", size=11, color="#E2E8F0")),
                            backgroundcolor="#0F1117",
                            gridcolor="#232833",
                            showbackground=True,
                            zerolinecolor="#475569"
                        ),
                        yaxis=dict(
                            title=dict(text="SPENDING SCORE (1-100)", font=dict(family="IBM Plex Mono", size=11, color="#E2E8F0")),
                            backgroundcolor="#0F1117",
                            gridcolor="#232833",
                            showbackground=True,
                            zerolinecolor="#475569"
                        ),
                        zaxis=dict(
                            title=dict(text="AGE (YEARS)", font=dict(family="IBM Plex Mono", size=11, color="#E2E8F0")),
                            backgroundcolor="#0F1117",
                            gridcolor="#232833",
                            showbackground=True,
                            zerolinecolor="#475569"
                        ),
                        camera=dict(
                            eye=dict(x=1.75, y=-1.75, z=1.2)
                        )
                    ),
                    paper_bgcolor="#0B0D11",
                    plot_bgcolor="#0B0D11",
                    margin=dict(l=0, r=0, t=5, b=0),
                    legend=dict(
                        bgcolor="rgba(15, 17, 23, 0.92)",
                        bordercolor="#232833",
                        borderwidth=1,
                        font=dict(family="IBM Plex Mono", color="#E2E8F0", size=10)
                    ),
                    height=590
                )

                st.plotly_chart(fig, use_container_width=True)

            elif "Income vs. Spending" in view_mode:
                # 2D Architectural Floorplan
                st.markdown("""
                <div style="background: #0B0D11; border: 2px solid #18181B; border-bottom: none; padding: 0.65rem 1rem; display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.08em;">
                        📐 2D FLOORPLAN // ANNUAL INCOME (k$) vs SPENDING SCORE (1-100)
                    </div>
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #10B981; font-weight: 600;">
                        ● ORTHOGRAPHIC
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fig2d = go.Figure()
                for c_id in range(5):
                    sub = df_raw[df_raw['Cluster'] == c_id]
                    fig2d.add_trace(go.Scatter(
                        x=sub['Annual Income (k$)'],
                        y=sub['Spending Score (1-100)'],
                        mode='markers',
                        name=f"C0{c_id} · {SWISS_METADATA[c_id]['name'][:14]}",
                        marker=dict(size=8, color=GALAXY_COLORS[c_id], opacity=0.85, line=dict(width=0.8, color="#0B0D11")),
                        hovertemplate=f"<b>{SWISS_METADATA[c_id]['name']}</b><br>Income: $%{{x}}k<br>Spending: %{{y}}<extra></extra>"
                    ))

                # Centroids 2D
                fig2d.add_trace(go.Scatter(
                    x=df_centroids['Annual Income (k$)'],
                    y=df_centroids['Spending Score (1-100)'],
                    mode='markers+text',
                    name='Centroids',
                    text=[f"C0{i}" for i in range(5)],
                    textposition="top center",
                    textfont=dict(family="IBM Plex Mono", size=11, color="#FFFFFF"),
                    marker=dict(size=14, symbol='diamond', color='#FFFFFF', line=dict(width=2, color='#F59E0B'))
                ))

                # Active target
                fig2d.add_trace(go.Scatter(
                    x=[income],
                    y=[spending],
                    mode='markers+text',
                    name='Active Target',
                    text=['TARGET'],
                    textposition="bottom center",
                    textfont=dict(family="IBM Plex Mono", size=12, color="#00E5FF"),
                    marker=dict(size=18, symbol='cross', color='#00E5FF', line=dict(width=3, color='#FFFFFF'))
                ))

                fig2d.update_layout(
                    xaxis=dict(title=dict(text="ANNUAL INCOME (k$)", font=dict(family="IBM Plex Mono", color="#E2E8F0")), gridcolor="#232833", zerolinecolor="#475569"),
                    yaxis=dict(title=dict(text="SPENDING SCORE (1-100)", font=dict(family="IBM Plex Mono", color="#E2E8F0")), gridcolor="#232833", zerolinecolor="#475569"),
                    paper_bgcolor="#0B0D11",
                    plot_bgcolor="#0F1117",
                    font=dict(color="#E2E8F0"),
                    margin=dict(l=40, r=20, t=10, b=40),
                    height=590,
                    legend=dict(bgcolor="rgba(15, 17, 23, 0.92)", bordercolor="#232833", borderwidth=1)
                )
                st.plotly_chart(fig2d, use_container_width=True)

            else:
                # 2D Age vs Spending
                st.markdown("""
                <div style="background: #0B0D11; border: 2px solid #18181B; border-bottom: none; padding: 0.65rem 1rem; display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.08em;">
                        👤 2D DEMOGRAPHIC PLANE // AGE (YEARS) vs SPENDING SCORE (1-100)
                    </div>
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #FF334B; font-weight: 600;">
                        ● DEMOGRAPHIC
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fig_age = go.Figure()
                for c_id in range(5):
                    sub = df_raw[df_raw['Cluster'] == c_id]
                    fig_age.add_trace(go.Scatter(
                        x=sub['Age'],
                        y=sub['Spending Score (1-100)'],
                        mode='markers',
                        name=f"C0{c_id} · {SWISS_METADATA[c_id]['name'][:14]}",
                        marker=dict(size=8, color=GALAXY_COLORS[c_id], opacity=0.85, line=dict(width=0.8, color="#0B0D11")),
                        hovertemplate=f"<b>{SWISS_METADATA[c_id]['name']}</b><br>Age: %{{x}} yrs<br>Spending: %{{y}}<extra></extra>"
                    ))

                # Centroids 2D
                fig_age.add_trace(go.Scatter(
                    x=df_centroids['Age'],
                    y=df_centroids['Spending Score (1-100)'],
                    mode='markers+text',
                    name='Centroids',
                    text=[f"C0{i}" for i in range(5)],
                    textposition="top center",
                    textfont=dict(family="IBM Plex Mono", size=11, color="#FFFFFF"),
                    marker=dict(size=14, symbol='diamond', color='#FFFFFF', line=dict(width=2, color='#F59E0B'))
                ))

                # Active target
                fig_age.add_trace(go.Scatter(
                    x=[age],
                    y=[spending],
                    mode='markers+text',
                    name='Active Target',
                    text=['TARGET'],
                    textposition="bottom center",
                    textfont=dict(family="IBM Plex Mono", size=12, color="#00E5FF"),
                    marker=dict(size=18, symbol='cross', color='#00E5FF', line=dict(width=3, color='#FFFFFF'))
                ))

                fig_age.update_layout(
                    xaxis=dict(title=dict(text="CUSTOMER AGE (YEARS)", font=dict(family="IBM Plex Mono", color="#E2E8F0")), gridcolor="#232833", zerolinecolor="#475569"),
                    yaxis=dict(title=dict(text="SPENDING SCORE (1-100)", font=dict(family="IBM Plex Mono", color="#E2E8F0")), gridcolor="#232833", zerolinecolor="#475569"),
                    paper_bgcolor="#0B0D11",
                    plot_bgcolor="#0F1117",
                    font=dict(color="#E2E8F0"),
                    margin=dict(l=40, r=20, t=10, b=40),
                    height=590,
                    legend=dict(bgcolor="rgba(15, 17, 23, 0.92)", bordercolor="#232833", borderwidth=1)
                )
                st.plotly_chart(fig_age, use_container_width=True)

        else:
            st.warning("Plotly is initializing...")

    with col_side:
        st.markdown('<div class="swiss-section-header">EUCLIDEAN DISTANCE MATRIX</div>', unsafe_allow_html=True)
        st.caption("Standardized Euclidean distance from input to each centroid vector.")

        for _, row in dist_df.iterrows():
            cid = int(row['Cluster'])
            cm = SWISS_METADATA[cid]
            is_active = (cid == predicted_cluster)
            border_weight = "2px solid #09090B" if is_active else "1px solid #E4E4E7"
            bg_tone = "#FFFFFF" if not is_active else "#F4F4F5"

            st.markdown(f"""
            <div style="background: {bg_tone}; border: {border_weight}; border-left: 6px solid {cm['color']}; padding: 0.65rem 0.85rem; margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.82rem; color: #09090B;">
                        CLUSTER 0{cid}
                    </span>
                    <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; font-weight: 600; color: {cm['color']};">
                        d = {row['Distance']}
                    </span>
                </div>
                <div style="font-size: 0.78rem; font-weight: 600; color: #52525B; margin-top: 0.15rem;">
                    {cm['name']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="swiss-section-header" style="margin-top: 1.5rem;">CLUSTER SIZE BREAKDOWN</div>', unsafe_allow_html=True)
        counts = df_raw['Cluster'].value_counts().sort_index()
        for cid in range(5):
            pct = (counts[cid] / 200.0) * 100
            st.markdown(f"""
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                <span style="color: {SWISS_METADATA[cid]['color']}; font-weight: 700;">C0{cid} · {SWISS_METADATA[cid]['name'][:14]}...</span>
                <span>{counts[cid]} ({pct:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(pct / 100.0)


# -----------------------------------------------------------------------------
# TAB 2: STRATEGIC PLAYBOOKS
# -----------------------------------------------------------------------------
with tab_playbook:
    st.markdown('<div class="swiss-section-header">02 // ENTERPRISE MARKETING PLAYBOOK</div>', unsafe_allow_html=True)
    st.write(
        f"Tactical commercial protocol tailored to **{meta['name']} (Cluster 0{predicted_cluster})**. "
        "Engineered for CMO and growth allocation strategies."
    )

    p_left, p_right = st.columns([1.6, 1.4])

    with p_left:
        st.markdown(f"""
        <div class="swiss-card" style="border-top: 4px solid {meta['color']};">
            <h4 style="font-family: 'Inter', sans-serif; font-weight: 800; font-size: 1.15rem; color: #09090B; margin-top: 0; text-transform: uppercase;">
                Recommended Conversion Levers
            </h4>
            <ul style="color: #3F3F46; font-size: 0.95rem; line-height: 1.9; padding-left: 1.2rem;">
        """, unsafe_allow_html=True)
        for tactic in meta['tactics']:
            st.markdown(f"<li><b>{tactic}</b></li>", unsafe_allow_html=True)
        st.markdown("""
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with p_right:
        st.markdown(f"""
        <div class="swiss-card" style="border-top: 4px solid #18181B;">
            <h4 style="font-family: 'Inter', sans-serif; font-weight: 800; font-size: 1.15rem; color: #09090B; margin-top: 0; text-transform: uppercase;">
                Cluster Benchmark Core (Real Units)
            </h4>
            <table style="width: 100%; font-size: 0.88rem; color: #27272A; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #E4E4E7;">
                    <td style="padding: 0.6rem 0; font-weight: 500;">Centroid Annual Income:</td>
                    <td style="font-family: 'IBM Plex Mono', monospace; font-weight: 700; text-align: right;">
                        ${df_centroids.loc[predicted_cluster, 'Annual Income (k$)']:.1f}k / yr
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid #E4E4E7;">
                    <td style="padding: 0.6rem 0; font-weight: 500;">Centroid Spending Score:</td>
                    <td style="font-family: 'IBM Plex Mono', monospace; font-weight: 700; text-align: right;">
                        {df_centroids.loc[predicted_cluster, 'Spending Score (1-100)']:.1f} / 100
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid #E4E4E7;">
                    <td style="padding: 0.6rem 0; font-weight: 500;">Centroid Mean Age:</td>
                    <td style="font-family: 'IBM Plex Mono', monospace; font-weight: 700; text-align: right;">
                        {df_centroids.loc[predicted_cluster, 'Age']:.1f} yrs
                    </td>
                </tr>
                <tr>
                    <td style="padding: 0.6rem 0; font-weight: 500;">Cluster Population Share:</td>
                    <td style="font-family: 'IBM Plex Mono', monospace; font-weight: 700; text-align: right; color: {meta['color']};">
                        {df_raw['Cluster'].value_counts()[predicted_cluster]} Shoppers ({df_raw['Cluster'].value_counts()[predicted_cluster]/2.0:.1f}%)
                    </td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="swiss-section-header" style="margin-top: 1.5rem;">ALL 5 COHORTS OVERVIEW</div>', unsafe_allow_html=True)
    grid_cols = st.columns(5)
    for idx, c_col in enumerate(grid_cols):
        c_info = SWISS_METADATA[idx]
        with c_col:
            st.markdown(f"""
            <div class="swiss-card" style="padding: 1rem; border-top: 5px solid {c_info['color']}; height: 100%;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 700; color: {c_info['color']};">
                    CLUSTER 0{idx}
                </div>
                <div style="font-family: 'Inter', sans-serif; font-weight: 800; font-size: 0.95rem; color: #09090B; margin: 0.3rem 0;">
                    {c_info['name']}
                </div>
                <div style="font-size: 0.78rem; color: #71717A; line-height: 1.4;">
                    {c_info['summary']}
                </div>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 3: CAMPAIGN ROI SIMULATOR
# -----------------------------------------------------------------------------
with tab_roi:
    st.markdown('<div class="swiss-section-header">03 // CAMPAIGN CAPITAL ALLOCATION & ROI SIMULATOR</div>', unsafe_allow_html=True)
    st.write("Quantitative model evaluating marketing capital deployment, customer acquisition costs (CAC), and net gross merchandise value (GMV).")

    roi_col1, roi_col2 = st.columns([1.5, 2.5])

    with roi_col1:
        st.markdown('<div class="swiss-card" style="border-top: 4px solid #18181B;">', unsafe_allow_html=True)
        st.markdown("#### 🎯 CAMPAIGN HYPOTHESIS")

        target_cohort = st.selectbox(
            "Target Cohort Partition:",
            options=range(5),
            index=predicted_cluster,
            format_func=lambda x: f"Cluster 0{x} — {SWISS_METADATA[x]['name']}"
        )

        cohort_meta = SWISS_METADATA[target_cohort]

        campaign_budget = st.slider("Campaign Budget ($ USD)", min_value=1000, max_value=50000, value=8500, step=500)
        audience_pool = st.number_input("Reachable Audience Pool", min_value=100, max_value=50000, value=4500, step=100)
        conv_rate = st.slider("Conversion Rate (%)", min_value=0.5, max_value=25.0, value=float(cohort_meta['default_conversion']), step=0.5)
        avg_basket = st.slider("Average Order Value ($ AOV)", min_value=20.0, max_value=1000.0, value=float(cohort_meta['roi_archetype_value']), step=10.0)

        st.markdown('</div>', unsafe_allow_html=True)

    with roi_col2:
        # Mathematical computation
        converted_shoppers = int(audience_pool * (conv_rate / 100.0))
        gross_revenue = converted_shoppers * avg_basket
        net_profit = gross_revenue - campaign_budget
        roi_percentage = (net_profit / campaign_budget) * 100.0 if campaign_budget > 0 else 0.0

        st.markdown(f"""
        <div class="swiss-card" style="border-top: 4px solid {cohort_meta['color']};">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; color: #71717A; text-transform: uppercase;">
                FINANCIAL OUTCOMES // {cohort_meta['name']}
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin: 1.2rem 0;">
                <div class="swiss-metric-tile">
                    <div class="swiss-metric-number">{converted_shoppers:,}</div>
                    <div class="swiss-metric-label">New Customers</div>
                </div>
                <div class="swiss-metric-tile">
                    <div class="swiss-metric-number" style="color: {cohort_meta['color']};">${gross_revenue:,.0f}</div>
                    <div class="swiss-metric-label">Gross Revenue</div>
                </div>
                <div class="swiss-metric-tile">
                    <div class="swiss-metric-number" style="color: {'#059669' if net_profit > 0 else '#FF2D20'};">${net_profit:,.0f}</div>
                    <div class="swiss-metric-label">Net Profit</div>
                </div>
                <div class="swiss-metric-tile">
                    <div class="swiss-metric-number" style="color: {'#059669' if roi_percentage > 0 else '#FF2D20'};">{roi_percentage:+.1f}%</div>
                    <div class="swiss-metric-label">Projected ROI</div>
                </div>
            </div>

            <div style="border: 1px solid #18181B; padding: 1.2rem; background: #FAFAFA; margin-top: 1rem;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 700; color: #09090B; text-transform: uppercase; margin-bottom: 0.4rem;">
                    Executive Commercial Appraisal:
                </div>
                <div style="font-size: 0.9rem; color: #3F3F46; line-height: 1.6;">
                    Deploying <b>${campaign_budget:,.0f}</b> into the <b>{cohort_meta['name']}</b> segment generates an estimated
                    <b>${gross_revenue:,.0f}</b> in gross merchandise value (GMV).
                    With a Customer Acquisition Cost (CAC) of <b>${(campaign_budget / max(1, converted_shoppers)):.2f}</b> against an AOV of <b>${avg_basket:.2f}</b>,
                    this campaign displays an estimated ROI of <b>{roi_percentage:+.1f}%</b>.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 4: EVALUATION BENCHMARKS
# -----------------------------------------------------------------------------
with tab_metrics:
    st.markdown('<div class="swiss-section-header">04 // QUANTITATIVE CLUSTERING BENCHMARKS</div>', unsafe_allow_html=True)
    st.write(
        "Rigorous geometric cluster validation metrics calculated across all 200 empirical observations. "
        "All measurements strictly reflect unsupervised data topology."
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="swiss-metric-tile" style="border-top: 3px solid #0047FF;">
            <div class="swiss-metric-number">0.4166</div>
            <div class="swiss-metric-label">Silhouette Coefficient</div>
            <div style="font-size: 0.72rem; color: #71717A; margin-top: 0.3rem;">Significantly exceeds random threshold (>0.35)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="swiss-metric-tile" style="border-top: 3px solid #059669;">
            <div class="swiss-metric-number">0.8841</div>
            <div class="swiss-metric-label">Davies-Bouldin Index</div>
            <div style="font-size: 0.72rem; color: #71717A; margin-top: 0.3rem;">Lower is better (< 1.0 confirms tight cluster separation)</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="swiss-metric-tile" style="border-top: 3px solid #FF2D20;">
            <div class="swiss-metric-number">125.74</div>
            <div class="swiss-metric-label">Calinski-Harabasz Score</div>
            <div style="font-size: 0.72rem; color: #71717A; margin-top: 0.3rem;">Elevated ratio of inter-to-intra cluster dispersion</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="swiss-metric-tile" style="border-top: 3px solid #18181B;">
            <div class="swiss-metric-number">k = 5</div>
            <div class="swiss-metric-label">Optimal Cardinality</div>
            <div style="font-size: 0.72rem; color: #71717A; margin-top: 0.3rem;">Confirmed via dual Elbow & Silhouette analysis</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="swiss-section-header" style="margin-top: 1.8rem;">CENTROIDS REFERENCE TABLE (REAL PHYSICAL UNITS)</div>', unsafe_allow_html=True)

    df_display = df_centroids.copy()
    df_display['Segment Archetype'] = [SWISS_METADATA[i]['name'] for i in range(5)]
    df_display['Shopper Count (n)'] = df_raw['Cluster'].value_counts().sort_index().values
    df_display['Share %'] = (df_display['Shopper Count (n)'] / 200.0 * 100).round(1).astype(str) + "%"
    df_display = df_display[['Cluster', 'Segment Archetype', 'Shopper Count (n)', 'Share %', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

    st.dataframe(df_display.round(2), use_container_width=True, hide_index=True)

    st.markdown("""
    > [!NOTE]
    > **Unsupervised Evaluation Standard:** Because unsupervised learning discovers latent structure rather than predicting historical labels, 
    > no test split or accuracy metric applies. Validation is established through silhouette separation ($0.4166$) and business interpretability.
    """)

# -----------------------------------------------------------------------------
# FOOTER // SWISS MINIMAL
# -----------------------------------------------------------------------------
st.markdown("---")
f1, f2 = st.columns([2, 1])
with f1:
    st.caption("Arithmatrix Virtual Internship Program (AVIP 2026) · Task 3: Customer Segmentation via K-Means++ Clustering")
with f2:
    st.caption("GitHub: [rshamith777-cpu/AIML_3_CustomerSegmentation_byte](https://github.com/rshamith777-cpu/AIML_3_CustomerSegmentation_byte)")
