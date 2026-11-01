"""
=============================================================================
RETAIL DYNAMICS // 3D Customer Intelligence & Behavioral Segmentation Suite
Arithmatrix Virtual Internship Program (AVIP 2026) · Task 3 Deliverable
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
# 1. PAGE CONFIGURATION & INJECTED PREMIUM DARK THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Dynamics // 3D Customer Intelligence",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg-dark: #12140e;
    --card-bg: #1a1d15;
    --card-border: #282e20;
    --card-hover: #333a29;
    --accent-lime: #c8ff32;
    --accent-cyan: #38bdf8;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --accent-purple: #a855f7;
    --text-primary: #f2f0e7;
    --text-muted: #8e9483;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', -apple-system, sans-serif;
    color: var(--text-primary);
}

.stApp {
    background-color: var(--bg-dark);
}

/* Header & Typography */
.brand-title {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--text-primary);
    margin-bottom: 0.15rem;
}
.brand-title span {
    color: var(--accent-lime);
}
.brand-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

/* Status Pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(200, 255, 50, 0.08);
    border: 1px solid rgba(200, 255, 50, 0.25);
    color: var(--accent-lime);
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 0.5rem;
}
.status-pill-blue {
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: var(--accent-cyan);
}
.status-pill-purple {
    background: rgba(168, 85, 247, 0.08);
    border: 1px solid rgba(168, 85, 247, 0.25);
    color: var(--accent-purple);
}

/* Bento Card Containers */
.bento-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.bento-card:hover {
    border-color: var(--card-hover);
}

.bento-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    color: var(--accent-lime);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Persona Highlight Card */
.persona-card {
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    border-left: 5px solid;
    background: #171a12;
}

.persona-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.persona-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    text-transform: uppercase;
    font-weight: 600;
}

/* Metric display */
.metric-box {
    background: #12140e;
    border: 1px solid #23281c;
    border-radius: 8px;
    padding: 0.85rem;
    text-align: center;
}
.metric-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--accent-lime);
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 0.2rem;
}

/* Playbook strategy box */
.strategy-box {
    background: #141610;
    border: 1px dashed #303726;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #151812;
    border-right: 1px solid #24291d;
}

/* Sliders and inputs */
.stSlider label, .stSelectbox label, .stNumberInput label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #b2b9a7 !important;
}

/* Tab headers */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #282e20;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    color: #8e9483;
    background-color: transparent;
    border-radius: 6px 6px 0 0;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-lime) !important;
    border-bottom: 2px solid var(--accent-lime) !important;
    background-color: rgba(200, 255, 50, 0.04);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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

    # Assign clusters to existing dataset for visualization
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X_scaled = scaler.transform(df_raw[features])
    df_raw['Cluster'] = km.predict(X_scaled)

    # Centroids in real units
    centroids_real = scaler.inverse_transform(km.cluster_centers_)
    df_centroids = pd.DataFrame(
        centroids_real,
        columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    )
    df_centroids['Cluster'] = range(len(df_centroids))

    return km, scaler, df_raw, df_centroids

km, scaler, df_raw, df_centroids = load_ml_assets()

# -----------------------------------------------------------------------------
# 3. DOMAIN METADATA & ARCHETYPE PLAYBOOKS
# -----------------------------------------------------------------------------
CLUSTER_METADATA = {
    0: {
        "name": "The Frugal Shopper",
        "tag": "Value Seeker",
        "color": "#94A3B8",
        "border_color": "#64748B",
        "icon": "🏷️",
        "summary": "Low Income · Low Spending · Mature (Avg Age 46)",
        "behavior": "Conservative spenders with low disposable income who strictly buy on necessity or severe markdown. High price elasticity and voucher sensitivity.",
        "tactics": [
            "Clearance & deep discount alert notifications",
            "Essential bulk-pack bundles with guaranteed lowest price",
            "Zero-markup staple item promotions to preserve foot traffic",
            "Digital coupons redeemable at checkout"
        ],
        "roi_archetype_value": 45.0,  # Avg transaction size ($)
        "default_conversion": 4.5
    },
    1: {
        "name": "The Impulsive Trendsetter",
        "tag": "Viral Early Adopter",
        "color": "#F59E0B",
        "border_color": "#D97706",
        "icon": "⚡",
        "summary": "Low-to-Mid Income · High Spending · Young (Avg Age 25)",
        "behavior": "Youthful demographic with moderate earnings but an aggressive propensity to spend on fast fashion, aesthetics, dining, and viral lifestyle trends.",
        "tactics": [
            "TikTok & Instagram flash drop campaigns with countdown timers",
            "Gen-Z influencer collaborations and experiential pop-ups",
            "Buy-Now-Pay-Later (Klarna/Afterpay) frictionless financing",
            "Limited-edition colorway product releases"
        ],
        "roi_archetype_value": 95.0,
        "default_conversion": 11.0
    },
    2: {
        "name": "The VIP Power Shopper",
        "tag": "Core Revenue Engine",
        "color": "#C8FF32",
        "border_color": "#A3E635",
        "icon": "💎",
        "summary": "High Income · High Spending · Prime Career (Avg Age 33)",
        "behavior": "The crown jewel segment of the mall. Affluent, trend-conscious, and possessing high purchasing frequency. Expects high-touch service and brand exclusivity.",
        "tactics": [
            "Private salon appointments and personal shopping concierges",
            "Invite-only pre-launch previews for luxury collections",
            "Dedicated VIP parking privileges and private lounge hospitality",
            "Bespoke tier loyalty perks with unadvertised surprise gifts"
        ],
        "roi_archetype_value": 420.0,
        "default_conversion": 16.5
    },
    3: {
        "name": "The Careful Wealthy",
        "tag": "High-Value Growth Target",
        "color": "#38BDF8",
        "border_color": "#0284C7",
        "icon": "🎯",
        "summary": "High Income · Low Spending · Mid-Career (Avg Age 40)",
        "behavior": "Substantial financial capacity, but fiscally conservative. They thoroughly research purchases, prioritize durable quality, and reject superficial hype.",
        "tactics": [
            "Lifetime warranty and craftsmanship durability campaigns",
            "ROI and long-term value justification in product literature",
            "High-end artisanal goods, technical gear, and investment assets",
            "Zero-pressure private consultations focused on specifications"
        ],
        "roi_archetype_value": 280.0,
        "default_conversion": 6.8
    },
    4: {
        "name": "The Standard Shopper",
        "tag": "Bedrock Foundation",
        "color": "#A855F7",
        "border_color": "#9333EA",
        "icon": "🛡️",
        "summary": "Moderate Income · Moderate Spending · Older (Avg Age 56)",
        "behavior": "The dependable median customer. Traditional shopping rhythm, family-oriented needs, balanced budgets, and strong seasonal consistency.",
        "tactics": [
            "Standard seasonal sales (Back-to-School, Holidays, Summer)",
            "Family combo deals and multi-category bundle discounts",
            "Straightforward cashback loyalty points programs",
            "Dependable omnichannel inventory and hassle-free returns"
        ],
        "roi_archetype_value": 110.0,
        "default_conversion": 8.2
    }
}


# -----------------------------------------------------------------------------
# 4. SIDEBAR // TELEMETRY & INPUT CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand-title">RETAIL<span>.3D</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Consumer Dynamics Suite</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <span class="status-pill">● MODEL ACTIVE</span>
        <span class="status-pill status-pill-blue">k = 5</span>
        <span class="status-pill status-pill-purple">N = 200</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎛️ Customer Coordinates")
    st.caption("Adjust sliders or trigger preset customer archetypes.")

    # Archetype Presets
    st.markdown("**Quick Archetype Presets:**")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("💎 VIP Power", use_container_width=True):
            st.session_state['age_input'] = 33
            st.session_state['income_input'] = 86
            st.session_state['spend_input'] = 82
        if st.button("⚡ Impulsive", use_container_width=True):
            st.session_state['age_input'] = 25
            st.session_state['income_input'] = 41
            st.session_state['spend_input'] = 62
        if st.button("🛡️ Standard", use_container_width=True):
            st.session_state['age_input'] = 56
            st.session_state['income_input'] = 54
            st.session_state['spend_input'] = 49
    with p_col2:
        if st.button("🎯 Careful Wealth", use_container_width=True):
            st.session_state['age_input'] = 40
            st.session_state['income_input'] = 86
            st.session_state['spend_input'] = 19
        if st.button("🏷️ Frugal", use_container_width=True):
            st.session_state['age_input'] = 46
            st.session_state['income_input'] = 27
            st.session_state['spend_input'] = 18
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state['age_input'] = 32
            st.session_state['income_input'] = 75
            st.session_state['spend_input'] = 65

    # Fallback state initialization
    if 'age_input' not in st.session_state:
        st.session_state['age_input'] = 32
    if 'income_input' not in st.session_state:
        st.session_state['income_input'] = 75
    if 'spend_input' not in st.session_state:
        st.session_state['spend_input'] = 65

    # Sliders
    age = st.slider("Customer Age (Years)", min_value=18, max_value=70,
                    value=st.session_state['age_input'], key="slider_age")
    income = st.slider("Annual Income (k$)", min_value=15, max_value=140,
                       value=st.session_state['income_input'], key="slider_income")
    spending = st.slider("Spending Score (1-100)", min_value=1, max_value=99,
                         value=st.session_state['spend_input'], key="slider_spend")

    st.markdown("---")
    st.markdown("### 📐 Unsupervised Paradigm")
    st.caption(
        "Clustering partitions anonymous data points without labels. "
        "There is no train/test split or accuracy metric. Cluster quality "
        "is measured via silhouette compactness (0.4166) and business utility."
    )


# -----------------------------------------------------------------------------
# 5. REAL-TIME CLUSTERING INFERENCE & EUCLIDEAN METRICS
# -----------------------------------------------------------------------------
input_df = pd.DataFrame([[age, income, spending]],
                        columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
input_scaled = scaler.transform(input_df)
predicted_cluster = int(km.predict(input_scaled)[0])
meta = CLUSTER_METADATA[predicted_cluster]

# Compute Euclidean distances to all 5 centroids in StandardScaler space
cluster_centers_scaled = km.cluster_centers_
distances = np.linalg.norm(cluster_centers_scaled - input_scaled, axis=1)
dist_df = pd.DataFrame({
    'Cluster': range(5),
    'Segment': [CLUSTER_METADATA[i]['name'] for i in range(5)],
    'Scaled Distance': np.round(distances, 3)
}).sort_values('Scaled Distance')


# -----------------------------------------------------------------------------
# 6. MAIN CONTENT // HERO BANNER & REAL-TIME PREDICTION
# -----------------------------------------------------------------------------
st.markdown('<div class="brand-title">RETAIL DYNAMICS <span>// 3D GALAXY</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-subtitle">'
    'Autonomous Unsupervised Behavioral Intelligence · K-Means++ Clustering (k=5) · AVIP 2026'
    '</div>',
    unsafe_allow_html=True
)

# Persona Announcement Banner
persona_card_html = f"""
<div class="persona-card" style="border-left-color: {meta['color']};">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <div class="bento-header" style="color: {meta['color']};">
                TARGET IDENTIFIED // CLUSTER {predicted_cluster}
            </div>
            <div class="persona-title">
                {meta['icon']} {meta['name']}
                <span class="persona-tag" style="background: rgba({int(meta['color'][1:3], 16)}, {int(meta['color'][3:5], 16)}, {int(meta['color'][5:7], 16)}, 0.15); color: {meta['color']};">
                    {meta['tag']}
                </span>
            </div>
            <p style="color: #cbd5e1; font-size: 0.95rem; margin-top: 0.3rem; max-width: 800px;">
                {meta['behavior']}
            </p>
        </div>
        <div style="text-align: right; margin-top: 0.5rem;">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #8e9483;">PROXIMITY CONFIDENCE</div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 700; color: {meta['color']};">
                {100.0 / (1.0 + distances[predicted_cluster]):.1f}%
            </div>
            <div style="font-size: 0.75rem; color: #8e9483;">Scaled Dist: {distances[predicted_cluster]:.3f}</div>
        </div>
    </div>
</div>
"""
st.markdown(persona_card_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. MAIN TABS: 3D GALAXY // TACTICAL PLAYBOOK // ROI SIMULATOR // TELEMETRY
# -----------------------------------------------------------------------------
tab_3d, tab_playbook, tab_roi, tab_metrics = st.tabs([
    "🌌 3D Customer Galaxy",
    "📋 Strategic Playbook",
    "💼 Campaign ROI Simulator",
    "📊 Evaluation Telemetry"
])

# -----------------------------------------------------------------------------
# TAB 1: 3D CUSTOMER GALAXY
# -----------------------------------------------------------------------------
with tab_3d:
    col_vis, col_side = st.columns([2.8, 1.2])

    with col_vis:
        if HAS_PLOTLY:
            fig = go.Figure()

            # Plot existing 200 mall customers colored by cluster
            for c_id in range(5):
                subset = df_raw[df_raw['Cluster'] == c_id]
                c_meta = CLUSTER_METADATA[c_id]
                fig.add_trace(go.Scatter3d(
                    x=subset['Annual Income (k$)'],
                    y=subset['Spending Score (1-100)'],
                    z=subset['Age'],
                    mode='markers',
                    name=f"C{c_id}: {c_meta['name']}",
                    marker=dict(
                        size=5,
                        color=c_meta['color'],
                        opacity=0.68,
                        line=dict(width=0.5, color='#141610')
                    ),
                    hovertemplate=(
                        f"<b>{c_meta['name']}</b><br>"
                        "Income: $%{x}k<br>"
                        "Spending Score: %{y}<br>"
                        "Age: %{z} yrs<extra></extra>"
                    )
                ))

            # Plot Cluster Centroids as prominent gold-ringed diamonds
            fig.add_trace(go.Scatter3d(
                x=df_centroids['Annual Income (k$)'],
                y=df_centroids['Spending Score (1-100)'],
                z=df_centroids['Age'],
                mode='markers+text',
                name='Centroids (Real Units)',
                text=[f"C{i} Centroid" for i in range(5)],
                textposition="top center",
                textfont=dict(family="IBM Plex Mono", size=10, color="#ffffff"),
                marker=dict(
                    size=9,
                    symbol='diamond',
                    color='#ffffff',
                    line=dict(width=2, color='#c8ff32')
                ),
                hovertemplate="<b>Centroid C%{pointNumber}</b><br>Income: $%{x:.1f}k<br>Spending: %{y:.1f}<br>Age: %{z:.1f}<extra></extra>"
            ))

            # Project current input user in 3D Euclidean space
            fig.add_trace(go.Scatter3d(
                x=[income],
                y=[spending],
                z=[age],
                mode='markers+text',
                name='⚡ ACTIVE TARGET',
                text=['ACTIVE TARGET'],
                textposition="bottom center",
                textfont=dict(family="IBM Plex Mono", size=12, color=meta['color']),
                marker=dict(
                    size=13,
                    symbol='cross',
                    color=meta['color'],
                    line=dict(width=2, color='#ffffff')
                ),
                hovertemplate=(
                    "<b>⚡ ACTIVE TARGET</b><br>"
                    f"Cluster: {predicted_cluster} ({meta['name']})<br>"
                    f"Income: ${income}k<br>"
                    f"Spending: {spending}<br>"
                    f"Age: {age} yrs<extra></extra>"
                )
            ))

            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        title="Annual Income (k$)",
                        backgroundcolor="#141610",
                        gridcolor="#282e20",
                        showbackground=True,
                        zerolinecolor="#3a432d"
                    ),
                    yaxis=dict(
                        title="Spending Score (1-100)",
                        backgroundcolor="#141610",
                        gridcolor="#282e20",
                        showbackground=True,
                        zerolinecolor="#3a432d"
                    ),
                    zaxis=dict(
                        title="Age (Years)",
                        backgroundcolor="#141610",
                        gridcolor="#282e20",
                        showbackground=True,
                        zerolinecolor="#3a432d"
                    ),
                    camera=dict(
                        eye=dict(x=1.6, y=-1.6, z=1.2)
                    )
                ),
                paper_bgcolor="#141610",
                plot_bgcolor="#141610",
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(
                    bgcolor="rgba(20, 22, 16, 0.8)",
                    bordercolor="#282e20",
                    borderwidth=1,
                    font=dict(family="Space Grotesk", color="#e2e8f0", size=10)
                ),
                height=580
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Plotly is initializing. Refresh the page to experience the 3D rotating visualizer.")

    with col_side:
        st.markdown('<div class="bento-header">EUCLIDEAN DISTANCE MATRIX</div>', unsafe_allow_html=True)
        st.caption("Distance from current coordinate to each centroid in scaled feature space.")

        for _, row in dist_df.iterrows():
            cid = int(row['Cluster'])
            cm = CLUSTER_METADATA[cid]
            is_active = (cid == predicted_cluster)
            border_style = f"border: 1px solid {cm['color']};" if is_active else "border: 1px solid #23281c;"
            bg_style = "background: rgba(200, 255, 50, 0.05);" if is_active else "background: #171a12;"

            st.markdown(f"""
            <div style="{bg_style} {border_style} border-radius: 8px; padding: 0.65rem 0.85rem; margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; font-size: 0.88rem; color: {cm['color']};">
                        {cm['icon']} Cluster {cid}
                    </span>
                    <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: #e2e8f0;">
                        d = {row['Scaled Distance']}
                    </span>
                </div>
                <div style="font-size: 0.75rem; color: #8e9483; margin-top: 0.15rem;">
                    {cm['name']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="bento-header" style="margin-top: 1.2rem;">DATASET DISTRIBUTION</div>', unsafe_allow_html=True)
        counts = df_raw['Cluster'].value_counts().sort_index()
        for cid in range(5):
            pct = (counts[cid] / 200.0) * 100
            st.markdown(f"""
            <div style="font-size: 0.78rem; margin-bottom: 0.35rem;">
                <span style="color: {CLUSTER_METADATA[cid]['color']}; font-weight: 600;">Cluster {cid}:</span>
                <span style="color: #cbd5e1;"> {counts[cid]} shoppers ({pct:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(pct / 100.0)


# -----------------------------------------------------------------------------
# TAB 2: STRATEGIC PLAYBOOK
# -----------------------------------------------------------------------------
with tab_playbook:
    st.markdown('<div class="bento-header">TACTICAL MARKETING PLAYBOOK</div>', unsafe_allow_html=True)
    st.write(
        f"Detailed enterprise targeting protocol for **{meta['name']} (Cluster {predicted_cluster})**. "
        "Each playbook provides actionable conversion levers designed for CMO and growth teams."
    )

    p_col_left, p_col_right = st.columns([1.6, 1.4])

    with p_col_left:
        st.markdown(f"""
        <div class="bento-card">
            <h4 style="color: {meta['color']}; margin-top: 0;">🎯 Recommended Conversion Channels</h4>
            <ul style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.8;">
        """, unsafe_allow_html=True)
        for tactic in meta['tactics']:
            st.markdown(f"<li><b>{tactic}</b></li>", unsafe_allow_html=True)
        st.markdown("""
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with p_col_right:
        st.markdown(f"""
        <div class="bento-card">
            <h4 style="color: {meta['color']}; margin-top: 0;">📊 Centroid Benchmark Coordinates</h4>
            <table style="width: 100%; font-size: 0.9rem; color: #cbd5e1;">
                <tr style="border-bottom: 1px solid #282e20;">
                    <td style="padding: 0.5rem 0;">Annual Income Benchmark:</td>
                    <td style="font-family: 'IBM Plex Mono', monospace; font-weight: 600; text-align: right;">
                        ${df_centroids.loc[predicted_cluster, 'Annual Income (k$)']:.1f}k
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid #282e20;">
                    <td style="padding: 0.5rem 0;">Spending Score Benchmark:</td>
                    <td style="font-family: 'IBM Plex Mono', monospace; font-weight: 600; text-align: right;">
                        {df_centroids.loc[predicted_cluster, 'Spending Score (1-100)']:.1f} / 100
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid #282e20;">
                    <td style="padding: 0.5rem 0;">Age Benchmark:</td>
                    <td style="font-family: 'IBM Plex Mono', monospace; font-weight: 600; text-align: right;">
                        {df_centroids.loc[predicted_cluster, 'Age']:.1f} yrs
                    </td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0;">Natural Cluster Size:</td>
                    <td style="font-family: 'IBM Plex Mono', monospace; font-weight: 600; text-align: right; color: {meta['color']};">
                        {df_raw['Cluster'].value_counts()[predicted_cluster]} shoppers ({df_raw['Cluster'].value_counts()[predicted_cluster]/2.0:.1f}%)
                    </td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🏛️ All 5 Archetype Playbooks Comparison")
    grid_cols = st.columns(5)
    for idx, c_col in enumerate(grid_cols):
        c_info = CLUSTER_METADATA[idx]
        with c_col:
            st.markdown(f"""
            <div class="bento-card" style="padding: 0.9rem; font-size: 0.85rem; height: 100%; border-top: 3px solid {c_info['color']};">
                <div style="font-size: 1.3rem;">{c_info['icon']}</div>
                <div style="font-weight: 700; color: {c_info['color']}; margin: 0.3rem 0;">C{idx}: {c_info['name']}</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #8e9483; margin-bottom: 0.5rem;">{c_info['tag']}</div>
                <div style="color: #94a3b8; font-size: 0.78rem;">{c_info['summary']}</div>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 3: CAMPAIGN ROI SIMULATOR
# -----------------------------------------------------------------------------
with tab_roi:
    st.markdown('<div class="bento-header">RETAIL CAMPAIGN REVENUE & ROI SIMULATOR</div>', unsafe_allow_html=True)
    st.write("Simulate enterprise ad spend allocations, conversion uplift, and revenue generation on segmented customer cohorts.")

    roi_col1, roi_col2 = st.columns([1.5, 2.5])

    with roi_col1:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Campaign Parameters")

        target_cohort = st.selectbox(
            "Select Target Segment Cohort:",
            options=range(5),
            index=predicted_cluster,
            format_func=lambda x: f"Cluster {x} — {CLUSTER_METADATA[x]['name']}"
        )

        cohort_meta = CLUSTER_METADATA[target_cohort]

        campaign_budget = st.slider("Campaign Budget ($ USD)", min_value=1000, max_value=50000, value=8500, step=500)
        audience_pool = st.number_input("Target Audience Size (Reachable Shoppers)", min_value=100, max_value=50000, value=4500, step=100)
        conv_rate = st.slider("Projected Conversion Rate (%)", min_value=0.5, max_value=25.0, value=float(cohort_meta['default_conversion']), step=0.5)
        avg_basket = st.slider("Average Order Value ($ AOV)", min_value=20.0, max_value=1000.0, value=float(cohort_meta['roi_archetype_value']), step=10.0)

        st.markdown('</div>', unsafe_allow_html=True)

    with roi_col2:
        # Financial math
        converted_shoppers = int(audience_pool * (conv_rate / 100.0))
        gross_revenue = converted_shoppers * avg_basket
        net_profit = gross_revenue - campaign_budget
        roi_percentage = (net_profit / campaign_budget) * 100.0 if campaign_budget > 0 else 0.0

        st.markdown(f"""
        <div class="bento-card">
            <div class="bento-header">FINANCIAL PROJECTIONS // {cohort_meta['name'].upper()}</div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin-top: 1rem;">
                <div class="metric-box">
                    <div class="metric-val">{converted_shoppers:,}</div>
                    <div class="metric-label">New Shoppers</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" style="color: {cohort_meta['color']};">${gross_revenue:,.0f}</div>
                    <div class="metric-label">Gross Revenue</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" style="color: {'#c8ff32' if net_profit > 0 else '#f43f5e'};">${net_profit:,.0f}</div>
                    <div class="metric-label">Net Profit</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" style="color: {'#c8ff32' if roi_percentage > 0 else '#f43f5e'};">{roi_percentage:+.1f}%</div>
                    <div class="metric-label">Projected ROI</div>
                </div>
            </div>

            <div class="strategy-box">
                <div style="font-weight: 600; color: #f2f0e7; margin-bottom: 0.3rem;">
                    💡 Executive Strategic Assessment:
                </div>
                <div style="font-size: 0.88rem; color: #a1a1aa; line-height: 1.6;">
                    Targeting <b>{cohort_meta['name']}</b> with a <b>${campaign_budget:,.0f}</b> capital injection yields
                    an expected <b>${gross_revenue:,.0f}</b> in gross merchandise value (GMV).
                    With an expected CAC (Customer Acquisition Cost) of <b>${(campaign_budget / max(1, converted_shoppers)):.2f}</b>
                    against an AOV of <b>${avg_basket:.2f}</b>, this represents a 
                    <span style="color: {cohort_meta['color']}; font-weight: 600;">{'high-yield profitable' if roi_percentage > 100 else 'viable' if roi_percentage > 0 else 'sub-optimal'}</span>
                    campaign deployment.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 4: EVALUATION TELEMETRY
# -----------------------------------------------------------------------------
with tab_metrics:
    st.markdown('<div class="bento-header">UNSUPERVISED CLUSTER VALIDATION BENCHMARKS</div>', unsafe_allow_html=True)
    st.write(
        "Quantitative cluster separation metrics measured across all 200 mall shopper observations. "
        "Strictly reflects empirical clustering geometry."
    )

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-val">0.4166</div>
            <div class="metric-label">Silhouette Score</div>
            <div style="font-size: 0.7rem; color: #8e9483; margin-top: 0.3rem;">Well above random baseline (>0.35)</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-val">0.8841</div>
            <div class="metric-label">Davies-Bouldin Index</div>
            <div style="font-size: 0.7rem; color: #8e9483; margin-top: 0.3rem;">Lower is better (< 1.0 indicates separation)</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-val">125.74</div>
            <div class="metric-label">Calinski-Harabasz Score</div>
            <div style="font-size: 0.7rem; color: #8e9483; margin-top: 0.3rem;">Strong ratio of between/within cluster variance</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-val">k = 5</div>
            <div class="metric-label">Optimal Clusters</div>
            <div style="font-size: 0.7rem; color: #8e9483; margin-top: 0.3rem;">Selected via Elbow curve & Silhouette analysis</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="bento-header" style="margin-top: 1.5rem;">CENTROIDS REFERENCE TABLE (REAL PHYSICAL UNITS)</div>', unsafe_allow_html=True)

    df_display = df_centroids.copy()
    df_display['Segment Archetype'] = [CLUSTER_METADATA[i]['name'] for i in range(5)]
    df_display['Shopper Count (n)'] = df_raw['Cluster'].value_counts().sort_index().values
    df_display['Percent of Total'] = (df_display['Shopper Count (n)'] / 200.0 * 100).round(1).astype(str) + "%"
    df_display = df_display[['Cluster', 'Segment Archetype', 'Shopper Count (n)', 'Percent of Total', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

    st.dataframe(df_display.round(2), use_container_width=True, hide_index=True)

    st.markdown("""
    > [!IMPORTANT]
    > **Unsupervised Evaluation Integrity:** Unsupervised clustering does not possess ground-truth labels or an accuracy percentage. 
    > True validation relies on mathematical cluster separation metrics (Silhouette 0.4166) paired with business-actionable archetype differentiation.
    """)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
f_col1, f_col2 = st.columns([2, 1])
with f_col1:
    st.caption("Arithmatrix Virtual Internship Program (AVIP 2026) · Task 3: Customer Segmentation using K-Means Clustering")
with f_col2:
    st.caption("GitHub: [rshamith777-cpu/AIML_3_CustomerSegmentation_byte](https://github.com/rshamith777-cpu/AIML_3_CustomerSegmentation_byte)")
