import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Customer Segmentation", page_icon="🛒", layout="centered")
st.title("🛒 Customer Segment Predictor")
st.caption("K-Means Clustering · AVIP 2026")

@st.cache_resource
def load_artifacts():
    km = joblib.load("models/kmeans_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    return km, scaler

km, scaler = load_artifacts()

SEGMENT_NAMES = {
    0: "Frugal (Low Income, Low Spending)",
    1: "Impulsive (Young, High Spending)",
    2: "VIP (High Income, High Spending)",
    3: "Careful Wealthy (High Income, Low Spending)",
    4: "Standard (Moderate Income, Moderate Spending)"
}

st.subheader("Enter Customer Details")
age = st.slider("Age", 18, 70, 35)
income = st.slider("Annual Income (k$)", 15, 140, 60)
spending = st.slider("Spending Score (1-100)", 1, 99, 50)

if st.button("Assign Segment", type="primary"):
    X = pd.DataFrame([[age, income, spending]], columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
    X_scaled = scaler.transform(X)
    cluster = int(km.predict(X_scaled)[0])
    name = SEGMENT_NAMES.get(cluster, f"Cluster {cluster}")

    st.markdown(f"### Segment: **{name}** (Cluster {cluster})")

    # Show all centroids for context in real units
    centroids = scaler.inverse_transform(km.cluster_centers_)
    df_cent = pd.DataFrame(centroids.round(1),
                           columns=["Age", "Income (k$)", "Spending"])
    df_cent["Segment"] = [SEGMENT_NAMES.get(i, str(i)) for i in range(5)]
    df_cent.index.name = "Cluster"
    st.subheader("Cluster Centroids Reference (Real Units)")
    st.dataframe(df_cent)
