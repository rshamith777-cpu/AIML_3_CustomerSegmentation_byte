#!/usr/bin/env python3
"""
Customer Segmentation — Production CLI Inference Interface
Repo: AIML_3_CustomerSegmentation_byte
Project: K-Means Customer Segmentation & Behavioral Profiling (AVIP 2026 Task 3)

Usage:
    python src/assign_cluster.py --age 35 --income 80 --spending 75
"""

import sys
import os
import argparse
import joblib
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, 'models', 'kmeans_model.joblib')
SCALER_PATH = os.path.join(REPO_ROOT, 'models', 'scaler.joblib')

SEGMENT_NAMES = {
    0: 'Frugal (Low Income, Low Spending)',
    1: 'Impulsive (Young, High Spending)',
    2: 'VIP (High Income, High Spending)',
    3: 'Careful Wealthy (High Income, Low Spending)',
    4: 'Standard (Moderate Income, Moderate Spending)'
}

SEGMENT_DESCRIPTIONS = {
    0: 'Cost-conscious older consumers. Responsive to essential discount offers and high utility promotions.',
    1: 'Young demographic with high spending propensity despite moderate earnings. Driven by trends and social commerce.',
    2: 'Top-tier commercial segment with high discretionary income and peak spending. Premium loyalty and concierge priority.',
    3: 'High disposable income but conservative spending behavior. Prime acquisition target for high-ROI premium upsells.',
    4: 'Mid-market older shoppers maintaining balanced expenditure. Reliable baseline revenue with family package affinity.'
}


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("=" * 65, file=sys.stderr)
        print("ARTIFACT NOTICE: Model or Scaler not found in models/.", file=sys.stderr)
        print("Please run notebooks/customer_segmentation.ipynb to generate artifacts.", file=sys.stderr)
        print("=" * 65, file=sys.stderr)
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def assign_customer(age: float, income: float, spending: float):
    model, scaler = load_artifacts()

    import pandas as pd
    features = pd.DataFrame(
        [[age, income, spending]],
        columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    )
    features_scaled = scaler.transform(features)

    # Predict nearest cluster centroid
    cluster_id = int(model.predict(features_scaled)[0])
    segment_name = SEGMENT_NAMES.get(cluster_id, f"Cluster {cluster_id}")
    description = SEGMENT_DESCRIPTIONS.get(cluster_id, "")

    # Calculate Euclidean distance to predicted centroid in scaled space
    centroid_scaled = model.cluster_centers_[cluster_id]
    distance = float(np.linalg.norm(features_scaled - centroid_scaled))

    return cluster_id, segment_name, description, distance


def main():
    parser = argparse.ArgumentParser(
        description="Assign a new mall customer to a behavioral cluster using trained K-Means model."
    )
    parser.add_argument('--age', type=float, required=True, help="Customer age in years (e.g. 35)")
    parser.add_argument('--income', type=float, required=True, help="Annual income in k$ (e.g. 80)")
    parser.add_argument('--spending', type=float, required=True, help="Spending score from 1 to 100 (e.g. 75)")

    args = parser.parse_args()

    cluster_id, segment_name, description, distance = assign_customer(args.age, args.income, args.spending)

    print("-" * 65)
    print("CUSTOMER SEGMENTATION INFERENCE TELEMETRY")
    print("-" * 65)
    print(f"INPUT METRICS     : Age={args.age:.0f} yrs | Annual Income=${args.income:.1f}k | Spending Score={args.spending:.0f}/100")
    print(f"ASSIGNED CLUSTER  : Cluster {cluster_id}")
    print(f"SEGMENT ARCHETYPE : {segment_name}")
    print(f"DISTANCE TO CORE  : {distance:.4f} (Euclidean z-distance)")
    print(f"STRATEGY NOTE     : {description}")
    print("-" * 65)


if __name__ == '__main__':
    main()
