import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing.artifact_detection import ArtifactRemover
from src.features.feature_engineering import FeatureEngineer
from src.models.anomaly_model import VitalAnomalyDetector


def verify_phase4_deep():
    print("--- Detailed Phase 4 Verification & Plotting ---")

    # 1. Load Data
    raw_df = pd.read_csv("data/synthetic_vitals.csv")

    # 2. Clean & Engineer
    clean_df = ArtifactRemover().fit_transform(raw_df)
    feat_df = FeatureEngineer(window_size="20s").fit_transform(clean_df)

    # 3. Predict using the saved model
    detector = VitalAnomalyDetector.load_model()
    results = detector.predict(feat_df)

    feat_df["anomaly_prob"] = results["anomaly_probabilities"]
    feat_df["anomaly_flag"] = results["anomaly_flags"]

    # 4. Check distress behavior dynamically.
    # Instead of just the last 5 minutes, let's track probability *before* deterioration vs *after*.
    # Deterioration starts roughly 1/3 to 2/3 of the way through for distress patients.

    distress_patients = feat_df[feat_df["event_label"] == "distress"][
        "patient_id"
    ].unique()

    print("\n--- Model Sensitivity Check (Distress Scenarios) ---")

    total_before = []
    total_after = []

    for pid in distress_patients:
        p_df = feat_df[feat_df["patient_id"] == pid]

        # Split into first 10 mins (always normal) vs last 10 mins (usually heavy distress)
        # Using row count since 1 row = 1 second
        first_10_mins = p_df.head(600)["anomaly_prob"].mean()
        last_10_mins = p_df.tail(600)["anomaly_prob"].mean()

        total_before.append(first_10_mins)
        total_after.append(last_10_mins)

        print(
            f"Patient {pid}: First 10m Prob = {first_10_mins:.3f} | Last 10m Prob = {last_10_mins:.3f}"
        )

    avg_before = np.mean(total_before)
    avg_after = np.mean(total_after)

    print(f"\nAverage Anomaly Probability BEFORE distress: {avg_before:.3f}")
    print(f"Average Anomaly Probability DRIUNG distress: {avg_after:.3f}")

    if avg_after > avg_before * 2:
        print(
            "\nVerification Passed: The model exhibits Strong discrimination finding distress anomalies."
        )
    elif avg_after > avg_before:
        print(
            "\nVerification Weak: The model detects distress, but the margin isn't huge. The Isolation Forest might be too insensitive."
        )
    else:
        print("\nVerification Failed: The model did not score distress periods higher.")

    # Check max threshold breach
    max_distress = max(total_after)
    print(f"Peak distress average probability across 10m: {max_distress:.3f}")


if __name__ == "__main__":
    verify_phase4_deep()
