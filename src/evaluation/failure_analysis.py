import pandas as pd
import numpy as np


def analyze_failures(df: pd.DataFrame):
    """
    Analyzes specific failure cases from the evaluated dataframe.
    Looks for:
    1. False Negatives (Missed deterioration)
    2. False Positives (Normal but alerted)
    """
    print("\n--- Phase 7: Failure Analysis ---")

    # 1. False Negatives (True deterioration missed)
    fns = df[(df["event_label"] == "distress") & (df["alert_flag"] == False)]
    if not fns.empty:
        # Get one continuous block of FNs to explain
        target_pid = fns["patient_id"].iloc[0]
        p_fns = fns[fns["patient_id"] == target_pid]

        print(f"\n[FAILURE CASE 1] True Deterioration Missed (False Negative)")
        print(
            f"Patient {target_pid} was in distress, but 0 alerts were fired at this moment."
        )
        print(f"Sample data during failure:")
        sample = p_fns.iloc[
            len(p_fns) // 2
        ]  # take a point in the middle of a failure block
        print(
            f"  HR: {sample['heart_rate']:.1f} | SpO2: {sample['spo2']:.1f} | BP Sys: {sample['bp_sys']:.1f}"
        )
        print(f"  ML Anomaly Prob: {sample.get('anomaly_prob', 0):.2f}")
        print(f"  Risk Score: {sample['risk_score']:.1f} (Threshold 65)")
        print(f"  Motion Signal: {sample['motion_signal']:.2f}")

    # 2. False Positives (Normal state flagged as alert)
    fps = df[(df["event_label"] == "normal") & (df["alert_flag"] == True)]
    if not fps.empty:
        target_pid = fps["patient_id"].iloc[0]
        p_fps = fps[fps["patient_id"] == target_pid]

        print(
            f"\n[FAILURE CASE 2] False Alert During Normal Transport (False Positive)"
        )
        print(f"Patient {target_pid} was perfectly normal, but an alert fired.")
        print(f"Sample data during failure:")
        sample = p_fps.iloc[0]
        print(
            f"  HR: {sample['heart_rate']:.1f} | SpO2: {sample['spo2']:.1f} | BP Sys: {sample['bp_sys']:.1f}"
        )
        print(f"  ML Anomaly Prob: {sample.get('anomaly_prob', 0):.2f}")
        print(f"  Risk Score: {sample['risk_score']:.1f} (Threshold 65)")
        print(f"  Motion Signal: {sample['motion_signal']:.2f}")

    # 3. High Latency (Caught distress, but too late)
    for pid in df["patient_id"].unique():
        p_df = df[df["patient_id"] == pid]
        distress = p_df[p_df["event_label"] == "distress"]
        if not distress.empty:
            start_time = pd.to_datetime(distress["timestamp"].iloc[0])
            alerts = distress[distress["alert_flag"] == True]
            if not alerts.empty:
                first_alert_time = pd.to_datetime(alerts["timestamp"].iloc[0])
                latency = (first_alert_time - start_time).total_seconds()

                if latency > 300:  # 5 minutes latency is considered high
                    print(f"\n[FAILURE CASE 3] High Alert Latency")
                    print(
                        f"Patient {pid} entered distress, but it took {latency} seconds to trigger an alert."
                    )

                    # Why? Let's look at the vitals at Start vs Alert
                    start = distress.iloc[0]
                    alert_pt = alerts.iloc[0]
                    print("  At Distress Start (No Alert):")
                    print(
                        f"    HR: {start['heart_rate']:.1f} | SpO2: {start['spo2']:.1f} | BP: {start['bp_sys']:.1f} | Risk: {start['risk_score']:.1f}"
                    )
                    print("  At First Alert:")
                    print(
                        f"    HR: {alert_pt['heart_rate']:.1f} | SpO2: {alert_pt['spo2']:.1f} | BP: {alert_pt['bp_sys']:.1f} | Risk: {alert_pt['risk_score']:.1f}"
                    )
                    break  # just show one


if __name__ == "__main__":
    from src.preprocessing.artifact_detection import ArtifactRemover
    from src.features.feature_engineering import FeatureEngineer
    from src.models.anomaly_model import VitalAnomalyDetector
    from src.models.risk_scoring import RiskScorer

    # Run Pipeline
    raw_df = pd.read_csv("data/synthetic_vitals.csv")
    clean_df = ArtifactRemover().fit_transform(raw_df)
    feat_df = FeatureEngineer(window_size="20s").fit_transform(clean_df)

    detector = VitalAnomalyDetector.load_model()
    results = detector.predict(feat_df)
    feat_df["anomaly_prob"] = results["anomaly_probabilities"]

    scorer = RiskScorer()
    final_df = scorer.score_data(feat_df)

    analyze_failures(final_df)
