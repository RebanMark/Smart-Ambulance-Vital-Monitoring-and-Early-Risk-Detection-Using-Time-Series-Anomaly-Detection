import pandas as pd
import numpy as np


class RiskScorer:
    def __init__(self):
        """
        Risk Scoring System mixing ML anomaly detection with hard Clinical boundaries.
        Total Target Score: 0 - 100
        """
        pass

    def calculate_confidence(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates signal confidence based on motion and sensor dropouts.
        Outputs 0.0 to 1.0 per row.
        """
        confidence = np.ones(len(df))

        # 1. Motion Penalty
        # If motion is above typical ambulance background (~1.5+), confidence drops
        motion_penalty = np.clip((df["motion_signal"] - 1.5) * 0.1, 0, 0.4)
        confidence -= motion_penalty

        # 2. Sensor Dropout Penalty
        # If the features required heavy imputation recently, reduce confidence.
        # We can't see the original NaNs here (since it's cleaned), but we can infer
        # flatlining variance from imputation if needed. For now, we trust the motion signal.

        return np.clip(confidence, 0.1, 1.0)

    def score_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates hybrid risk scores and alert flags.
        Expects `heart_rate`, `spo2`, `bp_sys`, `bp_dia`, and `anomaly_prob` to exist.
        """
        out_df = df.copy()
        risk_scores = np.zeros(len(df))

        # --- ML COMPONENT (Max 60 points) ---
        if "anomaly_prob" in out_df.columns:
            # Exponential scaling to penalize high confidence anomalies heavier
            risk_scores += (out_df["anomaly_prob"] ** 2) * 60

        # --- CLINICAL RULES (Max 60 points) ---

        # Heart Rate Rules
        hr = out_df["heart_rate"]
        hr_risk = np.where(
            (hr > 130) | (hr < 40), 20, np.where((hr > 110) | (hr < 50), 10, 0)
        )
        risk_scores += hr_risk

        # SpO2 Rules
        spo2 = out_df["spo2"]
        spo2_risk = np.where(spo2 < 90, 20, np.where(spo2 < 95, 10, 0))
        risk_scores += spo2_risk

        # Blood Pressure Rules (Systolic)
        bp = out_df["bp_sys"]
        bp_risk = np.where(
            (bp < 90) | (bp > 200), 20, np.where((bp < 100) | (bp > 180), 10, 0)
        )
        risk_scores += bp_risk

        # Cap Score at 100
        out_df["risk_score"] = np.clip(risk_scores, 0, 100)

        # --- CONFIDENCE ---
        out_df["confidence_score"] = self.calculate_confidence(out_df)

        # --- ALERT TRIGGER ---
        # Alert if Risk hits 65+ OR (ML is highly convinced >0.85 AND we have clinical confirmation Risk >= 35)
        out_df["alert_flag"] = (out_df["risk_score"] >= 65) | (
            (out_df.get("anomaly_prob", 0) >= 0.85) & (out_df["risk_score"] >= 35)
        )

        return out_df


if __name__ == "__main__":
    from src.preprocessing.artifact_detection import ArtifactRemover
    from src.features.feature_engineering import FeatureEngineer
    from src.models.anomaly_model import VitalAnomalyDetector

    print("Testing Risk Scoring System...")

    # Run full pipeline to get to scoring
    raw_df = pd.read_csv("data/synthetic_vitals.csv")
    clean_df = ArtifactRemover().fit_transform(raw_df)
    feat_df = FeatureEngineer(window_size="20s").fit_transform(clean_df)

    detector = VitalAnomalyDetector.load_model()
    results = detector.predict(feat_df)
    feat_df["anomaly_prob"] = results["anomaly_probabilities"]

    scorer = RiskScorer()
    final_df = scorer.score_data(feat_df)

    # Calculate Normal Score (Average over all normal patients)
    normal_df = final_df[final_df["event_label"] == "normal"]

    # Calculate Distress Score (Average over the LAST 5 mins of distress patients)
    distress_patients = final_df[final_df["event_label"] == "distress"][
        "patient_id"
    ].unique()
    distress_scores = []
    distress_alerts = []
    for pid in distress_patients:
        p_df = final_df[final_df["patient_id"] == pid].tail(300)  # Last 5 mins
        distress_scores.extend(p_df["risk_score"].tolist())
        distress_alerts.extend(p_df["alert_flag"].tolist())

    print("\n--- Risk Score Analysis (Active Periods) ---")
    print(
        f"Normal Average Risk:               {normal_df['risk_score'].mean():.1f} / 100"
    )
    print(f"Distress Average Risk (Last 5m):   {np.mean(distress_scores):.1f} / 100")

    print(f"\nPeak Risk observed globally: {final_df['risk_score'].max():.1f}")
    print(f"Minimum Confidence hit:      {final_df['confidence_score'].min():.2f}")

    # Check Alerts
    n_alerts_normal = normal_df["alert_flag"].sum()
    n_alerts_distress = sum(distress_alerts)
    print("\n--- Alert Triggers (True Positives vs False Positives) ---")
    print(
        f"Alerts during Normal Transport:          {n_alerts_normal} / {len(normal_df)} ({n_alerts_normal / len(normal_df) * 100:.1f}%)"
    )
    print(
        f"Alerts during Clinical Distress (5m):    {n_alerts_distress} / {len(distress_scores)} ({n_alerts_distress / len(distress_scores) * 100:.1f}%)"
    )
    print("\nRisk Scoring System Operational.")
