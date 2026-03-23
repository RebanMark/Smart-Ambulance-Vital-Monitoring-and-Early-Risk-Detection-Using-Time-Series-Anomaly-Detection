import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import os


class Evaluator:
    def __init__(self, df: pd.DataFrame, output_dir: str = "eval_results"):
        """
        Calculates Phase 6 evaluation metrics and plots results.
        Requires `event_label`, `alert_flag`, `risk_score` in the dataframe.
        """
        self.df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df["timestamp"]):
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        # Determine Ground Truth based on event_label
        self.df["true_alert"] = self.df["event_label"].apply(
            lambda x: 1 if x == "distress" else 0
        )
        self.df["pred_alert"] = self.df["alert_flag"].astype(int)

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def calculate_metrics(self):
        """
        Computes precision, recall, false alert rate, and average latency.
        """
        true_labels = self.df["true_alert"]
        pred_labels = self.df["pred_alert"]

        # Basic Metrics
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)

        # False Alert Rate (FAR): Alerts during "normal" time
        normal_df = self.df[self.df["true_alert"] == 0]
        false_alerts = normal_df["pred_alert"].sum()
        normal_duration_hours = len(normal_df) / 3600  # 1 sec per row
        far = false_alerts / normal_duration_hours if normal_duration_hours > 0 else 0

        # Alert Latency: Time difference between start of distress and first correct alert
        latencies = []
        for pid in self.df["patient_id"].unique():
            p_df = self.df[self.df["patient_id"] == pid]
            distress_starts = p_df[p_df["true_alert"] == 1]
            if not distress_starts.empty:
                start_time = distress_starts["timestamp"].iloc[0]
                alerts = distress_starts[distress_starts["pred_alert"] == 1]
                if not alerts.empty:
                    first_alert_time = alerts["timestamp"].iloc[0]
                    latency = (first_alert_time - start_time).total_seconds()
                    latencies.append(latency)

        avg_latency = np.mean(latencies) if latencies else float("nan")

        return {
            "Precision": precision,
            "Recall": recall,
            "False Alert Rate (per hour normal transport)": far,
            "Average Alert Latency (seconds)": avg_latency,
        }

    def plot_confusion_matrix(self):
        """Generates confusion matrix visualization."""
        cm = confusion_matrix(self.df["true_alert"], self.df["pred_alert"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Alert"],
            yticklabels=["Normal", "Distress"],
        )
        plt.xlabel("Predicted System State")
        plt.ylabel("True Patient State")
        plt.title("Phase 6: Confusion Matrix")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_alert_timeline(self, target_patient_id: int):
        """Generates an alert timeline for a specific patient."""
        p_df = self.df[self.df["patient_id"] == target_patient_id].copy()

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot Vitals
        ax1.plot(
            p_df["timestamp"],
            p_df["heart_rate"],
            label="Heart Rate",
            color="blue",
            alpha=0.6,
        )
        ax1.plot(
            p_df["timestamp"], p_df["spo2"], label="SpO2", color="green", alpha=0.6
        )
        ax1.plot(
            p_df["timestamp"],
            p_df["bp_sys"],
            label="Systolic BP",
            color="purple",
            alpha=0.6,
        )

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Vitals")
        ax1.legend(loc="upper left")

        # Highlight distress periods
        distress_times = p_df[p_df["true_alert"] == 1]["timestamp"]
        if not distress_times.empty:
            ax1.axvspan(
                distress_times.iloc[0],
                distress_times.iloc[-1],
                color="red",
                alpha=0.1,
                label="Actual Distress Window",
            )

        # Overlay Alerts
        alerts = p_df[p_df["pred_alert"] == 1]
        ax1.scatter(
            alerts["timestamp"],
            [150] * len(alerts),
            color="red",
            marker="x",
            label="System Alert Triggers",
        )

        plt.title(f"Alert Timeline for Patient {target_patient_id}")
        plt.tight_layout()
        save_path = os.path.join(
            self.output_dir, f"timeline_patient_{target_patient_id}.png"
        )
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_risk_score_trend(self, target_patient_id: int):
        """Generates risk score trend for a specific patient."""
        p_df = self.df[self.df["patient_id"] == target_patient_id].copy()

        plt.figure(figsize=(12, 6))
        plt.plot(
            p_df["timestamp"],
            p_df["risk_score"],
            color="darkorange",
            linewidth=2,
            label="Risk Score",
        )

        # Highlight distress periods
        distress_times = p_df[p_df["true_alert"] == 1]["timestamp"]
        if not distress_times.empty:
            plt.axvspan(
                distress_times.iloc[0],
                distress_times.iloc[-1],
                color="red",
                alpha=0.1,
                label="Actual Distress Window",
            )

        plt.axhline(65, color="red", linestyle="--", label="Alert Threshold (65)")

        plt.xlabel("Time")
        plt.ylabel("Risk Score (0-100)")
        plt.title(f"Risk Score Trend for Patient {target_patient_id}")
        plt.legend(loc="upper left")
        plt.tight_layout()
        save_path = os.path.join(
            self.output_dir, f"risk_score_patient_{target_patient_id}.png"
        )
        plt.savefig(save_path)
        plt.close()
        return save_path


if __name__ == "__main__":
    from src.preprocessing.artifact_detection import ArtifactRemover
    from src.features.feature_engineering import FeatureEngineer
    from src.models.anomaly_model import VitalAnomalyDetector
    from src.models.risk_scoring import RiskScorer

    print("Running Pipeline to generate Evaluation Metrics...")

    # 1. Full Pipeline Run
    raw_df = pd.read_csv("data/synthetic_vitals.csv")
    clean_df = ArtifactRemover().fit_transform(raw_df)
    feat_df = FeatureEngineer(window_size="20s").fit_transform(clean_df)

    detector = VitalAnomalyDetector.load_model()
    results = detector.predict(feat_df)
    feat_df["anomaly_prob"] = results["anomaly_probabilities"]

    scorer = RiskScorer()
    final_df = scorer.score_data(feat_df)

    # 2. Evaluate
    evaluator = Evaluator(final_df, output_dir="eval_results")

    metrics = evaluator.calculate_metrics()
    print("\n--- PHASE 6: SYSTEM EVALUATION METRICS ---")
    for k, v in metrics.items():
        if "Rate" in k or "Latency" in k:
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v:.4f}")

    # 3. Generate Visualizations
    cm_path = evaluator.plot_confusion_matrix()
    print(f"\nSaved Confusion Matrix to: {cm_path}")

    # Get a patient with distress
    distress_patients = final_df[final_df["event_label"] == "distress"][
        "patient_id"
    ].unique()
    if len(distress_patients) > 0:
        target_pid = distress_patients[0]
        t_path = evaluator.plot_alert_timeline(target_pid)
        r_path = evaluator.plot_risk_score_trend(target_pid)
        print(f"Saved Timeline for Patient {target_pid} to: {t_path}")
        print(f"Saved Risk Trend for Patient {target_pid} to: {r_path}")

    print("\nPhase 6 Evaluation Complete!")
