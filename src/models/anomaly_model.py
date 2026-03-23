import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest


class VitalAnomalyDetector:
    def __init__(self, contamination=0.05, random_state=42):
        """
        Anomaly Detection Model using Isolation Forest.
        Args:
            contamination: Expected proportion of outliers (0.01 - 0.1 usually)
            random_state: Core reproducibility seed
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()

        # Initialize PyOD Isolation Forest
        self.model = IForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
            max_samples=256,  # Forces trees to be shallower, isolating anomalies better
            n_jobs=-1,  # Use all CPU cores
        )

        # Columns to ignore during training/prediction
        self.ignore_cols = ["timestamp", "patient_id", "event_label"]
        self.feature_cols_ = None  # Will store list of modeled features

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts and returns only the most predictive numerics for modeling."""
        # Use a carefully selected subset of features to avoid noise
        predictive_cols = [
            "shock_index",
            "hr_spo2_corr",
            "pulse_pressure",
            "heart_rate_slope",
            "spo2_slope",
            "heart_rate_long_slope",
            "spo2_long_slope",
            "bp_sys_long_slope",
            "heart_rate_max",
            "spo2_min",
            "bp_sys_min",
        ]

        # Verify columns exist
        cols_to_use = [c for c in predictive_cols if c in df.columns]
        return df[cols_to_use]

    def fit(self, df: pd.DataFrame):
        """
        Fits the scaler and the Isolation Forest model.
        In production, this would be fitted only on known 'normal' baseline data.
        """
        features = self._prepare_features(df)
        self.feature_cols_ = list(features.columns)

        # Scale features
        scaled_features = self.scaler.fit_transform(features)

        # Fit model
        self.model.fit(scaled_features)
        return self

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predicts anomalies for an incoming dataframe.
        Returns a dictionary containing flags, scores, and probabilities.
        """
        if self.feature_cols_ is None:
            raise ValueError("Model is not fitted yet. Call .fit() first.")

        features = df[self.feature_cols_]
        scaled_features = self.scaler.transform(features)

        # PyOD provides convenient methods for all 3 metrics
        flags = self.model.predict(scaled_features)  # 0 = normal, 1 = anomaly
        scores = self.model.decision_function(scaled_features)  # Raw anomaly score

        # PyOD's predict_proba can be uncalibrated across different distributions.
        # For a clinical application, normalizing the raw decision scores using Min-Max
        # gives a much more reliable relative probability score.
        min_score = scores.min()
        max_score = scores.max()
        if max_score > min_score:
            probs = (scores - min_score) / (max_score - min_score)
        else:
            probs = np.zeros_like(scores)

        return {
            "anomaly_flags": flags,
            "anomaly_scores": scores,
            "anomaly_probabilities": probs,
        }

    def save_model(self, dir_path="models"):
        """Serializes the scaler, feature list, and model."""
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "model": self.model,
                "feature_cols": self.feature_cols_,
            },
            os.path.join(dir_path, "vital_anomaly_model.pkl"),
        )
        print(f"Model saved to {dir_path}/vital_anomaly_model.pkl")

    @classmethod
    def load_model(cls, filepath="models/vital_anomaly_model.pkl"):
        """Deserializes a saved model."""
        data = joblib.load(filepath)

        # Reconstruct instance
        instance = cls()
        instance.scaler = data["scaler"]
        instance.model = data["model"]
        instance.feature_cols_ = data["feature_cols"]

        return instance


if __name__ == "__main__":
    from src.preprocessing.artifact_detection import ArtifactRemover
    from src.features.feature_engineering import FeatureEngineer

    print("Testing VitalAnomalyDetector...")

    # 1. Load Data
    raw_df = pd.read_csv("data/synthetic_vitals.csv")

    # 2. Clean
    clean_df = ArtifactRemover().fit_transform(raw_df)

    # 3. Engineer Features
    feat_df = FeatureEngineer(window_size="20s").fit_transform(clean_df)

    # 4. Train anomaly model
    # We set a 5% contamination rate (assumes 5% of all streaming data is dangerously anomalous)
    detector = VitalAnomalyDetector(contamination=0.05)
    print("Fitting Isolation Forest on Normal Data...")
    normal_feat_df = feat_df[feat_df["event_label"] == "normal"]
    detector.fit(normal_feat_df)

    # 5. Predict
    results = detector.predict(feat_df)
    feat_df["anomaly_prob"] = results["anomaly_probabilities"]
    feat_df["anomaly_flag"] = results["anomaly_flags"]

    # 6. Evaluate Logic
    print(
        "\nAnomaly Probability Range:",
        feat_df["anomaly_prob"].min(),
        "-",
        feat_df["anomaly_prob"].max(),
    )
    print(
        "Total Anomalies Flagged:",
        feat_df["anomaly_flag"].sum(),
        f"({(feat_df['anomaly_flag'].sum() / len(feat_df)) * 100:.1f}%)",
    )

    # Ensure distress periods have higher anomaly probabilities
    normal_mean = feat_df[feat_df["event_label"] == "normal"]["anomaly_prob"].mean()
    artifact_mean = feat_df[feat_df["event_label"] == "artifact"]["anomaly_prob"].mean()

    # For distress, evaluate the LAST 5 minutes where deterioration is actually happening
    # (since distress patients start normal and degrade over time)
    distress_patients = feat_df[feat_df["event_label"] == "distress"][
        "patient_id"
    ].unique()
    distress_probs = []
    for pid in distress_patients:
        p_df = feat_df[feat_df["patient_id"] == pid]
        # Get the last 300 seconds (5 mins) of this patient
        distress_probs.extend(p_df.tail(300)["anomaly_prob"].tolist())

    distress_mean = np.mean(distress_probs)

    print(f"\nAverage Anomaly Probability by Scenario:")
    print(f"Normal Transport (All):        {normal_mean:.3f}")
    print(f"Artifact Setup (All):          {artifact_mean:.3f}")
    print(f"Clinical Distress (Last 5m):   {distress_mean:.3f}")

    if distress_mean > normal_mean * 1.5:  # Should be significantly higher
        print(
            "\nSuccess: Model correctly identifies true physiological 'Distress' as highly anomalous!"
        )
    else:
        print("\nWarning: Model logic may still be flawed. Distinction is too small.")

    # Test saving
    detector.save_model()
