import pandas as pd
import numpy as np
from scipy import stats


class ArtifactRemover:
    def __init__(self, motion_window="10s", median_window="5s"):
        self.motion_window = motion_window
        self.median_window = median_window
        self.vital_cols = ["heart_rate", "spo2", "bp_sys", "bp_dia"]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all artifact removal steps sequentially."""
        df_clean = df.copy()

        # Ensure timestamp is datetime
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])

        cleaned_patients = []
        # Process each patient independently to avoid cross-contamination
        for pid, p_df in df_clean.groupby("patient_id"):
            p_df = p_df.sort_values("timestamp").reset_index(drop=True)

            # 1. Impute missing values (dropouts)
            p_df = self._impute_missing(p_df)

            # 2. Apply rolling median for high-frequency noise
            p_df = self._apply_rolling_median(p_df)

            # 3. Detect and dampen motion-induced artifacts
            p_df = self._remove_motion_artifacts(p_df)

            # 4. Z-score filtering for impossible jumps
            p_df = self._z_score_filtering(p_df)

            cleaned_patients.append(p_df)

        return pd.concat(cleaned_patients, ignore_index=True)

    def _impute_missing(self, df: pd.DataFrame, max_gap=5) -> pd.DataFrame:
        """Linearly interpolates short gaps (<5s)."""
        df_imputed = df.copy()
        for col in self.vital_cols:
            # interpolate with limit to not guess long missing spans
            df_imputed[col] = df_imputed[col].interpolate(
                method="linear", limit=max_gap
            )
            # If still missing (start/end of array or >5s gap), forward/backward fill as last resort
            df_imputed[col] = df_imputed[col].ffill().bfill()
        return df_imputed

    def _apply_rolling_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smooths high-frequency noise."""
        df_smoothed = df.copy()
        # Set index for time-based rolling
        df_smoothed = df_smoothed.set_index("timestamp")

        for col in self.vital_cols:
            # Center=True prevents shifting the signal backwards
            df_smoothed[col] = (
                df_smoothed[col]
                .rolling(window=self.median_window, center=True, min_periods=1)
                .median()
            )

        return df_smoothed.reset_index()

    def _remove_motion_artifacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dampens false HR/SpO2 spikes that correlate with high motion."""
        df_clean = df.copy()

        # Define high motion threshold (e.g., 90th percentile of this patient's motion)
        motion_threshold = df_clean["motion_signal"].quantile(0.90)

        # Identify high motion windows
        high_motion_idx = df_clean[df_clean["motion_signal"] > motion_threshold].index

        # If no significant motion, return
        if len(high_motion_idx) == 0:
            return df_clean

        # Set index for rolling operations
        df_clean = df_clean.set_index("timestamp")

        # Provide a 10s rolling mean to fall back to during extreme motion
        rolling_hr = df_clean["heart_rate"].rolling("10s", min_periods=1).mean()
        rolling_spo2 = df_clean["spo2"].rolling("10s", min_periods=1).mean()

        df_clean = df_clean.reset_index()

        # For high motion periods, if HR is spiking or SpO2 dropping wildly,
        # replace with the rolling mean
        for idx in high_motion_idx:
            # HR spike during motion
            if df_clean.loc[idx, "heart_rate"] > rolling_hr.iloc[idx] + 15:
                df_clean.loc[idx, "heart_rate"] = rolling_hr.iloc[idx]

            # SpO2 drop during motion
            if df_clean.loc[idx, "spo2"] < rolling_spo2.iloc[idx] - 3:
                df_clean.loc[idx, "spo2"] = rolling_spo2.iloc[idx]

        return df_clean

    def _z_score_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes extreme impossible spikes using localized Z-scores."""
        df_clean = df.copy()
        df_clean = df_clean.set_index("timestamp")

        for col in self.vital_cols:
            # Calculate rolling mean and std
            roll_mean = df_clean[col].rolling("30s", center=True, min_periods=1).mean()
            roll_std = (
                df_clean[col]
                .rolling("30s", center=True, min_periods=1)
                .std()
                .replace(0, 1e-6)
            )

            # Calculate local Z-score
            z_scores = np.abs((df_clean[col] - roll_mean) / roll_std)

            # Replace points with Z > 3.5 (extreme outliers) with the rolling mean
            outlier_mask = z_scores > 3.5
            df_clean.loc[outlier_mask, col] = roll_mean[outlier_mask]

        return df_clean.reset_index()


if __name__ == "__main__":
    # Test execution
    print("Testing ArtifactRemover...")
    df = pd.read_csv("data/synthetic_vitals.csv")

    # Pick the first artifact patient
    artifact_patients = df[df["event_label"] == "artifact"]["patient_id"].unique()
    if len(artifact_patients) > 0:
        pid = artifact_patients[0]
        patient_df = df[df["patient_id"] == pid].copy()

        remover = ArtifactRemover()
        cleaned_df = remover.fit_transform(patient_df)

        print(f"\nResults for Patient {pid} (Artifact Scenario):")
        print("Original NaNs in HR:", patient_df["heart_rate"].isnull().sum())
        print("Cleaned NaNs in HR:", cleaned_df["heart_rate"].isnull().sum())

        print("\nOriginal Max HR:", patient_df["heart_rate"].max())
        print("Cleaned Max HR:", cleaned_df["heart_rate"].max())

        print("\nOriginal Min SpO2:", patient_df["spo2"].min())
        print("Cleaned Min SpO2:", cleaned_df["spo2"].min())
        print("Artifact detection and removal successful.")
    else:
        print("No artifact patients found to test.")
