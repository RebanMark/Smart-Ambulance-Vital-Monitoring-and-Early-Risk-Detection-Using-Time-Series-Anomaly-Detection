import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, window_size="20s"):
        self.window_size = window_size
        self.vital_cols = ["heart_rate", "spo2", "bp_sys", "bp_dia"]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates the feature engineering pipeline.
        Processes each patient independently to prevent temporal leakage.
        """
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # We need data sorted by time for rolling windows
        df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

        # Process per patient using groupby
        processed_dfs = []
        for pid, patient_df in df.groupby("patient_id"):
            # Set time index for rolling operations
            p_df = patient_df.set_index("timestamp").copy()

            p_df = self._generate_stats_features(p_df)
            p_df = self._generate_cross_features(p_df)
            p_df = self._generate_trend_features(p_df)
            p_df = self._generate_long_trend_features(p_df)

            # Since rolling windows require history, drop the first 'window_size' rows where features are NaN
            # `window_size` is a string like '20s', '20S' from pandas <3.0 is equivalent to just timedelta(seconds=20)
            window_timedelta = pd.to_timedelta(self.window_size.lower())
            start_threshold = p_df.index[0] + window_timedelta

            # Drop the warm-up period
            p_df = p_df[p_df.index >= start_threshold]

            processed_dfs.append(p_df.reset_index())

        final_df = pd.concat(processed_dfs, ignore_index=True)
        # Drop rows that still have NaNs in the engineered features (edge cases)
        final_df = final_df.dropna().reset_index(drop=True)
        return final_df

    def _generate_stats_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates mean, std, min, max over the sliding window."""
        df_feat = df.copy()

        for col in self.vital_cols:
            roller = df_feat[col].rolling(self.window_size, min_periods=2)

            df_feat[f"{col}_mean"] = roller.mean()
            df_feat[f"{col}_std"] = roller.std().fillna(0)
            df_feat[f"{col}_min"] = roller.min()
            df_feat[f"{col}_max"] = roller.max()

        return df_feat

    def _generate_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates Shock Index, Pulse Pressure, and HR-SpO2 Correlation."""
        df_feat = df.copy()

        # 1. Shock Index: HR / Systolic BP
        # High score (> 0.9) indicates potential shock / decompensation
        df_feat["shock_index"] = df_feat["heart_rate"] / df_feat["bp_sys"].replace(
            0, 1e-6
        )

        # 2. Pulse Pressure: Systolic - Diastolic
        df_feat["pulse_pressure"] = df_feat["bp_sys"] - df_feat["bp_dia"]

        # 3. Rolling Correlation: HR vs SpO2
        # Negative correlation (HR spiking while SpO2 drops) is a strong anomaly signal
        df_feat["hr_spo2_corr"] = (
            df_feat["heart_rate"]
            .rolling(self.window_size, min_periods=2)
            .corr(df_feat["spo2"])
        )
        # Fill NaN correlations with 0 (no correlation) when variance is 0
        df_feat["hr_spo2_corr"] = df_feat["hr_spo2_corr"].fillna(0)

        return df_feat

    def _generate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimates the rate of change (slope) using basic differences.
        For instantaneous monitoring, comparing current value to the start of the window is efficient.
        """
        df_feat = df.copy()

        # We need the value at the start of the running window.
        # Since our frequency is 1Hz, shifting by integer window_seconds works perfectly.
        window_seconds = int(pd.to_timedelta(self.window_size.lower()).total_seconds())

        for col in self.vital_cols:
            # Shift gets the value 'window_seconds' ago
            window_start_val = df_feat[col].shift(window_seconds)

            # Slope = (Current - Start) / Time
            df_feat[f"{col}_slope"] = (df_feat[col] - window_start_val) / window_seconds

        return df_feat

    def _generate_long_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates a 5-minute (300-second) rolling slope to catch slow deterioration.
        """
        df_feat = df.copy()

        # 5 minute window for long-term physiological trends
        window_seconds = 300

        for col in self.vital_cols:
            window_start_val = df_feat[col].shift(window_seconds)

            # Slope over 5 minutes
            slope_col = f"{col}_long_slope"
            df_feat[slope_col] = (df_feat[col] - window_start_val) / window_seconds

            # Since the first 5 minutes will legitimately be NaN,
            # we fill them with 0 (no slope) instead of dropping 5 mins of data
            df_feat[slope_col] = df_feat[slope_col].fillna(0)

        return df_feat


if __name__ == "__main__":
    print("Testing FeatureEngineer...")
    from src.preprocessing.artifact_detection import ArtifactRemover

    # 1. Load raw
    raw_df = pd.read_csv("data/synthetic_vitals.csv")

    # 2. Clean one patient
    p1_df = raw_df[raw_df["patient_id"] == "P001"]
    cleaner = ArtifactRemover()
    clean_df = cleaner.fit_transform(p1_df)

    # 3. Engineer features
    engineer = FeatureEngineer(window_size="20s")
    feat_df = engineer.fit_transform(clean_df)

    print("\nOriginal columns:", len(clean_df.columns))
    print("Engineered columns:", len(feat_df.columns))
    print("\nNew features sample:")
    for col in feat_df.columns:
        if col not in clean_df.columns:
            print(f"- {col}")

    print(
        "\nRow count reduction (Original -> Engineered):",
        len(clean_df),
        "->",
        len(feat_df),
    )
    print("Success: Feature engineering pipeline is functional.")
