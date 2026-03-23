import pandas as pd
import numpy as np
from src.preprocessing.artifact_detection import ArtifactRemover
from src.features.feature_engineering import FeatureEngineer


def verify_phase3():
    print("--- Detailed Phase 3 Verification ---")

    # 1. Load data
    df = pd.read_csv("data/synthetic_vitals.csv")

    # 2. Clean
    cleaner = ArtifactRemover()
    clean_df = cleaner.fit_transform(df)

    # 3. Engineer
    engineer = FeatureEngineer(window_size="20s")
    feat_df = engineer.fit_transform(clean_df)

    # Assertions and checks
    print("\n1. Schema Checks:")
    expected_cols = 31
    actual_cols = len(feat_df.columns)
    print(f"  Columns: {actual_cols} (Expected: {expected_cols})")
    assert actual_cols == expected_cols, "Column count mismatch!"

    print("\n2. Missing Value Checks:")
    null_counts = feat_df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"  Total NaNs across all engineered data: {total_nulls}")
    if total_nulls > 0:
        print("  WARNING: NaNs found in columns:")
        print(null_counts[null_counts > 0])

    print("\n3. Logic Checks (Ranges):")

    # Shock Index should be positive and usually between 0.5 - 1.5
    si_min, si_max = feat_df["shock_index"].min(), feat_df["shock_index"].max()
    print(f"  Shock Index range: {si_min:.2f} to {si_max:.2f}")
    assert si_min >= 0, "Shock index cannot be negative"

    # HR/SpO2 Correlation should be between -1 and 1
    corr_min, corr_max = feat_df["hr_spo2_corr"].min(), feat_df["hr_spo2_corr"].max()
    print(f"  HR-SpO2 Corr range: {corr_min:.2f} to {corr_max:.2f}")
    assert corr_min >= -1.01 and corr_max <= 1.01, "Correlation out of bounds"

    # Standard deviation should never be negative
    std_cols = [c for c in feat_df.columns if "_std" in c]
    for c in std_cols:
        min_std = feat_df[c].min()
        assert min_std >= 0, f"Negative Standard Deviation found in {c}!"
    print(f"  All {len(std_cols)} std dev columns are non-negative.")

    # Max >= Mean >= Min check
    print("\n4. Distribution Validity:")
    for base in ["heart_rate", "spo2", "bp_sys", "bp_dia"]:
        # Floating point math can have tiny errors, use numpy isclose or small buffer
        is_valid = (
            feat_df[f"{base}_max"] >= feat_df[f"{base}_mean"] - 1e-5
        ).all() and (feat_df[f"{base}_mean"] >= feat_df[f"{base}_min"] - 1e-5).all()
        print(f"  {base} Max >= Mean >= Min: {is_valid}")
        assert is_valid, f"Logic error in {base} stats"

    print("\n5. Patient count preservation:")
    patients_in = clean_df["patient_id"].nunique()
    patients_out = feat_df["patient_id"].nunique()
    print(f"  Patients processed: {patients_out} (Expected: {patients_in})")
    assert patients_in == patients_out, "Lost patients during engineering!"

    print(
        "\nVerification Passed: Phase 3 features are mathematically sound and robust."
    )


if __name__ == "__main__":
    verify_phase3()
