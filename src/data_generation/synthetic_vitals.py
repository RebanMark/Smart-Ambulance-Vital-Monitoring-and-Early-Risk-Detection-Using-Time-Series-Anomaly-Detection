import pandas as pd
import numpy as np
import argparse
import os


def generate_patient_vitals(patient_id, scenario, duration_s=1800, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Base timestamp
    start_time = pd.Timestamp.now().round("s")
    timestamps = pd.date_range(start_time, periods=duration_s, freq="s")

    # Base lists
    hr = np.zeros(duration_s)
    spo2 = np.zeros(duration_s)
    bp_sys = np.zeros(duration_s)
    bp_dia = np.zeros(duration_s)
    motion = np.zeros(duration_s)
    labels = np.array([scenario] * duration_s, dtype=object)

    # Common baseline (Normal)
    hr[:] = np.random.normal(80, 5, duration_s)
    spo2[:] = np.random.normal(98, 1, duration_s)
    bp_sys[:] = np.random.normal(120, 5, duration_s)
    bp_dia[:] = np.random.normal(80, 5, duration_s)
    motion[:] = np.random.normal(0.5, 0.2, duration_s)

    if scenario == "distress":
        # Deterioration starts around the middle
        deterioration_start = np.random.randint(duration_s // 3, 2 * duration_s // 3)
        # Linear degradation
        ramp = np.linspace(0, 1, duration_s - deterioration_start)

        hr[deterioration_start:] += ramp * 80  # Spikes HR up to ~160
        spo2[deterioration_start:] -= ramp * 25  # Drops SpO2 to ~73
        bp_sys[deterioration_start:] -= ramp * 60  # Drops BP sys to ~60
        bp_dia[deterioration_start:] -= ramp * 50  # Drops BP dia to ~30

    elif scenario == "artifact":
        # Add random motion spikes simulating ambulance bumps
        num_spikes = np.random.randint(5, 15)
        for _ in range(num_spikes):
            spike_start = np.random.randint(0, duration_s - 20)
            spike_len = np.random.randint(5, 20)

            # High motion
            motion[spike_start : spike_start + spike_len] += np.random.uniform(
                3, 8, spike_len
            )

            # Consequent HR bumps and SpO2 drops
            hr[spike_start : spike_start + spike_len] += np.random.uniform(
                20, 40, spike_len
            )
            spo2[spike_start : spike_start + spike_len] -= np.random.uniform(
                5, 15, spike_len
            )

        # Add random NaN dropouts (sensor detachment during motion)
        num_dropouts = np.random.randint(2, 5)
        for _ in range(num_dropouts):
            drop_start = np.random.randint(0, duration_s - 10)
            drop_len = np.random.randint(2, 5)
            # Sensors read NaN
            hr[drop_start : drop_start + drop_len] = np.nan
            spo2[drop_start : drop_start + drop_len] = np.nan
            bp_sys[drop_start : drop_start + drop_len] = np.nan
            bp_dia[drop_start : drop_start + drop_len] = np.nan

    # Clip values to physiological bounds
    hr = np.clip(hr, 30, 220)
    spo2 = np.clip(spo2, 50, 100)
    bp_sys = np.clip(bp_sys, 50, 250)
    bp_dia = np.clip(bp_dia, 30, 150)
    motion = np.clip(motion, 0, 10)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "patient_id": f"P{patient_id:03d}",
            "heart_rate": hr,
            "spo2": spo2,
            "bp_sys": bp_sys,
            "bp_dia": bp_dia,
            "motion_signal": motion,
            "event_label": labels,
        }
    )

    return df


def generate_dataset(n_patients=15, output_path="data/synthetic_vitals.csv", seed=42):
    if seed is not None:
        np.random.seed(seed)

    scenarios = ["normal", "distress", "artifact"]
    # Ensure at least somewhat even distribution of scenarios
    patient_scenarios = [scenarios[i % 3] for i in range(n_patients)]
    np.random.shuffle(patient_scenarios)

    dfs = []
    for i, scenario in enumerate(patient_scenarios):
        # Generate with specific seed per patient for reproducibility
        df = generate_patient_vitals(i + 1, scenario, seed=seed + i if seed else None)
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    final_df.to_csv(output_path, index=False)
    print(f"Generated dataset for {n_patients} patients at '{output_path}'")
    print(f"Shape: {final_df.shape}")
    print("Scenario distribution:")
    print(
        final_df["event_label"].value_counts() / 1800
    )  # Count of patients per scenario


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic ambulance vitals dataset."
    )
    parser.add_argument(
        "--patients", type=int, default=15, help="Number of patients to generate"
    )
    parser.add_argument(
        "--output", type=str, default="data/synthetic_vitals.csv", help="Output path"
    )
    args = parser.parse_args()

    generate_dataset(n_patients=args.patients, output_path=args.output)
