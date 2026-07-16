"""iForest Hyperparameter Search Script"""

import argparse
import configparser
from itertools import product
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def standard_scaler(latent_arr):
    """Apply Standard Scaler to latent array."""

    scaler = StandardScaler()

    latent_scaled = scaler.fit_transform(latent_arr)

    return latent_scaled


def main():
    """Main function to perform iforest hyperparameter search."""
    # PArse arguments
    parser = argparse.ArgumentParser(description="iForest Hyperparameter Search")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the .ini config file"
    )
    args = parser.parse_args()

    # 2. Parse .ini with Interpolation for ${common:bin}
    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )

    config.read(args.config)

    # Extract paths and variables
    bin_id = config["common"]["bin"]
    latent_dir = config["directory"]["latent"]
    file_name = config["file"]["train"]

    input_path = os.path.join(latent_dir, file_name)

    iforest_dir = os.path.join(latent_dir, "iforest")
    os.makedirs(iforest_dir, exist_ok=True)

    output_path = os.path.join(iforest_dir, f"iforest_hypersearch_{bin_id}.csv")
    config_backup_path = os.path.join(
        iforest_dir, f"iforest_config_backup_{bin_id}.ini"
    )

    # 3. Load Data (.npy)
    print(f"--- Processing {bin_id} ---")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    latent = np.load(input_path)
    latent_scaled = standard_scaler(latent)

    # 4. Grid Search Setup

    samples_range = [64, 128, 256, 512]
    estimators_range = [100, 200, 300]
    features_range = [1.0, 0.75, 0.5]
    param_combinations = list(product(samples_range, estimators_range, features_range))

    results_df = pd.DataFrame(index=range(len(latent_scaled)))

    # 5. Execute Search
    print(f"Running {len(param_combinations)} models...")

    for model_number, (s, e, f) in enumerate(param_combinations):

        f_pct = int(f * 100)
        col_name = f"s{s}_e{e}_f{f_pct}"  # e.g., s256_e100_f100
        print(
            f"Fit model {model_number+1} of " f"{len(param_combinations)}: {col_name}"
        )

        iforest = IsolationForest(
            n_estimators=e,
            max_samples=s,
            max_features=f,
            contamination=0.01,
            n_jobs=-1,
            random_state=42,
        )

        # Fit and predict
        iforest.fit(latent_scaled)
        preds = iforest.predict(latent_scaled)

        # Store results (Anomaly = -1, Normal = 1)
        results_df[col_name] = (preds == -1).astype(int)

    # 6. Save Results and Backup Config
    results_df["consensus_score"] = results_df.sum(axis=1)
    results_df.to_csv(output_path, index=False)

    # Provenance: Copy the config file used to the results directory
    shutil.copy(args.config, config_backup_path)

    print(f"Success! Results and config backup saved in: {latent_dir}")


if __name__ == "__main__":
    main()
