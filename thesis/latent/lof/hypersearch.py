"""LOF Hyperparameter Search Script"""

import argparse
import configparser
from itertools import product
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def standard_scaler(latent_arr):
    """Apply Standard Scaler to latent array."""

    scaler = StandardScaler()

    latent_scaled = scaler.fit_transform(latent_arr)

    return latent_scaled


def main():
    """Main function to perform LOF hyperparameter search."""
    # PArse arguments
    parser = argparse.ArgumentParser(description="LOF Hyperparameter Search")
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
    output_path = os.path.join(latent_dir, f"lof_hypersearch_{bin_id}.csv")
    config_backup_path = os.path.join(latent_dir, f"config_backup_{bin_id}.ini")

    # 3. Load Data (.npy)
    print(f"--- Processing {bin_id} ---")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    latent = np.load(input_path)
    latent_scaled = standard_scaler(latent)

    # 4. Grid Search Setup
    neighbors_range = [20, 40, 60, 80, 100]
    metrics = ["euclidean", "manhattan", "cosine"]
    param_combinations = list(product(neighbors_range, metrics))
    results_df = pd.DataFrame(index=range(len(latent_scaled)))

    # 5. Execute Search
    print(f"Running {len(param_combinations)} models...")
    for n, m in param_combinations:
        col_name = f"n{n}_{m}"
        print(f" -> {col_name}")
        lof = LocalOutlierFactor(n_neighbors=n, metric=m, contamination=0.01, n_jobs=-1)
        results_df[col_name] = (lof.fit_predict(latent_scaled) == -1).astype(int)

    # 6. Save Results and Backup Config
    results_df["consensus_score"] = results_df.sum(axis=1)
    results_df.to_csv(output_path, index=False)

    # Provenance: Copy the config file used to the results directory
    shutil.copy(args.config, config_backup_path)

    print(f"Success! Results and config backup saved in: {latent_dir}")


if __name__ == "__main__":
    main()
