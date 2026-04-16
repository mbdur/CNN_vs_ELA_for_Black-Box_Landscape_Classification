"""
03_ela_features.py — ELA feature extraction via pflacco.
Uses the same N=500 sample points as image rendering.
Outputs data/ela_features/ela_all_d{dim}.csv.
"""

import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

# pflacco feature computation functions
from pflacco.classical_ela_features import (
    calculate_ela_distribution,
    calculate_ela_meta,
    calculate_nbc,
    calculate_information_content,
    calculate_dispersion,
)

from config import (
    BBOB_DIM, BBOB_N_FUNCTIONS, BBOB_N_INSTANCES,
    SAMPLES_DIR, ELA_DIR, FUNCTION_TO_CLASS, ELA_FEATURE_SETS
)


def compute_ela_features(X, y, feature_sets=None):
    """Compute ELA feature sets for one (X, y) sample."""
    if feature_sets is None:
        feature_sets = ELA_FEATURE_SETS

    all_features = {}
    dim = X.shape[1]

    X_df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(dim)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if "ela_distribution" in feature_sets:
            try:
                feats = calculate_ela_distribution(X_df, y)
                all_features.update(feats)
            except Exception as e:
                all_features["ela_distribution_error"] = str(e)

        if "ela_meta" in feature_sets:
            try:
                feats = calculate_ela_meta(X_df, y)
                all_features.update(feats)
            except Exception as e:
                all_features["ela_meta_error"] = str(e)

        if "nbc" in feature_sets:
            try:
                feats = calculate_nbc(X_df, y)
                all_features.update(feats)
            except Exception as e:
                all_features["nbc_error"] = str(e)

        if "information_content" in feature_sets:
            try:
                feats = calculate_information_content(X_df, y)
                all_features.update(feats)
            except Exception as e:
                all_features["information_content_error"] = str(e)

        if "dispersion" in feature_sets:
            try:
                feats = calculate_dispersion(X_df, y)
                all_features.update(feats)
            except Exception as e:
                all_features["dispersion_error"] = str(e)

    return all_features


def run_ela_extraction(n_functions=BBOB_N_FUNCTIONS, n_instances=BBOB_N_INSTANCES,
                        dim=BBOB_DIM, samples_dir=SAMPLES_DIR,
                        out_dir=ELA_DIR, overwrite=False):
    """Extract ELA features from all saved sample files, save as CSV."""
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"ela_all_d{dim}.csv")

    if os.path.exists(out_csv) and not overwrite:
        print(f"Loading existing ELA features from {out_csv}")
        return pd.read_csv(out_csv)

    rows = []
    total = n_functions * n_instances

    print(f"\nExtracting ELA features: {n_functions}×{n_instances} = {total} instances")
    print(f"Feature sets: {ELA_FEATURE_SETS}\n")

    with tqdm(total=total, desc="ELA features") as pbar:
        for func_id in range(1, n_functions + 1):
            for instance in range(1, n_instances + 1):
                stem = f"f{func_id:02d}_i{instance:02d}_d{dim}"
                npz_path = os.path.join(samples_dir, stem + ".npz")

                if not os.path.exists(npz_path):
                    print(f"  WARNING: missing sample: {npz_path}")
                    pbar.update(1)
                    continue

                data = np.load(npz_path)
                X, y = data["X"], data["y"]

                features = compute_ela_features(X, y)

                row = {
                    "func_id":   func_id,
                    "instance":  instance,
                    "class_idx": FUNCTION_TO_CLASS[func_id],
                    **features
                }
                rows.append(row)

                pbar.set_postfix({"last": f"f{func_id:02d}_i{instance:02d}"})
                pbar.update(1)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved ELA features: {df.shape} → {out_csv}")
    return df


def load_ela_features(dim=BBOB_DIM, out_dir=ELA_DIR):
    """Load ELA features CSV."""
    out_csv = os.path.join(out_dir, f"ela_all_d{dim}.csv")
    if not os.path.exists(out_csv):
        raise FileNotFoundError(f"No ELA features at {out_csv}. Run 03_ela_features.py first.")
    return pd.read_csv(out_csv)


def get_feature_matrix(df):
    """Split ELA DataFrame into (X, y, feature_names) for sklearn."""
    meta_cols = ["func_id", "instance", "class_idx"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    feature_df = df[feature_cols].copy()
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.dropna(axis=1, how="any")

    X = feature_df.values
    y = df["class_idx"].values
    return X, y, list(feature_df.columns)


if __name__ == "__main__":
    df = run_ela_extraction()

    # Preview
    print("\nFeature matrix preview:")
    print(f"  Shape: {df.shape}")
    print(f"  Classes: {df['class_idx'].value_counts().to_dict()}")
    print(f"  First 5 columns: {list(df.columns[:5])}")

    X, y, feat_names = get_feature_matrix(df)
    print(f"\nClean feature matrix: X={X.shape}, y={y.shape}")
    print(f"Number of usable features: {len(feat_names)}")
