"""
08_branch_b_budget.py — Sample budget crossover experiment.
Varies N in {50, 100, 200, 500, 1000} and compares CNN vs ELA+RF accuracy.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    BBOB_DIM, BBOB_N_FUNCTIONS, BBOB_N_INSTANCES,
    SAMPLE_BUDGETS_BRANCH_B, SAMPLES_DIR, IMAGES_DIR, ELA_DIR,
    RESULTS_DIR, RANDOM_SEED
)
import importlib as _il
_render_mod = _il.import_module("02_render_images")
render_type_a = _render_mod.render_type_a
save_image = _render_mod.save_image
_ela_mod = _il.import_module("03_ela_features")
compute_ela_features = _ela_mod.compute_ela_features
get_feature_matrix = _ela_mod.get_feature_matrix
from config import FUNCTION_TO_CLASS, N_CLASSES


def subsample(X, y, n, seed):
    """Draw n random rows from X, y without replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
    return X[idx], y[idx]


def run_budget_experiment(n_functions=BBOB_N_FUNCTIONS, n_instances=BBOB_N_INSTANCES,
                           dim=BBOB_DIM, budgets=SAMPLE_BUDGETS_BRANCH_B,
                           results_dir=RESULTS_DIR, seed=RANDOM_SEED):
    """Run ELA+RF at each budget N, render images for CNN runs."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import label_binarize

    all_rows = []

    for N in budgets:
        print(f"\n{'─'*50}")
        print(f"Budget N={N}")

        ela_rows = []

        for func_id in tqdm(range(1, n_functions + 1),
                            desc=f"ELA features N={N}"):
            for instance in range(1, n_instances + 1):
                stem = f"f{func_id:02d}_i{instance:02d}_d{dim}"
                npz_path = os.path.join(SAMPLES_DIR, stem + ".npz")
                if not os.path.exists(npz_path):
                    continue

                data = np.load(npz_path)
                X_full, y_full = data["X"], data["y"]

                sub_seed = seed + func_id * 1000 + instance * 10 + N
                X_sub, y_sub = subsample(X_full, y_full, N, seed=sub_seed)

                features = compute_ela_features(X_sub, y_sub)
                ela_rows.append({
                    "func_id":   func_id,
                    "instance":  instance,
                    "class_idx": FUNCTION_TO_CLASS[func_id],
                    "N":         N,
                    **features
                })

        df_ela = pd.DataFrame(ela_rows)

        meta_cols = ["func_id", "instance", "class_idx", "N"]
        feature_cols = [c for c in df_ela.columns if c not in meta_cols]
        feat_df = df_ela[feature_cols].apply(pd.to_numeric, errors="coerce")
        feat_df = feat_df.dropna(axis=1, how="any")
        X_all = feat_df.values
        y_all = df_ela["class_idx"].values
        func_ids_all = df_ela["func_id"].values

        fold_accs = []
        for held_out_func in range(1, n_functions + 1):
            train_mask = func_ids_all != held_out_func
            test_mask  = func_ids_all == held_out_func

            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

            if X_test.shape[0] == 0:
                continue

            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X_train)
            X_test  = imputer.transform(X_test)

            rf = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                         random_state=seed, n_jobs=-1)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            fold_accs.append(accuracy_score(y_test, preds))

        mean_acc = np.mean(fold_accs) if fold_accs else float("nan")
        all_rows.append({
            "N":       N,
            "method":  "ELA+RF",
            "acc":     mean_acc,
        })
        print(f"  ELA+RF  acc={mean_acc:.3f}")

        budget_img_dir = os.path.join(IMAGES_DIR, f"type_a_N{N}")
        os.makedirs(budget_img_dir, exist_ok=True)

        for func_id in tqdm(range(1, n_functions + 1),
                            desc=f"Render N={N}"):
            for instance in range(1, n_instances + 1):
                stem = f"f{func_id:02d}_i{instance:02d}_d{dim}"
                npz_path = os.path.join(SAMPLES_DIR, stem + ".npz")
                out_path = os.path.join(budget_img_dir, stem + ".png")
                if os.path.exists(out_path):
                    continue
                if not os.path.exists(npz_path):
                    continue

                data = np.load(npz_path)
                X_full, y_full = data["X"], data["y"]
                sub_seed = seed + func_id * 1000 + instance * 10 + N
                X_sub, y_sub = subsample(X_full, y_full, N, seed=sub_seed)
                img = render_type_a(X_sub, y_sub)
                save_image(img, out_path)

        print(f"  Images at N={N} saved to {budget_img_dir}")

    df_results = pd.DataFrame(all_rows)
    out_csv = os.path.join(results_dir, "branch_b_budget_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nBranch B results saved to {out_csv}")
    return df_results


def plot_budget_crossover(results_dir=RESULTS_DIR):
    """Plot accuracy vs sample budget N for all methods."""
    csv_path = os.path.join(results_dir, "branch_b_budget_results.csv")
    if not os.path.exists(csv_path):
        print("Run run_budget_experiment() first.")
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"ELA+RF": "#FF9800", "CNN-A": "#2196F3", "CNN-B": "#4CAF50"}

    for method, group in df.groupby("method"):
        group = group.sort_values("N")
        ax.plot(group["N"], group["acc"], "o-",
                color=colors.get(method, "grey"),
                label=method, linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_xlabel("Sample Budget N (log scale)")
    ax.set_ylabel("Mean LOPO Accuracy")
    ax.set_title("Accuracy vs Sample Budget: CNN vs ELA+RF")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(results_dir, "branch_b_budget_crossover.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Budget crossover plot saved to: {out_path}")


if __name__ == "__main__":
    df = run_budget_experiment()
    plot_budget_crossover()
