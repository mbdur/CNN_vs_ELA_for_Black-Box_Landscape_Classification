"""
06_ela_rf_baseline.py — ELA + Random Forest LOPO baseline.
Same splits as CNN for fair comparison.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    BBOB_N_FUNCTIONS, BBOB_DIM, FUNCTION_TO_CLASS, N_CLASSES,
    CLASS_NAMES, RESULTS_DIR, RANDOM_SEED
)
import importlib.util, sys
_spec = importlib.util.spec_from_file_location("ela_features", "03_ela_features.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
load_ela_features = _m.load_ela_features
get_feature_matrix = _m.get_feature_matrix


def run_ela_rf_lopo(dim=BBOB_DIM, n_functions=BBOB_N_FUNCTIONS, seed=RANDOM_SEED):
    """LOPO CV with ELA+RF. Returns DataFrame with per-fold results."""
    print("\nLoading ELA features...")
    df = load_ela_features(dim=dim)
    X_all, y_all, feature_names = get_feature_matrix(df)
    func_ids = df["func_id"].values

    print(f"Feature matrix: X={X_all.shape}, y={y_all.shape}")
    print(f"Number of features: {len(feature_names)}")

    fold_results = []

    print(f"\nLOPO CV — ELA + Random Forest")

    for held_out_func in tqdm(range(1, n_functions + 1), desc="ELA+RF folds"):
        train_mask = func_ids != held_out_func
        test_mask  = func_ids == held_out_func

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

        if X_test.shape[0] == 0:
            continue

        X_train = np.where(np.isfinite(X_train), X_train, np.nan)
        X_test  = np.where(np.isfinite(X_test),  X_test,  np.nan)
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test  = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        rf = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)
        probs = rf.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average="macro", zero_division=0)

        y_bin = label_binarize(y_test, classes=list(range(N_CLASSES)))
        try:
            auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
        except ValueError:
            auc = float("nan")

        fold_results.append({
            "fold":     held_out_func,
            "func_id":  held_out_func,
            "class_idx": FUNCTION_TO_CLASS[held_out_func],
            "acc":      acc,
            "macro_f1": f1,
            "auc_roc":  auc,
            "n_test":   X_test.shape[0],
            "preds":    list(preds),
            "labels":   list(y_test),
        })

    # --- Pooled global metrics (all folds combined) ---
    all_preds_pooled  = []
    all_labels_pooled = []
    for fr in fold_results:
        all_preds_pooled.extend(fr["preds"])
        all_labels_pooled.extend(fr["labels"])

    global_acc = accuracy_score(all_labels_pooled, all_preds_pooled)
    global_f1  = f1_score(all_labels_pooled, all_preds_pooled,
                          average="macro", zero_division=0)

    # Build DataFrame for CSV (drop preds/labels columns)
    df_rows = []
    for fr in fold_results:
        df_rows.append({k: v for k, v in fr.items() if k not in ("preds", "labels")})
    df_results = pd.DataFrame(df_rows)

    out_csv = os.path.join(RESULTS_DIR, "ela_rf_lopo_results.csv")
    df_results.to_csv(out_csv, index=False)

    print(f"\n{'='*50}")
    print("ELA + RF — LOPO Summary")
    print(f"  Per-fold Accuracy (mean): {df_results['acc'].mean():.3f} ± {df_results['acc'].std():.3f}")
    print(f"  Global Accuracy (pooled): {global_acc:.3f}")
    print(f"  Global Macro F1 (pooled): {global_f1:.3f}")
    print(f"  Results saved to: {out_csv}")

    return df_results


def build_comparison_table(results_dir=RESULTS_DIR):
    """3-way comparison table: CNN-A vs CNN-B vs ELA+RF."""
    methods = {
        "CNN (Type A — PCA)":      "cnn_typea_lopo_results.csv",
        "CNN (Type B — Pairwise)": "cnn_typeb_lopo_results.csv",
        "ELA + RF":                "ela_rf_lopo_results.csv",
    }

    rows = []
    for method_name, fname in methods.items():
        fpath = os.path.join(results_dir, fname)
        if not os.path.exists(fpath):
            print(f"  Missing: {fpath}")
            continue
        df = pd.read_csv(fpath)
        rows.append({
            "Method":    method_name,
            "Accuracy":  f"{df['acc'].mean():.3f} ± {df['acc'].std():.3f}",
            "Macro F1":  f"{df['macro_f1'].mean():.3f} ± {df['macro_f1'].std():.3f}",
            "AUC-ROC":   f"{df['auc_roc'].mean():.3f} ± {df['auc_roc'].std():.3f}",
        })

    summary = pd.DataFrame(rows)
    print("\n=== COMPARISON TABLE ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(results_dir, "comparison_table.csv"), index=False)
    return summary


def plot_comparison_bar(results_dir=RESULTS_DIR):
    """Per-fold accuracy bar chart for all three methods."""
    methods = {
        "CNN-A": "cnn_typea_lopo_results.csv",
        "CNN-B": "cnn_typeb_lopo_results.csv",
        "ELA+RF": "ela_rf_lopo_results.csv",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    x = np.arange(1, 25)
    width = 0.25

    for i, (label, fname) in enumerate(methods.items()):
        fpath = os.path.join(results_dir, fname)
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath).sort_values("func_id")
        ax.bar(x + (i - 1) * width, df["acc"], width, label=label, color=colors[i], alpha=0.8)

    ax.set_xlabel("BBOB Function ID (held out)")
    ax.set_ylabel("Accuracy")
    ax.set_title("LOPO Accuracy per Function — CNN-A vs CNN-B vs ELA+RF")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)

    out_path = os.path.join(results_dir, "comparison_bar.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison bar chart saved to: {out_path}")


if __name__ == "__main__":
    run_ela_rf_lopo()
    build_comparison_table()
    plot_comparison_bar()
