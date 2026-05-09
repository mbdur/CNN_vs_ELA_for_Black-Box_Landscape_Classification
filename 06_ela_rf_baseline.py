
"""
06_ela_rf_baseline.py -- ELA + Random Forest LOPO baseline.
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
import importlib as _il
_ela_mod = _il.import_module("03_ela_features")
load_ela_features = _ela_mod.load_ela_features
get_feature_matrix = _ela_mod.get_feature_matrix


def run_ela_rf_lopo(dim=BBOB_DIM, n_functions=BBOB_N_FUNCTIONS, seed=RANDOM_SEED):
    """LOPO CV with ELA+RF. Returns DataFrame with per-fold results."""
    print("\nLoading ELA features...")
    df = load_ela_features(dim=dim)
    X_all, y_all, feature_names = get_feature_matrix(df)
    func_ids = df["func_id"].values

    print(f"Feature matrix: X={X_all.shape}, y={y_all.shape}")
    print(f"Number of features: {len(feature_names)}")

    fold_results = []
    all_predictions = []

    print(f"\nLOPO CV -- ELA + Random Forest")

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
        })

        # Store per-instance predictions for confusion matrix
        instances = df.loc[test_mask, "instance"].values if "instance" in df.columns else range(1, len(y_test) + 1)
        for inst, pred_val, true_val in zip(instances, preds, y_test):
            all_predictions.append({
                "func_id":  held_out_func,
                "instance": int(inst),
                "y_true":   int(true_val),
                "y_pred":   int(pred_val),
            })

    df_results = pd.DataFrame(fold_results)

    out_csv = os.path.join(RESULTS_DIR, "ela_rf_lopo_results.csv")
    df_results.to_csv(out_csv, index=False)

    # Save per-instance predictions
    pred_csv = os.path.join(RESULTS_DIR, "ela_rf_lopo_predictions.csv")
    pd.DataFrame(all_predictions).to_csv(pred_csv, index=False)
    print(f"  Per-instance predictions saved to: {pred_csv}")

    print(f"\n{'='*50}")
    print("ELA + RF -- LOPO Summary")
    print(f"  Accuracy:  {df_results['acc'].mean():.3f} +/- {df_results['acc'].std():.3f}")
    print(f"  Macro F1:  {df_results['macro_f1'].mean():.3f} +/- {df_results['macro_f1'].std():.3f}")
    print(f"  AUC-ROC:   {df_results['auc_roc'].mean():.3f} +/- {df_results['auc_roc'].std():.3f}")
    print(f"  Results saved to: {out_csv}")

    return df_results


def build_comparison_table(results_dir=RESULTS_DIR):
    """3-way comparison table: CNN-A vs CNN-B vs ELA+RF."""
    methods = {
        "CNN (Type A -- PCA)":      "cnn_typea_lopo_results.csv",
        "CNN (Type B -- Pairwise)": "cnn_typeb_lopo_results.csv",
        "ELA + RF":                 "ela_rf_lopo_results.csv",
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
            "Accuracy":  f"{df['acc'].mean():.3f} +/- {df['acc'].std():.3f}",
            "Macro F1":  f"{df['macro_f1'].mean():.3f} +/- {df['macro_f1'].std():.3f}",
            "AUC-ROC":   f"{df['auc_roc'].mean():.3f} +/- {df['auc_roc'].std():.3f}",
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
    ax.set_title("LOPO Accuracy per Function -- CNN-A vs CNN-B vs ELA+RF")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)

    out_path = os.path.join(results_dir, "comparison_bar.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison bar chart saved to: {out_path}")


def plot_confusion_matrices(results_dir=RESULTS_DIR):
    """Plot side-by-side confusion matrices for all three methods (from LOPO predictions)."""
    methods = {
        "CNN-A":  "cnn_typea_lopo_predictions.csv",
        "CNN-B":  "cnn_typeb_lopo_predictions.csv",
        "ELA+RF": "ela_rf_lopo_predictions.csv",
    }

    available = {}
    for label, fname in methods.items():
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            available[label] = pd.read_csv(fpath)
        else:
            print(f"  Skipping {label}: {fname} not found")

    if not available:
        print("No prediction files found -- run training first.")
        return

    n_methods = len(available)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, (label, df) in zip(axes, available.items()):
        cm = confusion_matrix(df["y_true"], df["y_pred"], labels=list(range(N_CLASSES)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, rotation=0, fontsize=8)

    fig.suptitle("LOPO Confusion Matrices -- All Methods", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(results_dir, "confusion_matrices.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrices saved to: {out_path}")


if __name__ == "__main__":
    run_ela_rf_lopo()
    build_comparison_table()
    plot_comparison_bar()
    plot_confusion_matrices()
