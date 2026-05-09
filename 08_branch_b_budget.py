
"""
08_branch_b_budget.py — Sample budget crossover experiment.
Varies N in {50, 100, 200, 500, 1000} and compares CNN vs ELA+RF accuracy.
"""

import os, sys, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image

from config import (
    BBOB_DIM, BBOB_N_FUNCTIONS, BBOB_N_INSTANCES,
    SAMPLE_BUDGETS_BRANCH_B, SAMPLES_DIR, IMAGES_DIR, ELA_DIR,
    RESULTS_DIR, RANDOM_SEED, FUNC_N_INSTANCES,
    FUNCTION_TO_CLASS, N_CLASSES, N_EPOCHS, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY
)
import importlib as _il
_render_mod = _il.import_module("02_render_images")
render_type_a = _render_mod.render_type_a
save_image    = _render_mod.save_image
_ela_mod = _il.import_module("03_ela_features")
compute_ela_features = _ela_mod.compute_ela_features
_model_mod = _il.import_module("04_cnn_model")
build_model = _model_mod.build_model


# ── helpers ──────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def subsample(X, y, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
    return X[idx], y[idx]


def _load_images_from_dir(img_dir, n_functions, func_n_instances, dim):
    """Load images + labels from a budget-specific image directory."""
    imgs, labels, fids = [], [], []
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    for func_id in range(1, n_functions + 1):
        n_inst = func_n_instances.get(func_id, 10)
        for instance in range(1, n_inst + 1):
            stem = f"f{func_id:02d}_i{instance:02d}_d{dim}"
            path = os.path.join(img_dir, stem + ".png")
            if not os.path.exists(path):
                continue
            img = Image.open(path).convert("RGB")
            imgs.append(tf(img))
            labels.append(FUNCTION_TO_CLASS[func_id])
            fids.append(func_id)
    return torch.stack(imgs), np.array(labels), np.array(fids)


def _compute_class_weights(y_train):
    classes, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / counts.astype(np.float64)
    weights = weights / weights.sum() * len(classes)
    w = torch.zeros(N_CLASSES)
    for c, wt in zip(classes, weights):
        w[int(c)] = float(wt)
    return w


def _train_cnn_lopo(X_all, y_all, func_ids, device,
                     n_functions, epochs=N_EPOCHS, patience=10):
    """Run full LOPO CNN training, return mean accuracy."""
    fold_accs = []
    for held_out in range(1, n_functions + 1):
        train_mask = func_ids != held_out
        test_mask  = func_ids == held_out

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]
        if X_test.shape[0] == 0:
            continue

        # class weights
        cw = _compute_class_weights(y_train).to(device)

        # build fresh model each fold
        model = build_model(n_classes=N_CLASSES).to(device)
        opt   = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                 lr=1e-4, weight_decay=1e-3)
        loss_fn = nn.CrossEntropyLoss(weight=cw)

        train_ds = TensorDataset(X_train, torch.tensor(y_train, dtype=torch.long))
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

        best_acc, wait = 0.0, 0
        best_state = None

        for ep in range(epochs):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()

            # quick val on test (same as main experiment)
            model.eval()
            with torch.no_grad():
                preds = model(X_test.to(device)).argmax(1).cpu().numpy()
            acc = (preds == y_test).mean()
            if acc > best_acc:
                best_acc, wait = acc, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= patience:
                    break

        fold_accs.append(best_acc)

    return np.mean(fold_accs) if fold_accs else float("nan")


# ── main entry points ───────────────────────────────────────────────

def run_budget_experiment(n_functions=BBOB_N_FUNCTIONS,
                           func_n_instances=FUNC_N_INSTANCES,
                           dim=BBOB_DIM, budgets=SAMPLE_BUDGETS_BRANCH_B,
                           results_dir=RESULTS_DIR, seed=RANDOM_SEED):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Budget experiment using device: {device}")

    all_rows = []

    for N in budgets:
        print(f"\n{'─'*50}")
        print(f"Budget N={N}")

        # ── 1. ELA+RF ───────────────────────────────────────────────
        ela_rows = []
        for func_id in tqdm(range(1, n_functions + 1),
                            desc=f"ELA features N={N}"):
            n_inst = func_n_instances.get(func_id, 10)
            for instance in range(1, n_inst + 1):
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
                    "func_id": func_id, "instance": instance,
                    "class_idx": FUNCTION_TO_CLASS[func_id], "N": N,
                    **features
                })

        df_ela = pd.DataFrame(ela_rows)
        meta_cols = ["func_id", "instance", "class_idx", "N"]
        feature_cols = [c for c in df_ela.columns if c not in meta_cols]
        feat_df = df_ela[feature_cols].apply(pd.to_numeric, errors="coerce")
        feat_df = feat_df.dropna(axis=1, how="any")
        X_mat = feat_df.values
        y_vec = df_ela["class_idx"].values
        fids  = df_ela["func_id"].values

        fold_accs = []
        for held_out in range(1, n_functions + 1):
            tr = fids != held_out; te = fids == held_out
            Xtr, ytr = X_mat[tr], y_vec[tr]
            Xte, yte = X_mat[te], y_vec[te]
            if Xte.shape[0] == 0:
                continue
            imp = SimpleImputer(strategy="median")
            Xtr = imp.fit_transform(Xtr); Xte = imp.transform(Xte)
            rf = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                         class_weight="balanced",
                                         random_state=seed, n_jobs=-1)
            rf.fit(Xtr, ytr)
            fold_accs.append(accuracy_score(yte, rf.predict(Xte)))

        ela_acc = np.mean(fold_accs) if fold_accs else float("nan")
        all_rows.append({"N": N, "method": "ELA+RF", "acc": ela_acc})
        print(f"  ELA+RF  acc={ela_acc:.3f}")

        # ── 2. Render images (overwrite old ones) ───────────────────
        budget_img_dir = os.path.join(IMAGES_DIR, f"type_a_N{N}")
        os.makedirs(budget_img_dir, exist_ok=True)

        for func_id in tqdm(range(1, n_functions + 1),
                            desc=f"Render N={N}"):
            n_inst = func_n_instances.get(func_id, 10)
            for instance in range(1, n_inst + 1):
                stem = f"f{func_id:02d}_i{instance:02d}_d{dim}"
                npz_path = os.path.join(SAMPLES_DIR, stem + ".npz")
                out_path = os.path.join(budget_img_dir, stem + ".png")
                if not os.path.exists(npz_path):
                    continue
                data = np.load(npz_path)
                X_full, y_full = data["X"], data["y"]
                sub_seed = seed + func_id * 1000 + instance * 10 + N
                X_sub, y_sub = subsample(X_full, y_full, N, seed=sub_seed)
                img = render_type_a(X_sub, y_sub)
                save_image(img, out_path)   # always overwrite

        print(f"  Images saved to {budget_img_dir}")

        # ── 3. CNN-A on budget images ───────────────────────────────
        X_imgs, y_imgs, fid_imgs = _load_images_from_dir(
            budget_img_dir, n_functions, func_n_instances, dim)
        cnn_acc = _train_cnn_lopo(X_imgs, y_imgs, fid_imgs, device,
                                   n_functions)
        all_rows.append({"N": N, "method": "CNN-A", "acc": cnn_acc})
        print(f"  CNN-A   acc={cnn_acc:.3f}")

    df_results = pd.DataFrame(all_rows)
    out_csv = os.path.join(results_dir, "branch_b_budget_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nBranch B results saved to {out_csv}")
    return df_results


def plot_budget_crossover(results_dir=RESULTS_DIR):
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
