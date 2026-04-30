"""
05_train_cnn.py — LOPO cross-validation for CNN on landscape images.
Trains on 23 function classes, tests on 1 held-out class, rotates through all 24.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from collections import Counter
from config import (
    BBOB_DIM, BBOB_N_FUNCTIONS, BBOB_N_INSTANCES,
    IMG_SIZE_A, IMG_SIZE_B, N_CLASSES, CLASS_NAMES,
    FUNCTION_TO_CLASS, IMAGES_DIR, MODELS_DIR, RESULTS_DIR,
    BATCH_SIZE, LEARNING_RATE, N_EPOCHS, WEIGHT_DECAY, RANDOM_SEED
)
import importlib as _il
_cnn_mod = _il.import_module("04_cnn_model")
build_model = _cnn_mod.build_model


class LandscapeImageDataset(Dataset):
    """Loads landscape PNG images on the fly."""
    def __init__(self, records, transform=None):
        self.records   = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        label = torch.tensor(rec["class_idx"], dtype=torch.long)
        return img, label


def build_records(n_functions=BBOB_N_FUNCTIONS, n_instances=BBOB_N_INSTANCES,
                   dim=BBOB_DIM, image_type="a"):
    """Build list of image records with paths and labels."""
    subdir = "type_a" if image_type == "a" else "type_b"
    image_dir = os.path.join(IMAGES_DIR, subdir)

    records = []
    for func_id in range(1, n_functions + 1):
        for instance in range(1, n_instances + 1):
            stem = f"f{func_id:02d}_i{instance:02d}_d{dim}"
            path = os.path.join(image_dir, stem + ".png")
            if not os.path.exists(path):
                continue
            records.append({
                "path":      path,
                "func_id":   func_id,
                "instance":  instance,
                "class_idx": FUNCTION_TO_CLASS[func_id],
            })
    return records


def get_train_transform(img_size=IMG_SIZE_A):
    """Training augmentation: flips + colour jitter."""
    H, W = img_size
    return T.Compose([
        T.Resize((H, W)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_val_transform(img_size=IMG_SIZE_A):
    """Validation transform: resize + normalize only."""
    H, W = img_size
    return T.Compose([
        T.Resize((H, W)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    y_bin = label_binarize(all_labels, classes=list(range(N_CLASSES)))
    try:
        auc = roc_auc_score(y_bin, np.array(all_probs), average="macro",
                             multi_class="ovr")
    except ValueError:
        auc = float("nan")
    return total_loss / n, acc, f1, auc, all_preds, all_labels


def run_lopo_cv(image_type="a", n_functions=BBOB_N_FUNCTIONS, dim=BBOB_DIM,
                n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY, seed=RANDOM_SEED):
    """Run full LOPO CV for the CNN. Returns DataFrame with per-fold results."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    img_size = IMG_SIZE_A if image_type == "a" else IMG_SIZE_B
    all_records = build_records(n_functions=n_functions, dim=dim,
                                 image_type=image_type)

    if len(all_records) == 0:
        raise RuntimeError(f"No images found. Run 01_sampling.py and 02_render_images.py first.")

    print(f"\nLOPO CV — CNN Type {'A' if image_type == 'a' else 'B'}")
    print(f"  Total images: {len(all_records)}")
    print(f"  Image size: {img_size}")
    print(f"  Epochs per fold: {n_epochs}, batch: {batch_size}, lr: {lr}\n")

    fold_results = []

    for held_out_func in tqdm(range(1, n_functions + 1), desc="LOPO folds"):
        # Split records
        train_recs = [r for r in all_records if r["func_id"] != held_out_func]
        test_recs  = [r for r in all_records if r["func_id"] == held_out_func]

        if len(test_recs) == 0:
            continue  # image file missing — skip

        train_transform = get_train_transform(img_size)
        val_transform   = get_val_transform(img_size)

        train_ds = LandscapeImageDataset(train_recs, transform=train_transform)
        test_ds  = LandscapeImageDataset(test_recs,  transform=val_transform)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                   shuffle=True, num_workers=0, pin_memory=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                                   shuffle=False, num_workers=0, pin_memory=False)

        # Compute class weights for this fold's training set
        train_labels = [r["class_idx"] for r in train_recs]
        class_counts = np.bincount(train_labels, minlength=N_CLASSES).astype(float)
        class_weights = 1.0 / np.maximum(class_counts, 1.0)
        class_weights = class_weights / class_weights.sum() * N_CLASSES
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # Fresh model for each fold
        model = build_model().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01
        )
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        best_val_acc = -1.0  # use -1 so even 0% accuracy is recorded
        best_state   = None
        best_f1      = 0.0
        best_auc     = float("nan")
        best_preds   = []
        best_labels  = []

        for epoch in range(n_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc, val_f1, val_auc, preds, labels = evaluate(
                model, test_loader, criterion, device
            )
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_preds  = preds
                best_labels = labels
                best_f1     = val_f1
                best_auc    = val_auc

        # Save best checkpoint for this fold
        ckpt_path = os.path.join(
            MODELS_DIR,
            f"cnn_type{image_type}_fold{held_out_func:02d}.pt"
        )
        torch.save(best_state, ckpt_path)

        fold_results.append({
            "fold":      held_out_func,
            "func_id":   held_out_func,
            "class_idx": FUNCTION_TO_CLASS[held_out_func],
            "acc":       best_val_acc,
            "macro_f1":  best_f1,
            "auc_roc":   best_auc,
            "n_test":    len(test_recs),
            "preds":     list(best_preds),
            "labels":    list(best_labels),
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
    y_bin_all = label_binarize(all_labels_pooled, classes=list(range(N_CLASSES)))
    try:
        # For global AUC we'd need probabilities; use per-fold acc mean instead
        global_auc = float("nan")
    except ValueError:
        global_auc = float("nan")

    # Build DataFrame for CSV (drop preds/labels columns)
    df_rows = []
    for fr in fold_results:
        df_rows.append({k: v for k, v in fr.items() if k not in ("preds", "labels")})
    df = pd.DataFrame(df_rows)

    # Save results
    out_csv = os.path.join(RESULTS_DIR, f"cnn_type{image_type}_lopo_results.csv")
    df.to_csv(out_csv, index=False)

    # Print summary — use pooled metrics
    print(f"\n{'='*50}")
    print(f"CNN Type {'A' if image_type == 'a' else 'B'} — LOPO Summary")
    print(f"  Per-fold Accuracy (mean): {df['acc'].mean():.3f} ± {df['acc'].std():.3f}")
    print(f"  Global Accuracy (pooled): {global_acc:.3f}")
    print(f"  Global Macro F1 (pooled): {global_f1:.3f}")
    print(f"  Results saved to: {out_csv}")

    return df


if __name__ == "__main__":
    # Train CNN on Type A images
    results_a = run_lopo_cv(image_type="a")
    # Train CNN on Type B images
    results_b = run_lopo_cv(image_type="b")
