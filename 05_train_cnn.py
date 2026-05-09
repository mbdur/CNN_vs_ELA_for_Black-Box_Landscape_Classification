
"""
05_train_cnn.py — LOPO cross-validation for CNN on landscape images.

Key improvements over original:
- Class-weighted CrossEntropyLoss computed per fold from training labels
- Heavier data augmentation for small dataset regime
- Filters frozen params so optimizer only updates trainable weights
- Early stopping patience to prevent overfitting
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from config import (
    BBOB_DIM, BBOB_N_FUNCTIONS,
    IMG_SIZE_A, IMG_SIZE_B, N_CLASSES, CLASS_NAMES,
    FUNCTION_TO_CLASS, IMAGES_DIR, MODELS_DIR, RESULTS_DIR,
    BATCH_SIZE, LEARNING_RATE, N_EPOCHS, WEIGHT_DECAY, RANDOM_SEED,
    FUNC_N_INSTANCES
)

import importlib as _il
_cnn_mod = _il.import_module("04_cnn_model")
build_model = _cnn_mod.build_model


_IMAGE_CACHE = {}


def _load_rgb_cached(path):
    """Load an image once, then reuse it from memory."""
    img = _IMAGE_CACHE.get(path)

    if img is None:
        with Image.open(path) as im:
            img = im.convert("RGB").copy()
        _IMAGE_CACHE[path] = img

    return img.copy()


class LandscapeImageDataset(Dataset):
    """Loads landscape PNG images using an in-memory cache."""

    def __init__(self, records, transform=None):
        self.records = list(records)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = _load_rgb_cached(rec["path"])

        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        label = torch.tensor(rec["class_idx"], dtype=torch.long)
        return img, label


def build_records(
    n_functions=BBOB_N_FUNCTIONS,
    func_n_instances=FUNC_N_INSTANCES,
    dim=BBOB_DIM,
    image_type="a"
):
    """Build list of image records with paths and labels."""
    subdir = "type_a" if image_type == "a" else "type_b"
    image_dir = os.path.join(IMAGES_DIR, subdir)

    records = []

    for func_id in range(1, n_functions + 1):
        n_inst = func_n_instances.get(func_id, 10)

        for instance in range(1, n_inst + 1):
            stem = f"f{func_id:02d}_i{instance:02d}_d{dim}"
            path = os.path.join(image_dir, stem + ".png")

            if not os.path.exists(path):
                continue

            records.append({
                "path": path,
                "func_id": func_id,
                "instance": instance,
                "class_idx": FUNCTION_TO_CLASS[func_id],
            })

    return records


def _detect_resnet():
    """Check if the model is ResNet-based (uses ImageNet normalization)."""
    try:
        import torchvision.models as _m
        return True
    except ImportError:
        return False


_USE_IMAGENET_NORM = _detect_resnet()

# ImageNet stats for pretrained ResNet
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Generic stats for custom CNN
_GENERIC_MEAN = [0.5, 0.5, 0.5]
_GENERIC_STD  = [0.5, 0.5, 0.5]


def get_train_transform(img_size=IMG_SIZE_A):
    H, W = img_size
    mean = _IMAGENET_MEAN if _USE_IMAGENET_NORM else _GENERIC_MEAN
    std  = _IMAGENET_STD  if _USE_IMAGENET_NORM else _GENERIC_STD

    return T.Compose([
        T.Resize((H, W)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=30),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_val_transform(img_size=IMG_SIZE_A):
    H, W = img_size
    mean = _IMAGENET_MEAN if _USE_IMAGENET_NORM else _GENERIC_MEAN
    std  = _IMAGENET_STD  if _USE_IMAGENET_NORM else _GENERIC_STD

    return T.Compose([
        T.Resize((H, W)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def compute_class_weights(records):
    """Compute inverse-frequency class weights from a list of records."""
    counts = Counter(r["class_idx"] for r in records)
    total = sum(counts.values())
    weights = torch.zeros(N_CLASSES)

    for c in range(N_CLASSES):
        if counts[c] > 0:
            weights[c] = total / (N_CLASSES * counts[c])
        else:
            weights[c] = 1.0

    return weights


def stratified_train_val_split(records, val_fraction=0.2, seed=RANDOM_SEED):
    """Split outer training records into train/validation while preserving classes."""
    rng = np.random.default_rng(seed)

    by_class = {}
    for rec in records:
        by_class.setdefault(rec["class_idx"], []).append(rec)

    train_recs = []
    val_recs = []

    for class_idx, recs in by_class.items():
        recs = list(recs)
        rng.shuffle(recs)

        if len(recs) <= 1:
            train_recs.extend(recs)
            continue

        n_val = max(1, int(round(len(recs) * val_fraction)))
        n_val = min(n_val, len(recs) - 1)

        val_recs.extend(recs[:n_val])
        train_recs.extend(recs[n_val:])

    if len(val_recs) == 0 and len(train_recs) > 1:
        rng.shuffle(train_recs)
        val_recs.append(train_recs.pop())

    return train_recs, val_recs


def safe_multiclass_auc(y_true, probs):
    """Return NaN when ROC-AUC is undefined instead of producing warnings."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    if len(y_true) == 0:
        return float("nan")

    if len(np.unique(y_true)) < 2:
        return float("nan")

    try:
        y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return roc_auc_score(
                y_bin,
                probs,
                average="macro",
                multi_class="ovr"
            )

    except Exception:
        return float("nan")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, compute_auc=True):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)

        total_loss += loss.item() * images.size(0)

        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    n = len(all_labels)

    if n == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            [],
            [],
            np.empty((0, N_CLASSES)),
        )

    all_probs = np.asarray(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    if compute_auc:
        auc = safe_multiclass_auc(all_labels, all_probs)
    else:
        auc = float("nan")

    return total_loss / n, acc, f1, auc, all_preds, all_labels, all_probs


def _fmt_mean_std(series):
    vals = pd.to_numeric(series, errors="coerce").dropna()

    if len(vals) == 0:
        return "N/A"

    if len(vals) == 1:
        return f"{vals.mean():.3f}"

    return f"{vals.mean():.3f} +/- {vals.std():.3f}"


def _make_loader(dataset, batch_size, shuffle, device, num_workers):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True

    return DataLoader(dataset, **kwargs)


def run_lopo_cv(
    image_type="a",
    n_functions=BBOB_N_FUNCTIONS,
    dim=BBOB_DIM,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    seed=RANDOM_SEED,
    val_fraction=0.20,
    num_workers=0,
    limit_folds=None,
    patience=10,
):
    """
    Run LOPO CV for the CNN.

    Use limit_folds=2 for a quick smoke test.
    Use limit_folds=None for the full experiment.
    patience: early stopping patience (epochs without val improvement).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"\nDevice: {device}")

    img_size = IMG_SIZE_A if image_type == "a" else IMG_SIZE_B

    all_records = build_records(
        n_functions=n_functions,
        dim=dim,
        image_type=image_type
    )

    if len(all_records) == 0:
        raise RuntimeError("No images found. Run sampling and rendering first.")

    print(f"\nLOPO CV — CNN Type {'A' if image_type == 'a' else 'B'}")
    print(f"  Total images: {len(all_records)}")
    print(f"  Image size: {img_size}")
    print(f"  Epochs per fold: {n_epochs}, batch: {batch_size}, lr: {lr}")
    print(f"  Early stopping patience: {patience}")
    print("  Class-weighted CrossEntropyLoss: YES")
    print("  Test-fold AUC-ROC: skipped because each LOPO test fold has one true class.\n")

    print("Preloading/caching images in memory...")
    for rec in tqdm(all_records, desc="Image cache", leave=False):
        _load_rgb_cached(rec["path"])

    fold_results = []
    all_predictions = []

    fold_ids = list(range(1, n_functions + 1))

    if limit_folds is not None:
        fold_ids = fold_ids[:int(limit_folds)]

    for held_out_func in tqdm(fold_ids, desc="LOPO folds"):
        outer_train_recs = [
            r for r in all_records
            if r["func_id"] != held_out_func
        ]

        test_recs = [
            r for r in all_records
            if r["func_id"] == held_out_func
        ]

        if len(test_recs) == 0:
            print(f"Skipping f{held_out_func:02d}: no test images found.")
            continue

        train_recs, val_recs = stratified_train_val_split(
            outer_train_recs,
            val_fraction=val_fraction,
            seed=seed + held_out_func,
        )

        # Compute class weights from this fold's training data
        class_weights = compute_class_weights(train_recs).to(device)

        train_transform = get_train_transform(img_size)
        val_transform = get_val_transform(img_size)

        train_ds = LandscapeImageDataset(train_recs, transform=train_transform)
        val_ds = LandscapeImageDataset(val_recs, transform=val_transform)
        test_ds = LandscapeImageDataset(test_recs, transform=val_transform)

        train_loader = _make_loader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            device=device,
            num_workers=num_workers,
        )

        val_loader = _make_loader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            device=device,
            num_workers=num_workers,
        )

        test_loader = _make_loader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            device=device,
            num_workers=num_workers,
        )

        model = build_model().to(device)

        # Only optimize parameters that require gradients (skips frozen backbone)
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(n_epochs), 1),
            eta_min=lr * 0.01,
        )

        # Class-weighted loss
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_acc = -1.0
        best_state = None
        best_epoch = -1
        epochs_no_improve = 0

        for epoch in range(int(n_epochs)):
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
            )

            val_loss, val_acc, val_f1, val_auc, _, _, _ = evaluate(
                model,
                val_loader,
                criterion,
                device,
                compute_auc=True,
            )

            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        if best_state is None:
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        model.load_state_dict({
            k: v.to(device)
            for k, v in best_state.items()
        })

        # Use unweighted loss for test evaluation (fair accuracy measure)
        test_criterion = nn.CrossEntropyLoss()

        test_loss, test_acc, test_f1, test_auc, preds, labels, probs = evaluate(
            model,
            test_loader,
            test_criterion,
            device,
            compute_auc=False,
        )

        ckpt_path = os.path.join(
            MODELS_DIR,
            f"cnn_type{image_type}_fold{held_out_func:02d}.pt"
        )

        torch.save(best_state, ckpt_path)

        fold_results.append({
            "fold": held_out_func,
            "func_id": held_out_func,
            "class_idx": FUNCTION_TO_CLASS[held_out_func],
            "best_epoch": best_epoch,
            "val_acc": best_val_acc,
            "acc": test_acc,
            "macro_f1": test_f1,
            "auc_roc": test_auc,
            "n_train": len(train_recs),
            "n_val": len(val_recs),
            "n_test": len(test_recs),
        })

        for rec, pred, true, prob_vec in zip(test_recs, preds, labels, probs):
            row = {
                "func_id": rec["func_id"],
                "instance": rec["instance"],
                "y_true": int(true),
                "y_pred": int(pred),
            }

            for c in range(N_CLASSES):
                row[f"prob_class_{c}"] = float(prob_vec[c])

            all_predictions.append(row)

    df = pd.DataFrame(fold_results)

    out_csv = os.path.join(
        RESULTS_DIR,
        f"cnn_type{image_type}_lopo_results.csv"
    )

    pred_csv = os.path.join(
        RESULTS_DIR,
        f"cnn_type{image_type}_lopo_predictions.csv"
    )

    df.to_csv(out_csv, index=False)
    pd.DataFrame(all_predictions).to_csv(pred_csv, index=False)

    print(f"  Per-instance predictions saved to: {pred_csv}")

    print(f"\n{'=' * 50}")
    print(f"CNN Type {'A' if image_type == 'a' else 'B'} — LOPO Summary")
    print(f"  Accuracy:  {_fmt_mean_std(df['acc'])}")
    print(f"  Macro F1:  {_fmt_mean_std(df['macro_f1'])}")
    print("  AUC-ROC:   N/A per fold; each held-out test fold has one true class")
    print(f"  Results saved to: {out_csv}")

    return df
