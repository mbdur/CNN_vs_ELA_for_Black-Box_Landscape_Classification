"""
07_gradcam.py — Grad-CAM visualisation for trained CNN models.
Overlays activation heatmaps on landscape images to show what the CNN attends to.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import (
    BBOB_DIM, BBOB_N_FUNCTIONS, BBOB_N_INSTANCES,
    IMG_SIZE_A, N_CLASSES, CLASS_NAMES,
    FUNCTION_TO_CLASS, IMAGES_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_SEED
)
import importlib as _il
_cnn_mod = _il.import_module("04_cnn_model")
build_model = _cnn_mod.build_model
LandscapeCNN = _cnn_mod.LandscapeCNN


# BBOB function names for readable labelling
BBOB_FUNCTION_NAMES = {
    1:  "Sphere",
    2:  "Ellipsoidal",
    3:  "Rastrigin",
    4:  "Büche-Rastrigin",
    5:  "Linear Slope",
    6:  "Attractive Sector",
    7:  "Step Ellipsoidal",
    8:  "Rosenbrock (orig.)",
    9:  "Rosenbrock (rotated)",
    10: "Ellipsoidal (rotated)",
    11: "Discus",
    12: "Bent Cigar",
    13: "Sharp Ridge",
    14: "Different Powers",
    15: "Rastrigin (rotated)",
    16: "Weierstrass",
    17: "Schaffer F7",
    18: "Schaffer F7 (moderate)",
    19: "Griewank-Rosenbrock",
    20: "Schwefel",
    21: "Gallagher 101 peaks",
    22: "Gallagher 21 peaks",
    23: "Katsuura",
    24: "Lunacek bi-Rastrigin",
}


def load_image_tensor(func_id, instance, image_type="a", dim=BBOB_DIM,
                       img_size=IMG_SIZE_A, device=torch.device("cpu")):
    """Load a single image as normalised tensor + raw float version for overlay."""
    subdir = "type_a" if image_type == "a" else "type_b"
    stem   = f"f{func_id:02d}_i{instance:02d}_d{dim}"
    path   = os.path.join(IMAGES_DIR, subdir, stem + ".png")

    H, W = img_size
    transform = T.Compose([
        T.Resize((H, W)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img_pil    = Image.open(path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    img_float  = np.array(img_pil.resize((W, H))).astype(np.float32) / 255.0

    return img_tensor, img_float


def compute_gradcam(model, img_tensor, target_class=None):
    """Run Grad-CAM, return heatmap array (H, W) in [0, 1]."""
    target_layers = [model.conv_blocks[-1].conv]

    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

    return grayscale_cam[0]


def visualise_gradcam_panel(func_ids, image_type="a", dim=BBOB_DIM,
                              models_dir=MODELS_DIR, results_dir=RESULTS_DIR,
                              instance=1):
    """Create a panel of Grad-CAM overlays for the given BBOB functions."""
    device = torch.device("cpu")
    img_size = IMG_SIZE_A if image_type == "a" else (192, 256)

    n = len(func_ids)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, func_id in enumerate(func_ids):
        class_idx   = FUNCTION_TO_CLASS[func_id]
        func_name   = BBOB_FUNCTION_NAMES.get(func_id, f"f{func_id}")
        class_name  = CLASS_NAMES[class_idx]

        # Load checkpoint for this fold
        ckpt_path = os.path.join(
            models_dir,
            f"cnn_type{image_type}_fold{func_id:02d}.pt"
        )
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}")
            continue

        model = build_model().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        img_tensor, img_float = load_image_tensor(
            func_id=func_id, instance=instance,
            image_type=image_type, dim=dim,
            img_size=img_size, device=device
        )

        with torch.no_grad():
            logits = model(img_tensor)
            pred_class = logits.argmax(dim=1).item()
            pred_conf  = torch.softmax(logits, dim=1)[0, pred_class].item()

        cam_map = compute_gradcam(model, img_tensor, target_class=pred_class)
        visualization = show_cam_on_image(img_float, cam_map, use_rgb=True)

        axes[0, col].imshow(img_float)
        axes[0, col].set_title(
            f"f{func_id}: {func_name}\n(class: {class_name})",
            fontsize=9
        )
        axes[0, col].axis("off")

        is_correct = pred_class == class_idx
        pred_name  = CLASS_NAMES[pred_class]
        axes[1, col].imshow(visualization)
        result_str = "✓" if is_correct else "✗"
        axes[1, col].set_title(
            f"Grad-CAM | pred: {pred_name} ({pred_conf:.2f}) {result_str}",
            fontsize=9,
            color="green" if is_correct else "red"
        )
        axes[1, col].axis("off")

    plt.suptitle(
        f"Grad-CAM Interpretability — Type {'A (PCA)' if image_type == 'a' else 'B (Pairwise)'}",
        fontsize=13, y=1.01
    )
    plt.tight_layout()

    out_path = os.path.join(
        results_dir,
        f"gradcam_panel_type{image_type}_f{'_'.join(str(f) for f in func_ids)}.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM panel saved to: {out_path}")
    return out_path


def interpret_gradcam_results(func_ids):
    """Print class names for the given functions (interpretation is in the report)."""
    print("\n=== Grad-CAM functions analysed ===")
    for fid in func_ids:
        cname = CLASS_NAMES[FUNCTION_TO_CLASS[fid]]
        fname = BBOB_FUNCTION_NAMES.get(fid, f"f{fid}")
        print(f"  f{fid}: {fname} ({cname})")


if __name__ == "__main__":
    # One representative from each major group
    target_funcs = [1, 10, 21]
    print(f"\nGenerating Grad-CAM for functions: {target_funcs}")
    visualise_gradcam_panel(target_funcs, image_type="a")
    interpret_gradcam_results(target_funcs)
