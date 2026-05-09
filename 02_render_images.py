
"""
02_render_images.py — Render sample point clouds as PNG images.
Type A: PCA scatter (128x128), Type B: pairwise dimension grid (256x192).
Both built only from (X, f(X)) pairs (respects black-box constraint).

Uses rank-based color normalization so the full viridis colormap is always
used, regardless of how narrow or skewed the fitness distribution is.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving to file
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from tqdm import tqdm
from config import (
    BBOB_DIM, BBOB_N_FUNCTIONS, BBOB_N_INSTANCES,
    IMG_SIZE_A, IMG_SIZE_B, N_PAIRS_B,
    SAMPLES_DIR, IMAGES_DIR, FUNC_N_INSTANCES
)
from pathlib import Path


def normalize_log_fitness(y):
    """Legacy: Map fitness values to [0, 1] via log normalisation."""
    y_shifted = y - y.min() + 1e-10
    y_log = np.log(y_shifted)
    y_norm = (y_log - y_log.min()) / (y_log.max() - y_log.min() + 1e-10)
    return y_norm


def normalize_rank_fitness(y):
    """Map fitness values to [0, 1] via rank normalisation.

    Each point gets a color proportional to its rank among all N samples.
    Rank 0 (best fitness) -> 0.0 (dark purple in viridis)
    Rank N-1 (worst fitness) -> 1.0 (bright yellow in viridis)

    This guarantees full colormap usage regardless of the fitness
    distribution shape — no more all-yellow images.
    """
    ranks = np.argsort(np.argsort(y)).astype(np.float64)
    return ranks / max(len(y) - 1, 1)


def render_type_a(X, y, size=IMG_SIZE_A, colormap="viridis"):
    """Render PCA scatter image (Type A). Returns uint8 RGB array."""
    H, W = size
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    c = normalize_rank_fitness(y)

    dpi = 100
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=c, cmap=colormap,
               s=8, linewidths=0, alpha=0.85)
    ax.set_axis_off()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    plt.tight_layout(pad=0)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )
    img_rgb = img_rgba[:, :, :3]
    plt.close(fig)
    from PIL import Image
    img_pil = Image.fromarray(img_rgb).resize((W, H), Image.LANCZOS)
    return np.array(img_pil)


def render_type_b(X, y, n_pairs=N_PAIRS_B, size=IMG_SIZE_B, colormap="viridis"):
    """Render pairwise scatter grid (Type B). Returns uint8 RGB array."""
    H, W = size

    n_dims_used = min(4, X.shape[1])
    pairs = [(i, j) for i in range(n_dims_used)
             for j in range(i + 1, n_dims_used)][:n_pairs]

    n_cols = 3
    n_rows = 2
    c = normalize_rank_fitness(y)

    dpi = 100
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(W / dpi, H / dpi), dpi=dpi)
    axes = axes.flatten()
    fig.patch.set_facecolor("black")

    for idx, (di, dj) in enumerate(pairs):
        ax = axes[idx]
        ax.scatter(X[:, di], X[:, dj], c=c, cmap=colormap,
                   s=6, linewidths=0, alpha=0.8)
        ax.set_axis_off()
        ax.set_facecolor("black")
        ax.set_title(f"d{di+1} vs d{dj+1}", color="white", fontsize=6, pad=2)

    for idx in range(len(pairs), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(pad=0.1)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )
    img_rgb = img_rgba[:, :, :3]
    plt.close(fig)

    from PIL import Image
    img_pil = Image.fromarray(img_rgb).resize((W, H), Image.LANCZOS)
    return np.array(img_pil)


def save_image(img, out_path):
    """Save uint8 RGB array as PNG."""
    from PIL import Image
    Image.fromarray(img).save(out_path)


def render_all_images(n_functions=BBOB_N_FUNCTIONS,
                      func_n_instances=FUNC_N_INSTANCES,
                      dim=BBOB_DIM, samples_dir=SAMPLES_DIR,
                      images_dir=IMAGES_DIR, overwrite=False):
    """Render Type A and Type B images for all function/instance pairs."""
    dir_a = os.path.join(images_dir, "type_a")
    dir_b = os.path.join(images_dir, "type_b")
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)

    total = sum(func_n_instances.get(f, 10) for f in range(1, n_functions + 1))
    print(f"\nRendering images for {n_functions} functions ({total} total instances)")
    print(f"  Type A (PCA scatter): {IMG_SIZE_A[1]}x{IMG_SIZE_A[0]} px -> {dir_a}")
    print(f"  Type B (Pairwise):    {IMG_SIZE_B[1]}x{IMG_SIZE_B[0]} px -> {dir_b}")
    print(f"  Color normalization:  RANK-BASED (full colormap guaranteed)\n")

    with tqdm(total=total, desc="Rendering") as pbar:
        for func_id in range(1, n_functions + 1):
            n_inst = func_n_instances.get(func_id, 10)
            for instance in range(1, n_inst + 1):
                stem = f"f{func_id:02d}_i{instance:02d}_d{dim}"
                path_a = os.path.join(dir_a, stem + ".png")
                path_b = os.path.join(dir_b, stem + ".png")

                # Skip if both exist and not overwriting
                if os.path.exists(path_a) and os.path.exists(path_b) and not overwrite:
                    pbar.update(1)
                    continue

                # Load sample
                npz_path = os.path.join(samples_dir, stem + ".npz")
                if not os.path.exists(npz_path):
                    print(f"  WARNING: sample not found: {npz_path} — run 01_sampling.py first")
                    pbar.update(1)
                    continue

                data = np.load(npz_path)
                X, y = data["X"], data["y"]

                # Render and save Type A
                if not os.path.exists(path_a) or overwrite:
                    img_a = render_type_a(X, y)
                    save_image(img_a, path_a)

                # Render and save Type B
                if not os.path.exists(path_b) or overwrite:
                    img_b = render_type_b(X, y)
                    save_image(img_b, path_b)

                pbar.set_postfix({"last": stem})
                pbar.update(1)

    print(f"\nDone. Images saved to {images_dir}")


if __name__ == "__main__":
    render_all_images()

    # Sanity check
    import matplotlib.pyplot as plt
    from PIL import Image

    stem = f"f01_i01_d{BBOB_DIM}"
    path_a = os.path.join(IMAGES_DIR, "type_a", stem + ".png")
    path_b = os.path.join(IMAGES_DIR, "type_b", stem + ".png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(Image.open(path_a))
    axes[0].set_title("Type A — PCA Scatter (f01 i01)")
    axes[0].axis("off")
    axes[1].imshow(Image.open(path_b))
    axes[1].set_title("Type B — Pairwise Grid (f01 i01)")
    axes[1].axis("off")
    plt.tight_layout()
    out_check = os.path.join(IMAGES_DIR, "sanity_check_f01.png")
    plt.savefig(out_check, dpi=100, bbox_inches="tight")
    print(f"Sanity check image saved to: {out_check}")
    plt.close()
