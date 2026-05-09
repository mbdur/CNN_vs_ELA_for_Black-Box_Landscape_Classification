
"""
01_sampling.py — LHS sampling on all 24 BBOB functions.
Outputs data/samples/f{func_id}_i{instance}_d{dim}.npz with X and y arrays.
"""

import os
import numpy as np
import cocoex
from scipy.stats import qmc
from tqdm import tqdm
from config import (
    BBOB_DIM, BBOB_N_FUNCTIONS, BBOB_N_INSTANCES,
    BBOB_BOUNDS, N_SAMPLES, SAMPLES_DIR, RANDOM_SEED,
    FUNC_N_INSTANCES
)


def latin_hypercube_sample(n, dim, lower, upper, seed=None):
    """Generate n LHS points in [lower, upper]^dim."""
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    X_unit = sampler.random(n=n)
    X = qmc.scale(X_unit, lower, upper)
    return X


def evaluate_bbob_function(problem, X):
    """Evaluate BBOB problem on all rows of X, return fitness array."""
    return np.array([problem(x) for x in X])


def sample_bbob_instance(func_id, instance, dim, n_samples, bounds, seed=None):
    """Sample one BBOB instance, return dict with X, y arrays."""
    suite = cocoex.Suite(
        "bbob",
        f"instances:{instance}",
        f"function_indices:{func_id} dimensions:{dim}"
    )

    problem = next(iter(suite))
    lower, upper = bounds

    X = latin_hypercube_sample(n_samples, dim, lower, upper, seed=seed)
    y = evaluate_bbob_function(problem, X)

    problem.free()
    suite.free()

    return {"X": X, "y": y, "func_id": func_id, "instance": instance, "dim": dim}


def run_sampling(n_samples=N_SAMPLES, dim=BBOB_DIM,
                 n_functions=BBOB_N_FUNCTIONS,
                 func_n_instances=FUNC_N_INSTANCES,
                 bounds=BBOB_BOUNDS, out_dir=SAMPLES_DIR,
                 base_seed=RANDOM_SEED, overwrite=False):
    """Sample all BBOB functions with per-function instance counts, save .npz files."""
    os.makedirs(out_dir, exist_ok=True)
    total = sum(func_n_instances.get(f, 10) for f in range(1, n_functions + 1))
    print(f"\nSampling {n_functions} BBOB functions ({total} total instances) "
          f"of {n_samples} points each in {dim}D")
    print(f"Output directory: {out_dir}\n")

    with tqdm(total=total, desc="Sampling") as pbar:
        for func_id in range(1, n_functions + 1):
            n_inst = func_n_instances.get(func_id, 10)
            for instance in range(1, n_inst + 1):
                fname = f"f{func_id:02d}_i{instance:02d}_d{dim}.npz"
                fpath = os.path.join(out_dir, fname)

                if os.path.exists(fpath) and not overwrite:
                    pbar.set_postfix({"status": f"skip f{func_id} i{instance}"})
                    pbar.update(1)
                    continue

                seed = base_seed + (func_id - 1) * 100 + (instance - 1)

                data = sample_bbob_instance(
                    func_id=func_id,
                    instance=instance,
                    dim=dim,
                    n_samples=n_samples,
                    bounds=bounds,
                    seed=seed
                )

                np.savez_compressed(
                    fpath,
                    X=data["X"],
                    y=data["y"],
                    func_id=np.int32(func_id),
                    instance=np.int32(instance),
                    dim=np.int32(dim)
                )

                pbar.set_postfix({"saved": fname})
                pbar.update(1)

    print(f"\nDone. Up to {total} files saved to {out_dir}")


def load_sample(func_id, instance, dim=BBOB_DIM, out_dir=SAMPLES_DIR):
    """Load a saved sample .npz file."""
    fname = f"f{func_id:02d}_i{instance:02d}_d{dim}.npz"
    fpath = os.path.join(out_dir, fname)
    data = np.load(fpath)
    return {
        "X": data["X"],
        "y": data["y"],
        "func_id": int(data["func_id"]),
        "instance": int(data["instance"]),
        "dim": int(data["dim"])
    }


if __name__ == "__main__":
    run_sampling()
    # Quick sanity check: load one file and print its shape
    sample = load_sample(func_id=1, instance=1)
    print(f"\nSanity check — f01 i01:")
    print(f"  X shape: {sample['X'].shape}   (should be ({N_SAMPLES}, {BBOB_DIM}))")
    print(f"  y shape: {sample['y'].shape}   (should be ({N_SAMPLES},))")
    print(f"  y range: [{sample['y'].min():.2f}, {sample['y'].max():.2f}]")
