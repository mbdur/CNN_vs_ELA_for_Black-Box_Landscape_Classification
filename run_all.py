"""run_all.py — Run the full pipeline or individual phases."""

import argparse
import sys
import os

def run_phase_1():
    print("\n" + "="*60)
    print("PHASE 1: Sampling + Image Rendering + ELA Features")
    print("="*60)

    print("\nStep 1/3: BBOB sampling...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("sampling", "01_sampling.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.run_sampling()

    print("\nStep 2/3: Rendering images...")
    spec = importlib.util.spec_from_file_location("render", "02_render_images.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.render_all_images()

    print("\nStep 3/3: ELA feature extraction...")
    spec = importlib.util.spec_from_file_location("ela", "03_ela_features.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.run_ela_extraction()


def run_phase_2():
    print("\n" + "="*60)
    print("PHASE 2: CNN Training + ELA+RF Baseline")
    print("="*60)

    print("\nStep 1/3: CNN on Type A (PCA images)...")
    spec = importlib.util.spec_from_file_location("train", "05_train_cnn.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.run_lopo_cv(image_type="a")

    print("\nStep 2/3: CNN on Type B (Pairwise images)...")
    m.run_lopo_cv(image_type="b")

    print("\nStep 3/3: ELA + Random Forest baseline...")
    spec = importlib.util.spec_from_file_location("rf", "06_ela_rf_baseline.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.run_ela_rf_lopo()
    m.build_comparison_table()
    m.plot_comparison_bar()


def run_phase_3a():
    print("\n" + "="*60)
    print("PHASE 3A: Grad-CAM Interpretability")
    print("="*60)

    spec = importlib.util.spec_from_file_location("gradcam", "07_gradcam.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.visualise_gradcam_panel([1, 10, 21], image_type="a")
    m.interpret_gradcam_results([1, 10, 21])


def run_phase_3b():
    print("\n" + "="*60)
    print("PHASE 3B: Budget Crossover Experiment")
    print("="*60)

    spec = importlib.util.spec_from_file_location("budget", "08_branch_b_budget.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.run_budget_experiment()
    m.plot_budget_crossover()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Topic 7 pipeline")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "3a", "3b", "all"],
                        help="Which phase to run (default: all)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.phase in ("1", "all"):
        run_phase_1()

    if args.phase in ("2", "all"):
        run_phase_2()

    if args.phase in ("3a", "all"):
        run_phase_3a()

    if args.phase in ("3b",):
        run_phase_3b()

    print("\n" + "="*60)
    print("Pipeline complete.")
    print(f"All results are in the 'results/' directory.")
