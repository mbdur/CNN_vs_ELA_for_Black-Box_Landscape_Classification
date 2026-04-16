"""config.py — Central configuration (paths, hyperparams, label scheme)."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
IMAGES_DIR  = os.path.join(DATA_DIR, "images")
ELA_DIR     = os.path.join(DATA_DIR, "ela_features")
LABELS_DIR  = os.path.join(DATA_DIR, "labels")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for d in [SAMPLES_DIR, IMAGES_DIR, ELA_DIR, LABELS_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# BBOB settings
BBOB_DIM         = 5
BBOB_N_FUNCTIONS = 24
BBOB_N_INSTANCES = 10
BBOB_BOUNDS      = (-5.0, 5.0)

N_SAMPLES = 500
SAMPLE_BUDGETS_BRANCH_B = [50, 100, 200, 500, 1000]

# Label scheme: "bbob_3class" or "modcma_4class"
LABEL_SCHEME = "modcma_4class"
# 3-class: standard BBOB groups
BBOB_GROUPS_3CLASS = {
    "separable_unimodal": [1, 2, 3, 4, 5],
    "low_moderate_conditioning": [6, 7, 8, 9, 10, 11, 12, 13, 14],
    "multimodal": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
}

# 4-class: grouped by best modCMA config (elitist, step_size_adaptation)
# Derived from Zenodo 5947076 performance data
MODCMA_GROUPS_4CLASS = {
    "elitist_csa":     [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14],
    "elitist_psr":     [7, 20, 21, 22],
    "nonelitist_csa":  [3, 4, 15, 16, 17, 18],
    "nonelitist_psr":  [19, 23, 24],
}

if LABEL_SCHEME == "bbob_3class":
    BBOB_GROUPS = BBOB_GROUPS_3CLASS
elif LABEL_SCHEME == "modcma_4class":
    BBOB_GROUPS = MODCMA_GROUPS_4CLASS
else:
    raise ValueError(f"Unknown LABEL_SCHEME: {LABEL_SCHEME}")

FUNCTION_TO_CLASS = {}
CLASS_NAMES = list(BBOB_GROUPS.keys())
for class_idx, (group_name, func_ids) in enumerate(BBOB_GROUPS.items()):
    for fid in func_ids:
        FUNCTION_TO_CLASS[fid] = class_idx

N_CLASSES = len(BBOB_GROUPS)

# Image sizes
IMG_SIZE_A = (128, 128)
IMG_SIZE_B = (192, 256)
N_PAIRS_B  = 6

# CNN architecture
CNN_CHANNELS   = [32, 64, 128, 256]
CNN_DENSE_DIMS = [256, 128]
DROPOUT_RATE   = 0.5

# Training
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
N_EPOCHS      = 50
WEIGHT_DECAY  = 1e-4
RANDOM_SEED   = 42

# ELA feature sets
ELA_FEATURE_SETS = [
    "ela_distribution",
    "ela_meta",
    "nbc",
    "information_content",
    "dispersion",
]
