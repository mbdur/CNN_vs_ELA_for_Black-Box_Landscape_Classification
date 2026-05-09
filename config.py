
"""config.py -- Central configuration (paths, hyperparams, label scheme)."""

import os

BASE_DIR = os.environ.get("CNN_ALG_SEL_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = os.environ.get("CNN_ALG_SEL_WORK_DIR", BASE_DIR)
# Persistent experiment outputs live under BASE_DIR (project root).
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
BBOB_BOUNDS      = (-5.0, 5.0)

N_SAMPLES = 500
SAMPLE_BUDGETS_BRANCH_B = [50, 100, 200, 500, 1000]

# Label scheme: "bbob_3class" or "modcma_4class"
LABEL_SCHEME = "bbob_3class"
# 3-class: standard BBOB groups
BBOB_GROUPS_3CLASS = {
    "separable_unimodal": [1, 2, 3, 4, 5],
    "low_moderate_conditioning": [6, 7, 8, 9, 10, 11, 12, 13, 14],
    "multimodal": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
}

if LABEL_SCHEME == "bbob_3class":
    BBOB_GROUPS = BBOB_GROUPS_3CLASS
else:
    raise ValueError(f"Unknown LABEL_SCHEME: {LABEL_SCHEME}")

FUNCTION_TO_CLASS = {}
CLASS_NAMES = list(BBOB_GROUPS.keys())
for class_idx, (group_name, func_ids) in enumerate(BBOB_GROUPS.items()):
    for fid in func_ids:
        FUNCTION_TO_CLASS[fid] = class_idx

N_CLASSES = len(BBOB_GROUPS)

# Flat 10 instances per function (original spec, no balance-up)
BBOB_N_INSTANCES = 10
FUNC_N_INSTANCES = {fid: 10 for fid in range(1, BBOB_N_FUNCTIONS + 1)}

# Image sizes
IMG_SIZE_A = (128, 128)
IMG_SIZE_B = (192, 256)
N_PAIRS_B  = 6

# CNN architecture — reduced capacity to avoid overfitting on small dataset
CNN_CHANNELS   = [64, 128]
CNN_DENSE_DIMS = [128]
DROPOUT_RATE   = 0.5

# Training
BATCH_SIZE    = 16
LEARNING_RATE = 1e-4
N_EPOCHS      = 50
WEIGHT_DECAY  = 1e-3
RANDOM_SEED   = 42

# ELA feature sets
ELA_FEATURE_SETS = [
    "ela_distribution",
    "ela_meta",
    "ela_level",
    "nbc",
    "information_content",
    "dispersion",
]
