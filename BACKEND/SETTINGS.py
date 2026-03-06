from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FAMILIES = ["Adware", "Banking", "Benign", "Riskware", "SMSware"]
TRAIN_FAMILIES_DEFAULT = ["Adware", "Banking", "Riskware"]

DATA_DIR = BASE_DIR / "DATA"
RAW_APK_DIR = DATA_DIR / "RAW_APK"
SUPPORT_SET_APK_DIR = DATA_DIR / "SUPPORT_SET_APK"

GRAY_DIR = DATA_DIR / "GRAY_IMAGES"
GRAY_DATASET_DIR = GRAY_DIR / "dataset"
GRAY_SUPPORT_DIR = GRAY_DIR / "support"
GRAY_USER_DIR = GRAY_DIR / "user"

SUPPORT_EMB_DIR = DATA_DIR / "SUPPORT_EMBEDDINGS"
TRAINED_MODELS_DIR = DATA_DIR / "TRAINED_MODELS"
REPORTS_DIR = DATA_DIR / "REPORTS"

EXP_IMAGE_SIZE = {"exp1": 105, "exp2": 240, "exp3": 300}
DEFAULT_THRESHOLD = {"exp1": 0.30, "exp2": 0.30, "exp3": 0.30}

EXPERIMENTS = ["exp1", "exp2", "exp3"]

def ensure_folders():
    for p in [
        DATA_DIR,
        RAW_APK_DIR,
        SUPPORT_SET_APK_DIR,
        GRAY_DIR,
        GRAY_DATASET_DIR,
        GRAY_SUPPORT_DIR,
        GRAY_USER_DIR,
        SUPPORT_EMB_DIR,
        TRAINED_MODELS_DIR,
        REPORTS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    for fam in FAMILIES:
        (RAW_APK_DIR / fam).mkdir(parents=True, exist_ok=True)
        (SUPPORT_SET_APK_DIR / fam).mkdir(parents=True, exist_ok=True)

        for exp in EXPERIMENTS:
            (GRAY_DATASET_DIR / exp / fam).mkdir(parents=True, exist_ok=True)
            (GRAY_SUPPORT_DIR / exp / fam).mkdir(parents=True, exist_ok=True)
            (SUPPORT_EMB_DIR / exp / fam).mkdir(parents=True, exist_ok=True)

    for exp in EXPERIMENTS:
        (GRAY_USER_DIR / exp).mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / exp).mkdir(parents=True, exist_ok=True)

def model_weight_path(exp: str):
    if exp == "exp1":
        return TRAINED_MODELS_DIR / "exp1_cnn4.pt"
    if exp == "exp2":
        return TRAINED_MODELS_DIR / "exp2_cnn6.pt"
    if exp == "exp3":
        return TRAINED_MODELS_DIR / "exp3_resnet34.pt"
    raise ValueError(f"Unknown experiment: {exp}")

def history_path(exp: str):
    return REPORTS_DIR / exp / "training_history.json"

def evaluation_path(exp: str):
    return REPORTS_DIR / exp / "evaluation.json"
