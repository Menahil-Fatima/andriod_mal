from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FAMILIES = ["Adware", "Banking", "Benign", "Riskware", "SMSware"]

DATA_DIR = BASE_DIR / "DATA"
RAW_APK_DIR = DATA_DIR / "RAW_APK"
SUPPORT_SET_APK_DIR = DATA_DIR / "SUPPORT_SET_APK"

GRAY_DIR = DATA_DIR / "GRAY_IMAGES"
GRAY_DATASET_DIR = GRAY_DIR / "dataset"
GRAY_SUPPORT_DIR = GRAY_DIR / "support"
GRAY_USER_DIR = GRAY_DIR / "user"

SUPPORT_EMB_DIR = DATA_DIR / "SUPPORT_EMBEDDINGS"
TRAINED_MODELS_DIR = DATA_DIR / "TRAINED_MODELS"

# Thesis experiment sizes
EXP_IMAGE_SIZE = {"exp1": 105, "exp2": 240, "exp3": 300}

# Default thresholds (tune later via validation)
DEFAULT_THRESHOLD = {"exp1": 0.30, "exp2": 0.30, "exp3": 0.30}

def ensure_folders():
    for p in [
        DATA_DIR, RAW_APK_DIR, SUPPORT_SET_APK_DIR,
        GRAY_DIR, GRAY_DATASET_DIR, GRAY_SUPPORT_DIR, GRAY_USER_DIR,
        SUPPORT_EMB_DIR, TRAINED_MODELS_DIR
    ]:
        p.mkdir(parents=True, exist_ok=True)

    for fam in FAMILIES:
        (RAW_APK_DIR / fam).mkdir(parents=True, exist_ok=True)
        (SUPPORT_SET_APK_DIR / fam).mkdir(parents=True, exist_ok=True)

        (GRAY_DATASET_DIR / fam).mkdir(parents=True, exist_ok=True)
        (GRAY_SUPPORT_DIR / fam).mkdir(parents=True, exist_ok=True)

    for exp in ["exp1", "exp2", "exp3"]:
        for fam in FAMILIES:
            (SUPPORT_EMB_DIR / exp / fam).mkdir(parents=True, exist_ok=True)
