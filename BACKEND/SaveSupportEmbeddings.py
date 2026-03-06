import numpy as np
import torch

from SETTINGS import (
    ensure_folders, FAMILIES, SUPPORT_SET_APK_DIR,
    SUPPORT_EMB_DIR, EXP_IMAGE_SIZE, TRAINED_MODELS_DIR
)
from ApkToGrayImage import apk_bytes_to_gray_image
from ResizeImage import resize_gray
from MakeEmbedding import make_embedding

from Models.Exp1_CNN4 import Exp1_CNN4
from Models.Exp2_CNN6 import Exp2_CNN6
from Models.Exp3_ResNet34 import Exp3_ResNet34

def load_model(exp: str, device="cpu"):
    if exp == "exp1":
        m = Exp1_CNN4(128)
        w = TRAINED_MODELS_DIR / "exp1_cnn4.pt"
    elif exp == "exp2":
        m = Exp2_CNN6(128, dropout=0.5)
        w = TRAINED_MODELS_DIR / "exp2_cnn6.pt"
    elif exp == "exp3":
        m = Exp3_ResNet34(128)
        w = TRAINED_MODELS_DIR / "exp3_resnet34.pt"
    else:
        raise ValueError("exp must be exp1/exp2/exp3")

    if w.exists():
        m.load_state_dict(torch.load(w, map_location=device))
    m.to(device)
    return m

def save_support_embeddings(exp="exp1", device="cpu"):
    ensure_folders()
    size = EXP_IMAGE_SIZE[exp]
    model = load_model(exp, device=device)

    for fam in FAMILIES:
        out_dir = SUPPORT_EMB_DIR / exp / fam
        out_dir.mkdir(parents=True, exist_ok=True)

        for apk_path in (SUPPORT_SET_APK_DIR / fam).glob("*.apk"):
            img = apk_bytes_to_gray_image(apk_path.read_bytes())
            img = resize_gray(img, size)
            emb = make_embedding(model, img, device=device)
            np.save(out_dir / f"{apk_path.stem}.npy", emb)

    print("Saved support embeddings for", exp)

if __name__ == "__main__":
    save_support_embeddings("exp1", device="cpu")  # change exp1/exp2/exp3
