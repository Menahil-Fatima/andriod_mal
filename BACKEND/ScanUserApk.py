import numpy as np
from SETTINGS import ensure_folders, FAMILIES, SUPPORT_EMB_DIR, EXP_IMAGE_SIZE, DEFAULT_THRESHOLD
from ApkToGrayImage import apk_bytes_to_gray_image
from ResizeImage import resize_gray
from MakeEmbedding import make_embedding
from CompareSupport import euclidean, abs_distance
from DecideResult import decide

def load_support_embeddings(exp: str):
    items = []
    base = SUPPORT_EMB_DIR / exp
    for fam in FAMILIES:
        for p in (base / fam).glob("*.npy"):
            items.append((fam, p.stem, np.load(p)))
    if not items:
        raise RuntimeError("Support embeddings not found. Run SaveSupportEmbeddings.py first.")
    return items

def scan_apk(apk_bytes: bytes, exp: str, model, threshold=None, device="cpu"):
    ensure_folders()
    size = EXP_IMAGE_SIZE[exp]
    threshold = DEFAULT_THRESHOLD[exp] if threshold is None else float(threshold)

    img = apk_bytes_to_gray_image(apk_bytes)
    img = resize_gray(img, size)
    q = make_embedding(model, img, device=device)

    support = load_support_embeddings(exp)

    best = None
    for fam, sample, s in support:
        if exp == "exp1":
            d = euclidean(q, s)      # thesis-like for exp1
        else:
            d = abs_distance(q, s)   # thesis-like for exp2/exp3
        if best is None or d < best["score"]:
            best = {"family": fam, "matched_sample": sample, "score": d}

    decision = decide(best["family"], best["score"], threshold)
    return {"experiment": exp, "threshold": threshold, "best_match": best, "decision": decision}
