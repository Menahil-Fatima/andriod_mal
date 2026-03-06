# Evaluate_SeenVsUnseen.py

import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from SETTINGS import (
    ensure_folders, FAMILIES,
    GRAY_DATASET_DIR, SUPPORT_EMB_DIR,
    TRAINED_MODELS_DIR, DEFAULT_THRESHOLD
)

from MakeEmbedding import make_embedding
from CompareSupport import euclidean, abs_distance
from DecideResult import decide

from Models.Exp1_CNN4 import Exp1_CNN4
from Models.Exp2_CNN6 import Exp2_CNN6
from Models.Exp3_ResNet34 import Exp3_ResNet34

DEVICE = "cpu"

# ✅ Training families (seen classes)
TRAIN_FAMILIES = ["Adware", "Banking", "Riskware"]


def load_model(exp: str):
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

    if not w.exists():
        raise FileNotFoundError(f"Trained weights not found: {w}")

    m.load_state_dict(torch.load(w, map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    return m


def load_support_embeddings(exp: str):
    support = []
    base = SUPPORT_EMB_DIR / exp
    if not base.exists():
        raise FileNotFoundError(f"Support embeddings folder missing: {base}")

    for fam in FAMILIES:
        fam_dir = base / fam
        if not fam_dir.exists():
            continue
        for p in fam_dir.glob("*.npy"):
            support.append((fam, p.stem, np.load(p)))

    if not support:
        raise RuntimeError(f"No support embeddings found in {base}. Run SaveSupportEmbeddings.py first.")

    return support


def support_stems_by_family(exp: str):
    stems = defaultdict(set)
    base = SUPPORT_EMB_DIR / exp
    for fam in FAMILIES:
        fam_dir = base / fam
        if not fam_dir.exists():
            continue
        for p in fam_dir.glob("*.npy"):
            stems[fam].add(p.stem)
    return stems


def predict_one_image(exp: str, model, img_path: Path, support, threshold: float):
    from PIL import Image
    img = Image.open(img_path).convert("L")
    q = make_embedding(model, img, device=DEVICE)

    best = None
    for fam, sample_stem, emb in support:
        if exp == "exp1":
            d = euclidean(q, emb)
        else:
            d = abs_distance(q, emb)
        if best is None or d < best["score"]:
            best = {"family": fam, "matched_sample": sample_stem, "score": float(d)}

    decision = decide(best["family"], best["score"], threshold)
    return decision["family"], best["score"], best["matched_sample"]


def evaluate(exp: str = "exp1", threshold: float | None = None):
    ensure_folders()

    threshold = DEFAULT_THRESHOLD[exp] if threshold is None else float(threshold)
    seen_set = set(TRAIN_FAMILIES)
    unseen_fams = [f for f in FAMILIES if f not in seen_set]

    model = load_model(exp)
    support = load_support_embeddings(exp)
    support_stems = support_stems_by_family(exp)

    total = 0
    correct = 0
    seen_total = 0
    seen_correct = 0
    unseen_total = 0
    unseen_correct = 0

    per_family = defaultdict(lambda: {"total": 0, "correct": 0})
    confusion = defaultdict(lambda: defaultdict(int))

    for true_fam in FAMILIES:
        img_dir = GRAY_DATASET_DIR / true_fam
        if not img_dir.exists():
            continue
        for img_path in img_dir.glob("*.png"):
            if img_path.stem in support_stems[true_fam]:
                continue

            pred_fam, score, matched = predict_one_image(exp, model, img_path, support, threshold)

            total += 1
            per_family[true_fam]["total"] += 1
            confusion[true_fam][pred_fam] += 1

            if pred_fam == true_fam:
                correct += 1
                per_family[true_fam]["correct"] += 1

            if true_fam in seen_set:
                seen_total += 1
                if pred_fam == true_fam:
                    seen_correct += 1
            else:
                unseen_total += 1
                if pred_fam == true_fam:
                    unseen_correct += 1

    def pct(a, b):
        return 0.0 if b == 0 else round((a / b) * 100.0, 2)

    # per-family accuracy results
    per_family_results = {}
    for fam in FAMILIES:
        t = per_family[fam]["total"]
        c = per_family[fam]["correct"]
        per_family_results[fam] = {
            "correct": c,
            "total": t,
            "accuracy": pct(c, t)
        }

    # confusion matrix results
    confusion_results = {}
    for true_fam in FAMILIES:
        confusion_results[true_fam] = dict(confusion[true_fam])

    # return all results as a dictionary
    return {
        "experiment": exp,
        "threshold": threshold,
        "seen_families": list(TRAIN_FAMILIES),
        "unseen_families": unseen_fams,
        "overall_accuracy": pct(correct, total),
        "seen_accuracy": pct(seen_correct, seen_total),
        "unseen_accuracy": pct(unseen_correct, unseen_total),
        "counts": {
            "correct": correct,
            "total": total,
            "seen_correct": seen_correct,
            "seen_total": seen_total,
            "unseen_correct": unseen_correct,
            "unseen_total": unseen_total
        },
        "per_family": per_family_results,
        "confusion": confusion_results
    }


if __name__ == "__main__":
    results = evaluate(exp="exp1", threshold=None)
    print("Returned Results Dictionary:")
    print(results)