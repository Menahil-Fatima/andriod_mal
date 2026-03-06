import torch
from pathlib import Path
from ScanUserApk import scan_apk
from Models.Exp1_CNN4 import Exp1_CNN4

# ===== SET YOUR PATHS HERE =====
MODEL_PATH = Path(r"D:\fyp\FYP_Android_Malware_Thesis\BACKEND\DATA\TRAINED_MODELS\exp1_cnn4.pt")
APK_PATH = Path(r"D:\fyp\FYP_Android_Malware_Thesis\BACKEND\DATA\SUPPORT_SET_APK\Banking\ d91b9e71b01460207b2275e5bc6cf6f8b91e3963b6f837fec3ae6e98725866ba.apk")
# ===============================

# --- Verify model file ---
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# --- Load model ---
print("Loading model...")
model = Exp1_CNN4()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("Model loaded.")

# --- Verify APK file ---
if not APK_PATH.exists():
    raise FileNotFoundError(f"APK file not found: {APK_PATH}")
if not APK_PATH.is_file():
    raise ValueError(f"Not a file (maybe a directory?): {APK_PATH}")

# --- Read APK as bytes (this is the key fix!) ---
print(f"Reading APK: {APK_PATH.name}")
apk_bytes = APK_PATH.read_bytes()   # returns bytes, not string

# --- Run scan ---
result = scan_apk(apk_bytes, "exp1", model)

# --- Show result ---
print("\n=== SCAN RESULT ===")
print(f"Experiment:     {result['experiment']}")
print(f"Threshold:      {result['threshold']}")
print(f"Best match:     {result['best_match']['family']} (sample: {result['best_match']['matched_sample']})")
print(f"Distance score: {result['best_match']['score']:.4f}")
print(f"Decision:       {result['decision']}")