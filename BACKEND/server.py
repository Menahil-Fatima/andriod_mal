from contextlib import asynccontextmanager
from Evaluate_SeenVsUnseen import evaluate
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from Evaluate_SeenVsUnseen import evaluate
from SETTINGS import ensure_folders, TRAINED_MODELS_DIR
from ScanUserApk import scan_apk

from Models.Exp1_CNN4 import Exp1_CNN4
from Models.Exp2_CNN6 import Exp2_CNN6
from Models.Exp3_ResNet34 import Exp3_ResNet34


from SETTINGS import ensure_folders, TRAINED_MODELS_DIR
from ScanUserApk import scan_apk

from Models.Exp1_CNN4 import Exp1_CNN4
from Models.Exp2_CNN6 import Exp2_CNN6
from Models.Exp3_ResNet34 import Exp3_ResNet34

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic (runs before server starts)
    ensure_folders()
    print("Server starting up...")
    yield
    # Shutdown logic (runs when server stops)
    print("Server shutting down...")

app = FastAPI(title="Android Malware Detection (Thesis)", lifespan=lifespan)
DEVICE = "cpu"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Malware Detection API Running"}

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
        raise ValueError("experiment must be exp1/exp2/exp3")

    if w.exists():
        m.load_state_dict(torch.load(w, map_location=DEVICE))
    m.to(DEVICE)
    return m


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/scan")
async def scan(
    apk: UploadFile = File(...),
    experiment: str = Form("exp1"),
    threshold: float | None = Form(None),
):
    b = await apk.read()
    model = load_model(experiment)
    result = scan_apk(b, experiment, model, threshold=threshold, device=DEVICE)
    return result

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("validation error:", exc.errors())
    print("body:", exc.body)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@app.get("/performance")
def performance():
    try:
        exp1 = evaluate("exp1")
        return {
            "exp1": exp1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

