from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json, time
import numpy as np, time
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

from .schemas import PredictMLResponse
from .inference_ml import predict_from_image_bytes, predict_from_features

app = FastAPI(title="MRI Tumor API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

"""app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)"""

APP_DIR = Path(__file__).resolve().parent
ML_DIR  = APP_DIR / "artifacts_ml"


@app.get("/health")
def health():
    return {"status": "ok", "service": "ml"}

@app.post("/predict/ml", response_model=PredictMLResponse)
async def predict_ml(file: UploadFile = File(...)):
    try:
        content = await file.read()
        top_class, probs = predict_from_image_bytes(content)
        return PredictMLResponse(top_class=top_class, probabilities=probs)
    except Exception as e:
        raise HTTPException(400, f"Error procesando imagen: {e}")

@app.post("/predict/ml-features", response_model=PredictMLResponse)
async def predict_ml_features(
    brightness: float = Form(...),
    contrast: float = Form(...),
    homogeneity: float = Form(...),
    energy: float = Form(...),
):
    try:
        top_class, probs = predict_from_features(brightness, contrast, homogeneity, energy)
        return PredictMLResponse(top_class=top_class, probabilities=probs)
    except Exception as e:
        raise HTTPException(400, f"Error con features: {e}")

@app.get("/metrics/ml")
def metrics_ml():
    p = ML_DIR / "metrics_ml.json"
    if not p.exists():
        return {"precision": None, "recall": None, "f1": None, "inference_ms": None}
    return json.loads(p.read_text())

# Recalcular métricas (sobre Testing/)
RAW = ["glioma_tumor","meningioma_tumor","no_tumor","pituitary_tumor"]
MAP = {"glioma_tumor":"glioma","meningioma_tumor":"meningioma","no_tumor":"notumor","pituitary_tumor":"pituitary"}

def _four_features_from_uint8(arr: np.ndarray) -> np.ndarray:
    brightness = float(arr.mean())
    contrast   = float(arr.std())
    glcm = graycomatrix(arr, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
    energy      = float(graycoprops(glcm, "energy")[0, 0])
    return np.array([brightness, contrast, homogeneity, energy], dtype=np.float32)

@app.post("/metrics/ml/refresh")
def refresh_ml_metrics():
    clf_p = ML_DIR / "classifier.joblib"
    le_p  = ML_DIR / "label_encoder.joblib"
    if not clf_p.exists():
        raise HTTPException(500, f"No existe {clf_p}")
    clf = joblib.load(clf_p)
    le  = joblib.load(le_p) if le_p.exists() else None

    repo_root = APP_DIR.parents[2]
    test_root = repo_root / "Testing"
    if not test_root.exists():
        raise HTTPException(404, f"No existe {test_root}")

    X, y = [], []
    for raw in RAW:
        cls_dir = test_root / raw
        if not cls_dir.exists(): continue
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".tif",".tiff"):
                arr = np.array(Image.open(p).convert("L"), dtype=np.uint8)
                X.append(_four_features_from_uint8(arr))
                y.append(MAP[raw])
    if not X:
        raise HTTPException(404, "Sin imágenes en Testing/")
    X = np.stack(X)
    from sklearn.preprocessing import LabelEncoder
    if le is None: le = LabelEncoder().fit(y)
    y_true = le.transform(y)

    t0 = time.perf_counter()
    y_pred = clf.predict(X)
    elapsed = time.perf_counter() - t0
    infer_ms = float((elapsed / len(X)) * 1000)

    from sklearn.metrics import precision_score, recall_score, f1_score
    metrics = {
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true,  y_pred, average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true,     y_pred, average="macro", zero_division=0)),
        "inference_ms": infer_ms,
    }
    ML_DIR.mkdir(parents=True, exist_ok=True)
    (ML_DIR / "metrics_ml.json").write_text(json.dumps(metrics, indent=2))
    return metrics
