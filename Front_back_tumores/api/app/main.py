çfrom fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json, time
import numpy as np, time
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from io import BytesIO
from skimage.transform import resize


from .schemas import PredictMLResponse
from .inference_ml import predict_from_image_bytes, predict_from_features

app = FastAPI(title="MRI Tumor API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

APP_DIR = Path(__file__).resolve().parent
ML_DIR  = APP_DIR / "artifacts_ml"

RF_DIR  = APP_DIR / "artifacts_rf"

# Carga perezosa: solo intentamos cuando llamen la ruta RF
_clf_rf = None
_le_rf  = None
_CLASSES_RF = None

def _lazy_load_rf():
    global _clf_rf, _le_rf, _CLASSES_RF
    if _clf_rf is not None and _le_rf is not None:
        return
    clf_p = RF_DIR / "classifier.joblib"
    le_p  = RF_DIR / "label_encoder.joblib"
    if not clf_p.exists() or not le_p.exists():
      # dejamos que la ruta maneje el error 501
        raise FileNotFoundError("RF artifacts missing")
    _clf_rf = joblib.load(clf_p)
    _le_rf  = joblib.load(le_p)
    _CLASSES_RF = list(_le_rf.classes_)

def _rf_features_from_uint8(arr: np.ndarray) -> np.ndarray:
    """Vector 73: [brightness] + 72 GLCM (3 dist x 4 ángulos x 6 props)."""
    img_small = (resize(arr, (256, 256), anti_aliasing=True) * 255).astype(np.uint8)
    brightness = float(img_small.mean())
    distances = [1, 2, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img_small, distances=distances, angles=angles,
                        symmetric=True, normed=True, levels=256)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feats = []
    for p in props:
        vals = graycoprops(glcm, p)  # (3,4)
        feats.append(vals.flatten())
    glcm_vec = np.concatenate(feats, axis=0)  # 72
    return np.concatenate([[brightness], glcm_vec]).astype("float32").reshape(1, -1)




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
    
@app.post("/predict/rf", response_model=PredictMLResponse)
async def predict_rf(file: UploadFile = File(...)):
    """
    Predicción con Random Forest (artefactos en api/app/artifacts_rf/).
    Responde el mismo esquema: model='rf', top_class, probabilities.
    """
    # leer imagen
    try:
        content = await file.read()
        arr = np.array(Image.open(BytesIO(content)).convert("L"), dtype=np.uint8)
    except Exception as e:
        raise HTTPException(400, f"Error procesando imagen: {e}")

    # cargar artefactos RF si existen
    try:
        _lazy_load_rf()
    except FileNotFoundError:
        raise HTTPException(501, "RF artifacts not found. Esperados: artifacts_rf/classifier.joblib y label_encoder.joblib")

    # inferencia
    X = _rf_features_from_uint8(arr)
    yhat = int(_clf_rf.predict(X)[0])
    top  = _CLASSES_RF[yhat]

    probs = {}
    if hasattr(_clf_rf, "predict_proba"):
        p = _clf_rf.predict_proba(X)[0]
        probs = {cls: float(p[i]) for i, cls in enumerate(_CLASSES_RF)}

    return PredictMLResponse(model="rf", top_class=top, probabilities=probs)


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
