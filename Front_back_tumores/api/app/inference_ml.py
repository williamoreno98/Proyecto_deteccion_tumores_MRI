from pathlib import Path
from typing import Tuple, Dict
import numpy as np
from PIL import Image
import joblib
from skimage.feature import graycomatrix, graycoprops

APP_DIR = Path(__file__).resolve().parent
ML_DIR  = APP_DIR / "artifacts_ml"

def _features_from_image_bytes(content: bytes) -> np.ndarray:
    img = Image.open(BytesIO(content)).convert("L")
    arr = np.array(img, dtype=np.uint8)
    return _four_features_from_uint8(arr)

def _four_features_from_uint8(arr: np.ndarray) -> np.ndarray:
    brightness = float(arr.mean())
    contrast   = float(arr.std())
    glcm = graycomatrix(arr, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
    energy      = float(graycoprops(glcm, "energy")[0, 0])
    return np.array([brightness, contrast, homogeneity, energy], dtype=np.float32)

def _load_clf_and_le():
    clf = joblib.load(ML_DIR / "classifier.joblib")
    le  = joblib.load(ML_DIR / "label_encoder.joblib")
    return clf, le

from io import BytesIO

def predict_from_image_bytes(content: bytes) -> Tuple[str, Dict[str, float]]:
    clf, le = _load_clf_and_le()
    x = _features_from_image_bytes(content).reshape(1, -1)
    probs = clf.predict_proba(x)[0]
    classes = le.inverse_transform(np.arange(len(probs)))
    # map a tus labels finales
    mapping = {"glioma_tumor":"glioma","meningioma_tumor":"meningioma","no_tumor":"notumor","pituitary_tumor":"pituitary"}
    probs_dict = {mapping.get(c, c): float(p) for c, p in zip(classes, probs)}
    top_class = max(probs_dict, key=probs_dict.get)
    return top_class, probs_dict

def predict_from_features(brightness, contrast, homogeneity, energy):
    clf, le = _load_clf_and_le()
    x = np.array([[brightness, contrast, homogeneity, energy]], dtype=np.float32)
    probs = clf.predict_proba(x)[0]
    classes = le.inverse_transform(np.arange(len(probs)))
    mapping = {"glioma_tumor":"glioma","meningioma_tumor":"meningioma","no_tumor":"notumor","pituitary_tumor":"pituitary"}
    probs_dict = {mapping.get(c, c): float(p) for c, p in zip(classes, probs)}
    top_class = max(probs_dict, key=probs_dict.get)
    return top_class, probs_dict
