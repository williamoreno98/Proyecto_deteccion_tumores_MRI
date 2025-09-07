# ==========================================
# detection_model_XGB.py
# Entrenamiento de XGBoost con MLflow
# ==========================================
import os
from pathlib import Path
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from skimage.feature import hog, local_binary_pattern
from scipy.stats import skew, kurtosis

# XGBoost
from xgboost import XGBClassifier

# MLflow
import mlflow
import mlflow.sklearn

# -----------------------------
# 1) Configuración general
# -----------------------------
RANDOM_STATE = 42
IMG_SIZE = 50
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
LBP_RADIUS = 2
LBP_POINTS = LBP_RADIUS * 8
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

BASE_DIR = Path("data/archive")
TRAIN_DIR = BASE_DIR / "Training"
TEST_DIR = BASE_DIR / "Testing"

print("Usando dataset en:", BASE_DIR)

# -----------------------------
# 2) Funciones auxiliares
# -----------------------------
def preprocess_image(path, img_size=IMG_SIZE):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No pude leer la imagen: {path}")
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def extract_features(img):
    # HOG
    hog_feat = hog(
        img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    # LBP
    lbp = local_binary_pattern((img * 255).astype("uint8"), P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    n_bins = int(LBP_POINTS + 2)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    # Estadísticas globales
    flat = img.ravel()
    stats = np.array([flat.mean(), flat.std(), skew(flat), kurtosis(flat)])
    return np.concatenate([hog_feat, lbp_hist, stats])


def build_dataset(root_dir):
    X, y, paths = [], [], []
    for cls in CLASSES:
        cls_dir = Path(root_dir) / cls
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                try:
                    img = preprocess_image(p)
                    feat = extract_features(img)
                    X.append(feat)
                    y.append(cls)
                    paths.append(p)
                except Exception as e:
                    print("Saltando", p, "->", e)
    return np.array(X), np.array(y), paths


# -----------------------------
# 3) Main
# -----------------------------
def main():
    print("Extrayendo características de TRAIN...")
    X_train_all, y_train_all, _ = build_dataset(TRAIN_DIR)

    print("Extrayendo características de TEST...")
    X_test_all, y_test_all, _ = build_dataset(TEST_DIR)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_all)
    y_test_enc = le.transform(y_test_all)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_all, y_train_enc, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y_train_enc
    )

    # -----------------------------
    # 4) Experimentos XGB
    # -----------------------------
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 10]
    }

    results_xgb = []

    for n_est in param_grid["n_estimators"]:
        for depth in param_grid["max_depth"]:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                    n_jobs=-1
                ))
            ])

            model.fit(X_tr, y_tr)
            y_val_pred = model.predict(X_val)

            acc = accuracy_score(y_val, y_val_pred)
            f1 = f1_score(y_val, y_val_pred, average="weighted")

            run_result = {"n_estimators": n_est, "max_depth": depth,
                          "accuracy": acc, "f1": f1}
            results_xgb.append(run_result)

            # Log en MLflow
            with mlflow.start_run(run_name=f"XGB_{n_est}_depth{depth}"):
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_param("max_depth", depth)
                mlflow.log_metric("val_accuracy", acc)
                mlflow.log_metric("val_f1", f1)
                mlflow.sklearn.log_model(model, "xgb_model")

            print(f"XGB(n_estimators={n_est}, depth={depth}) -> Acc: {acc:.4f}, F1: {f1:.4f}")

    # -----------------------------
    # 5) Mejor modelo en test
    # -----------------------------
    best_run = max(results_xgb, key=lambda x: x["accuracy"])
    print("\nMejor configuración en validación:", best_run)

    best_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=best_run["n_estimators"],
            max_depth=best_run["max_depth"],
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1
        ))
    ])

    best_model.fit(X_train_all, y_train_enc)
    y_test_pred = best_model.predict(X_test_all)

    acc_test = accuracy_score(y_test_enc, y_test_pred)
    f1_test = f1_score(y_test_enc, y_test_pred, average="weighted")

    print(f"\n=== Resultados en TEST ===")
    print(f"Accuracy: {acc_test:.4f}")
    print(f"F1-score: {f1_test:.4f}")

    with mlflow.start_run(run_name="XGB_FINAL_BEST"):
        mlflow.log_param("n_estimators", best_run["n_estimators"])
        mlflow.log_param("max_depth", best_run["max_depth"])
        mlflow.log_metric("test_accuracy", acc_test)
        mlflow.log_metric("test_f1", f1_test)
        mlflow.sklearn.log_model(best_model, "xgb_final_model")


if __name__ == "__main__":
    main()
