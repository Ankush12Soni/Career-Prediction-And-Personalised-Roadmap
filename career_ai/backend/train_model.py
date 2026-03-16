"""
Neural Network Career Domain Predictor — Training Script
PyTorch model with BatchNorm + Dropout for higher accuracy.
Saves a sklearn-compatible wrapper so predictor.py needs no changes.
Run: python train_model.py   (from backend/ folder)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# ── Import shared PyTorch model (makes unpickling work everywhere) ────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from models.torch_model import TorchCareerClassifier  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(BASE_DIR, "data", "career_ai_training_dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Feature columns (23 quiz features) ───────────────────────────────────────
FEATURE_COLS = [
    "python_interest", "java_interest", "javascript_interest",
    "cpp_interest", "mobile_dev_interest",
    "linear_algebra", "statistics_probability", "discrete_math", "calculus",
    "system_design_interest", "networking_interest", "security_interest", "cloud_interest",
    "design_interest", "product_thinking",
    "communication_skill", "leadership",
    "research_interest", "analytical_skill", "curiosity_learning",
    "dsa_skill", "projects_built", "build_and_deploy"
]
TARGET_COL = "career_domain"


# ── Main training function ────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  Career Domain Neural Network  (PyTorch + Dropout + BatchNorm)")
    print("=" * 60)

    # 1. Load data ─────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading dataset …  {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"      Rows: {len(df)}  |  Domains: {sorted(df[TARGET_COL].unique())}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values

    # 2. Encode labels ─────────────────────────────────────────────────────────
    print("\n[2/5] Encoding labels …")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"      Classes ({len(le.classes_)}): {list(le.classes_)}")

    # 3. Scale features ────────────────────────────────────────────────────────
    print("\n[3/5] Scaling features …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # 4. Split: 72% train | 8% val | 20% test  ─────────────────────────────────
    X_tmp,  X_test,  y_tmp,  y_test  = train_test_split(
        X_scaled, y_enc, test_size=0.20, random_state=42, stratify=y_enc)
    X_train, X_val,  y_train, y_val  = train_test_split(
        X_tmp,   y_tmp,  test_size=0.10, random_state=42, stratify=y_tmp)
    print(f"      Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

    # 5. Train ─────────────────────────────────────────────────────────────────
    print("\n[4/5] Training PyTorch network (BatchNorm + Dropout) …")
    clf = TorchCareerClassifier()
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val,
            epochs=300, lr=1e-3, batch_size=64, patience=25)
    clf.classes_ = le.classes_   # expose for predictor.py

    # 6. Evaluate ──────────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on held-out test set …")
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"      Test Accuracy : {acc * 100:.2f}%")
    print(f"      Epochs ran    : {clf.n_iter_}")
    print(f"      Best val acc  : {max(clf.validation_scores_)*100:.2f}%  "
          f"(epoch {clf.validation_scores_.index(max(clf.validation_scores_))+1})")
    print("\n" + classification_report(y_test, y_pred, target_names=le.classes_))

    # 7. Save artefacts ────────────────────────────────────────────────────────
    model_path  = os.path.join(MODEL_DIR, "career_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    le_path     = os.path.join(MODEL_DIR, "label_encoder.pkl")

    joblib.dump(clf,    model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le,     le_path)

    print("\n[OK] Artefacts saved:")
    print(f"     {model_path}")
    print(f"     {scaler_path}")
    print(f"     {le_path}")
    print("=" * 60)

    return acc


if __name__ == "__main__":
    train()
