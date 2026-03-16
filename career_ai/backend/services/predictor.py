"""
Career Domain Predictor Service
Loads trained ML model and predicts career domains from quiz scores.
"""

import os
import sys
import numpy as np
import joblib

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Register TorchCareerClassifier so joblib can unpickle the saved model
sys.path.insert(0, BASE_DIR)
from models.torch_model import TorchCareerClassifier  # noqa: F401, E402
MODEL_DIR  = os.path.join(BASE_DIR, "models")

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

_model  = None
_scaler = None
_le     = None

def _load():
    global _model, _scaler, _le
    if _model is None:
        _model  = joblib.load(os.path.join(MODEL_DIR, "career_model.pkl"))
        _scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        _le     = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

def predict_domains(quiz_scores: dict) -> list[dict]:
    """
    Accept a dict of {feature_name: score (0-10)}.
    Returns a ranked list of domains with confidence percentages.
    """
    _load()

    # Build feature vector in correct column order
    features = np.array([[quiz_scores.get(col, 0) for col in FEATURE_COLS]], dtype=float)
    features_scaled = _scaler.transform(features)

    # Get probability for every class
    probas = _model.predict_proba(features_scaled)[0]

    # Pair with domain names and sort descending
    domain_scores = [
        {"domain": _le.classes_[i], "confidence": round(float(p) * 100, 1)}
        for i, p in enumerate(probas)
    ]
    domain_scores.sort(key=lambda x: x["confidence"], reverse=True)
    return domain_scores
