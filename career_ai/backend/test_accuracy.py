"""
Career AI - Model Accuracy Test
Tests with known profiles + random inputs
"""

import numpy as np
import joblib
import os
from collections import Counter

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "career_model.pkl")
ENC_PATH    = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

FEATURES = [
    "python_interest", "java_interest", "javascript_interest", "cpp_interest",
    "mobile_dev_interest", "linear_algebra", "statistics_probability",
    "discrete_math", "calculus", "system_design_interest", "networking_interest",
    "security_interest", "cloud_interest", "design_interest", "product_thinking",
    "communication_skill", "leadership", "research_interest",
    "analytical_skill", "curiosity_learning"
]

model   = joblib.load(MODEL_PATH)
encoder = joblib.load(ENC_PATH)
scaler  = joblib.load(SCALER_PATH)

# ── Known profiles with expected domain ──────────────────────────────────────
profiles = [
    {"name": "Alex  - AI Enthusiast",      "expected": "Artificial Intelligence", "scores": [9,3,2,2,1, 8,9,6,8, 2,2,1,2, 1,2, 6,4,8,9,9]},
    {"name": "Bob   - Web Developer",      "expected": "Web Development",         "scores": [5,3,9,2,6, 2,3,2,1, 5,3,1,3, 7,6, 7,5,2,4,5]},
    {"name": "Carol - Data Scientist",     "expected": "Data Science",            "scores": [8,2,3,1,1, 7,9,5,7, 4,2,1,2, 2,4, 6,4,7,9,8]},
    {"name": "Dave  - Cybersecurity",      "expected": "Cybersecurity",           "scores": [5,4,3,5,2, 4,4,6,3, 6,9,9,5, 2,2, 5,6,5,7,6]},
    {"name": "Eve   - Mobile Dev",         "expected": "Mobile Development",      "scores": [6,7,6,3,9, 3,3,3,2, 4,3,1,2, 5,5, 7,5,2,4,5]},
    {"name": "Frank - Cloud/DevOps",       "expected": "Cloud & DevOps",          "scores": [6,4,3,4,2, 4,4,5,3, 8,8,5,9, 2,4, 6,7,3,5,5]},
    {"name": "Grace - Designer",           "expected": "Design",                  "scores": [2,1,5,1,3, 1,2,1,1, 2,1,1,1, 9,8, 8,6,3,4,7]},
    {"name": "Hank  - Research Scientist", "expected": "Research",                "scores": [6,4,2,4,1, 9,8,8,9, 3,3,2,2, 2,3, 6,5,9,9,9]},
    {"name": "Iris  - Product Manager",    "expected": "Product & Business",      "scores": [3,2,3,1,2, 3,5,2,2, 5,3,2,3, 6,9, 9,9,4,7,6]},
    {"name": "Jack  - Software Engineer",  "expected": "Software Engineering",    "scores": [7,8,6,8,4, 6,5,7,6, 8,5,3,4, 3,4, 7,6,4,6,6]},
    {"name": "Liam  - AI Researcher",      "expected": "Artificial Intelligence", "scores": [9,2,1,1,1,10,10,7,9, 1,1,1,1, 1,1, 5,3,9,10,9]},
    {"name": "Mia   - Frontend Dev",       "expected": "Web Development",         "scores": [3,2,10,1,4, 1,2,2,1, 4,2,1,2, 8,6, 8,4,1,3,5]},
    {"name": "Noah  - Data Engineer",      "expected": "Data Science",            "scores": [8,5,4,3,2, 6,8,5,5, 6,3,2,6, 2,3, 5,4,5,7,7]},
    {"name": "Olivia- Ethical Hacker",     "expected": "Cybersecurity",           "scores": [6,5,4,6,2, 5,5,7,4, 5,9,9,4, 2,2, 5,5,5,7,6]},
    {"name": "Pete  - Cloud Architect",    "expected": "Cloud & DevOps",          "scores": [5,5,3,3,2, 4,4,4,3, 8,7,5,9, 2,5, 6,7,3,5,6]},
]

# ── 20 fully random profiles ──────────────────────────────────────────────────
np.random.seed(99)
random_profiles = [
    {"name": f"Random #{i+1:02d}", "expected": None,
     "scores": np.random.randint(0, 11, 20).tolist()}
    for i in range(20)
]

def predict(scores):
    x = np.array(scores).reshape(1, -1)
    x_s = scaler.transform(x)
    proba = model.predict_proba(x_s)[0]
    top3  = np.argsort(proba)[::-1][:3]
    return (
        encoder.inverse_transform([top3[0]])[0],
        round(float(proba[top3[0]]) * 100, 1),
        [(encoder.inverse_transform([i])[0], round(float(proba[i])*100,1)) for i in top3]
    )

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("  CAREER AI  --  MODEL ACCURACY CHECK")
print("=" * 90)

# ── SECTION 1: Known profiles ─────────────────────────────────────────────────
print(f"\n{'STUDENT':<32} {'EXPECTED':<26} {'PREDICTED':<26} {'CONF':>6}  RESULT")
print("-" * 90)

correct = 0
conf_known = []
for p in profiles:
    pred, conf, top3 = predict(p["scores"])
    ok = pred == p["expected"]
    if ok:
        correct += 1
    conf_known.append(conf)
    flag = "PASS" if ok else "FAIL"
    print(f"  {p['name']:<30} {p['expected']:<26} {pred:<26} {conf:>5}%  [{flag}]")

acc = round(correct / len(profiles) * 100, 1)
print("-" * 90)
print(f"  Known Profile Accuracy: {correct}/{len(profiles)} = {acc}%   |   Avg Confidence: {round(sum(conf_known)/len(conf_known),1)}%")

# ── SECTION 2: Random profiles ────────────────────────────────────────────────
print(f"\n{'STUDENT':<20} {'PREDICTED DOMAIN':<28} {'TOP-2':<28} {'TOP-3':<28} {'CONF':>6}")
print("-" * 90)

conf_rand = []
rand_preds = []
for p in random_profiles:
    pred, conf, top3 = predict(p["scores"])
    rand_preds.append(pred)
    conf_rand.append(conf)
    t2 = f"{top3[1][0]} ({top3[1][1]}%)" if len(top3) > 1 else ""
    t3 = f"{top3[2][0]} ({top3[2][1]}%)" if len(top3) > 2 else ""
    print(f"  {p['name']:<18} {pred:<28} {t2:<28} {t3:<28} {conf:>5}%")

print("-" * 90)
print(f"  Random Avg Confidence: {round(sum(conf_rand)/len(conf_rand),1)}%   Min: {min(conf_rand)}%   Max: {max(conf_rand)}%")

# ── SECTION 3: Domain distribution of random students ─────────────────────────
print(f"\n  DOMAIN DISTRIBUTION (Random Students):")
print("  " + "-" * 50)
dist = Counter(rand_preds)
for domain, count in dist.most_common():
    bar = "#" * (count * 3)
    print(f"  {domain:<30}  {bar} ({count})")

# ── SECTION 4: Top-3 breakdown for known profiles ────────────────────────────
print(f"\n  TOP-3 DOMAIN BREAKDOWN (Known Profiles):")
print("  " + "-" * 60)
for p in profiles:
    _, _, top3 = predict(p["scores"])
    status = "OK" if top3[0][0] == p["expected"] else "WRONG"
    print(f"\n  {p['name'].strip()} [{status}]  (expected: {p['expected']})")
    for rank, (dom, pct) in enumerate(top3, 1):
        marker = " <-- CORRECT" if dom == p["expected"] else ""
        bar = "=" * int(pct / 5)
        print(f"    {rank}. {dom:<30} {pct:>5}%  {bar}{marker}")

# ── SECTION 5: Overall summary ────────────────────────────────────────────────
all_conf = conf_known + conf_rand
print(f"\n{'='*90}")
print(f"  OVERALL SUMMARY")
print(f"{'='*90}")
print(f"  Known Profile Accuracy  : {correct}/{len(profiles)} = {acc}%")
print(f"  Overall Avg Confidence  : {round(sum(all_conf)/len(all_conf),1)}%")
print(f"  Overall Min Confidence  : {min(all_conf)}%")
print(f"  Overall Max Confidence  : {max(all_conf)}%")
print(f"  Total Profiles Tested   : {len(profiles) + len(random_profiles)}")
print(f"{'='*90}")
