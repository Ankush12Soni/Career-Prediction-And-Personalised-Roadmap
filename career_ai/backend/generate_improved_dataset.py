"""
Improved Career AI Training Dataset Generator
- Balanced classes (1500 samples each, 15000 total)
- Tighter, non-overlapping domain assignment rules
- Gaussian noise added to make boundaries realistic
- Stratified so every domain is equally represented
"""

import pandas as pd
import numpy as np

np.random.seed(42)

SAMPLES_PER_DOMAIN = 1500

def make_profile(base: dict, noise: float = 2.0) -> list:
    """
    Build a feature vector from a base profile + Gaussian noise.
    Values are clipped to [0, 10].
    """
    FEATURES = [
        "python_interest", "java_interest", "javascript_interest", "cpp_interest",
        "mobile_dev_interest", "linear_algebra", "statistics_probability",
        "discrete_math", "calculus", "system_design_interest", "networking_interest",
        "security_interest", "cloud_interest", "design_interest", "product_thinking",
        "communication_skill", "leadership", "research_interest",
        "analytical_skill", "curiosity_learning",
        "dsa_skill", "projects_built", "build_and_deploy"
    ]
    row = []
    for f in FEATURES:
        val = base.get(f, 3)
        noisy = val + np.random.normal(0, noise)
        row.append(int(np.clip(round(noisy), 0, 10)))
    return row

# ── Domain base profiles ──────────────────────────────────────────────────────
# Each profile defines the "ideal" student for that domain.
# Features not listed default to 3 (low-medium).
DOMAIN_PROFILES = {

    "Artificial Intelligence": {
        "python_interest": 9, "java_interest": 2, "javascript_interest": 1,
        "cpp_interest": 5, "mobile_dev_interest": 1,
        "linear_algebra": 10, "statistics_probability": 7, "discrete_math": 5, "calculus": 10,
        "system_design_interest": 5, "networking_interest": 1, "security_interest": 1, "cloud_interest": 4,
        "design_interest": 1, "product_thinking": 2,
        "communication_skill": 4, "leadership": 2,
        "research_interest": 6, "analytical_skill": 10, "curiosity_learning": 8,
        "dsa_skill": 7, "projects_built": 6, "build_and_deploy": 5
    },

    "Data Science": {
        "python_interest": 9, "java_interest": 2, "javascript_interest": 3,
        "cpp_interest": 1, "mobile_dev_interest": 1,
        "linear_algebra": 5, "statistics_probability": 10, "discrete_math": 4, "calculus": 4,
        "system_design_interest": 5, "networking_interest": 2, "security_interest": 1, "cloud_interest": 6,
        "design_interest": 3, "product_thinking": 8,
        "communication_skill": 8, "leadership": 5,
        "research_interest": 5, "analytical_skill": 9, "curiosity_learning": 7,
        "dsa_skill": 5, "projects_built": 6, "build_and_deploy": 5
    },

    "Web Development": {
        "python_interest": 4, "java_interest": 3, "javascript_interest": 9,
        "cpp_interest": 1, "mobile_dev_interest": 4,
        "linear_algebra": 2, "statistics_probability": 2, "discrete_math": 2, "calculus": 1,
        "system_design_interest": 5, "networking_interest": 3, "security_interest": 2, "cloud_interest": 4,
        "design_interest": 7, "product_thinking": 6,
        "communication_skill": 7, "leadership": 5,
        "research_interest": 2, "analytical_skill": 5, "curiosity_learning": 7,
        "dsa_skill": 4, "projects_built": 8, "build_and_deploy": 9
    },

    "Mobile Development": {
        "python_interest": 4, "java_interest": 8, "javascript_interest": 5,
        "cpp_interest": 3, "mobile_dev_interest": 9,
        "linear_algebra": 2, "statistics_probability": 2, "discrete_math": 3, "calculus": 2,
        "system_design_interest": 5, "networking_interest": 3, "security_interest": 2, "cloud_interest": 3,
        "design_interest": 6, "product_thinking": 6,
        "communication_skill": 6, "leadership": 5,
        "research_interest": 2, "analytical_skill": 5, "curiosity_learning": 7,
        "dsa_skill": 5, "projects_built": 8, "build_and_deploy": 9
    },

    "Software Engineering": {
        "python_interest": 6, "java_interest": 8, "javascript_interest": 5,
        "cpp_interest": 7, "mobile_dev_interest": 4,
        "linear_algebra": 5, "statistics_probability": 4, "discrete_math": 7, "calculus": 5,
        "system_design_interest": 8, "networking_interest": 5, "security_interest": 3, "cloud_interest": 5,
        "design_interest": 3, "product_thinking": 4,
        "communication_skill": 6, "leadership": 6,
        "research_interest": 3, "analytical_skill": 7, "curiosity_learning": 7,
        "dsa_skill": 9, "projects_built": 7, "build_and_deploy": 7
    },

    "Cloud & DevOps": {
        "python_interest": 6, "java_interest": 4, "javascript_interest": 3,
        "cpp_interest": 3, "mobile_dev_interest": 1,
        "linear_algebra": 3, "statistics_probability": 3, "discrete_math": 4, "calculus": 2,
        "system_design_interest": 8, "networking_interest": 8, "security_interest": 5, "cloud_interest": 9,
        "design_interest": 2, "product_thinking": 4,
        "communication_skill": 6, "leadership": 7,
        "research_interest": 3, "analytical_skill": 6, "curiosity_learning": 7,
        "dsa_skill": 4, "projects_built": 6, "build_and_deploy": 9
    },

    "Cybersecurity": {
        "python_interest": 5, "java_interest": 4, "javascript_interest": 3,
        "cpp_interest": 5, "mobile_dev_interest": 2,
        "linear_algebra": 3, "statistics_probability": 4, "discrete_math": 6, "calculus": 2,
        "system_design_interest": 6, "networking_interest": 9, "security_interest": 9, "cloud_interest": 5,
        "design_interest": 2, "product_thinking": 3,
        "communication_skill": 5, "leadership": 6,
        "research_interest": 5, "analytical_skill": 8, "curiosity_learning": 8,
        "dsa_skill": 6, "projects_built": 5, "build_and_deploy": 5
    },

    "Design": {
        "python_interest": 2, "java_interest": 1, "javascript_interest": 4,
        "cpp_interest": 1, "mobile_dev_interest": 3,
        "linear_algebra": 1, "statistics_probability": 2, "discrete_math": 1, "calculus": 1,
        "system_design_interest": 3, "networking_interest": 1, "security_interest": 1, "cloud_interest": 2,
        "design_interest": 9, "product_thinking": 7,
        "communication_skill": 8, "leadership": 6,
        "research_interest": 3, "analytical_skill": 5, "curiosity_learning": 8,
        "dsa_skill": 1, "projects_built": 7, "build_and_deploy": 5
    },

    "Product & Business": {
        "python_interest": 3, "java_interest": 2, "javascript_interest": 3,
        "cpp_interest": 1, "mobile_dev_interest": 2,
        "linear_algebra": 2, "statistics_probability": 5, "discrete_math": 2, "calculus": 2,
        "system_design_interest": 5, "networking_interest": 3, "security_interest": 2, "cloud_interest": 3,
        "design_interest": 6, "product_thinking": 9,
        "communication_skill": 9, "leadership": 9,
        "research_interest": 4, "analytical_skill": 7, "curiosity_learning": 7,
        "dsa_skill": 2, "projects_built": 5, "build_and_deploy": 3
    },

    "Research": {
        "python_interest": 5, "java_interest": 2, "javascript_interest": 1,
        "cpp_interest": 6, "mobile_dev_interest": 1,
        "linear_algebra": 8, "statistics_probability": 7, "discrete_math": 10, "calculus": 8,
        "system_design_interest": 2, "networking_interest": 1, "security_interest": 1, "cloud_interest": 2,
        "design_interest": 1, "product_thinking": 2,
        "communication_skill": 5, "leadership": 4,
        "research_interest": 10, "analytical_skill": 8, "curiosity_learning": 10,
        "dsa_skill": 8, "projects_built": 5, "build_and_deploy": 3
    },
}

# ── Generate balanced dataset ─────────────────────────────────────────────────
FEATURES = [
    "python_interest", "java_interest", "javascript_interest", "cpp_interest",
    "mobile_dev_interest", "linear_algebra", "statistics_probability",
    "discrete_math", "calculus", "system_design_interest", "networking_interest",
    "security_interest", "cloud_interest", "design_interest", "product_thinking",
    "communication_skill", "leadership", "research_interest",
    "analytical_skill", "curiosity_learning",
    "dsa_skill", "projects_built", "build_and_deploy"
]

rows = []
labels = []

for domain, base_profile in DOMAIN_PROFILES.items():
    for _ in range(SAMPLES_PER_DOMAIN):
        # Vary noise level per sample to get more realistic spread
        noise = np.random.uniform(0.8, 1.5)
        row = make_profile(base_profile, noise=noise)
        rows.append(row)
        labels.append(domain)

df = pd.DataFrame(rows, columns=FEATURES)
df["career_domain"] = labels

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
output_path = "data/career_ai_training_dataset.csv"
df.to_csv(output_path, index=False)

print("Improved dataset generated successfully!")
print(f"Total samples : {len(df)}")
print(f"Domains       : {df['career_domain'].nunique()}")
print("\nDomain distribution:")
dist = df["career_domain"].value_counts()
for domain, count in dist.items():
    print(f"  {domain:<30} {count}")
