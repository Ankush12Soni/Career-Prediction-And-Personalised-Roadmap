# 🎓 Career Prediction & Personalised Roadmap

An AI-powered career guidance system that helps students discover suitable career paths, analyse skill gaps, and generate personalised learning roadmaps.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Train the Model](#train-the-model)
  - [Run the Streamlit App](#run-the-streamlit-app)
  - [Run the FastAPI Server](#run-the-fastapi-server)
- [API Endpoints](#api-endpoints)
  - [Example: Predict Domain](#example-predict-domain)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Career Knowledge Base](#career-knowledge-base)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Overview

**Career AI** is a REST API backend built with FastAPI that serves as the intelligent core of "The Student Hub" platform. Students answer a 20-question quiz about their interests and aptitude, and the system:

1. **Predicts** the best-matching career domains using a trained neural network.
2. **Recommends** specific careers within those domains based on skill overlap.
3. **Identifies** skill gaps between a student's current abilities and their target career.
4. **Generates** a step-by-step personalised learning roadmap with curated free and paid resources.

---

## Features

| Feature | Description |
|---|---|
| 🧠 **Domain Prediction** | Multi-class neural network predicts top career domains with confidence scores |
| 💼 **Career Recommendations** | Ranks 58 careers by skill-match percentage within a domain |
| 🔍 **Skill Gap Analysis** | Compares student skills to career requirements with a readiness score |
| 🗺️ **Personalised Roadmap** | Step-by-step learning paths enriched with resources from freeCodeCamp, Coursera, MIT OCW, etc. |
| 📝 **20-Question Assessment** | Covers Programming, Mathematics, Systems, Creative & Product, Soft Skills, and Research & Analytical |
| 📊 **Full Result (One-Shot)** | Single endpoint returns predictions + recommendations for the top 3 domains |

---

## Tech Stack

- **UI**: [Streamlit](https://streamlit.io/) + [Plotly](https://plotly.com/python/) (interactive charts)
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **ML**: [PyTorch](https://pytorch.org/) (CareerNet neural network) + [scikit-learn](https://scikit-learn.org/) (preprocessing)
- **Data**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Database**: [MongoDB](https://www.mongodb.com/) via [Motor](https://motor.readthedocs.io/) (async driver)
- **Validation**: [Pydantic v2](https://docs.pydantic.dev/)
- **Serialization**: [Joblib](https://joblib.readthedocs.io/)

---

## Project Structure

```
Career-Prediction-And-Personalised-Roadmap/
└── career_ai/
    └── backend/
        ├── app.py                          # Streamlit UI application
        ├── main.py                         # FastAPI application & all route definitions
        ├── requirements.txt                # Python dependencies
        ├── train_model.py                  # Model training script
        ├── generate_improved_dataset.py    # Synthetic training data generator
        ├── test_examples.py                # End-to-end persona tests
        ├── test_services.py                # Service-level unit tests
        ├── test_accuracy.py                # Model accuracy evaluation
        │
        ├── .streamlit/
        │   └── config.toml                 # Streamlit theme & server configuration
        │
        ├── models/
        │   ├── schemas.py                  # Pydantic request/response schemas
        │   ├── torch_model.py              # CareerNet architecture + sklearn wrapper
        │   ├── career_model.pkl            # Trained classifier
        │   ├── scaler.pkl                  # StandardScaler
        │   └── label_encoder.pkl           # Domain LabelEncoder
        │
        ├── services/
        │   ├── predictor.py                # Domain prediction service
        │   ├── recommender.py              # Career recommendation engine
        │   ├── skill_gap.py                # Skill gap analyser
        │   └── roadmap.py                  # Personalised roadmap generator
        │
        ├── routes/
        │   └── quiz.py                     # 20 quiz question definitions
        │
        └── data/
            ├── career_ai_training_dataset.csv   # 15,000 balanced training samples
            └── career_knowledge_dataset.csv     # 58 careers across 10 domains
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- (Optional) MongoDB instance if database features are used

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Ankush12Soni/Career-Prediction-And-Personalised-Roadmap.git
cd Career-Prediction-And-Personalised-Roadmap

# 2. Navigate to the backend directory
cd career_ai/backend

# 3. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

> **Note:** PyTorch is now included in `requirements.txt` (`torch>=2.6.0`).
> If you prefer a GPU build or a different platform-specific wheel, follow the [official guide](https://pytorch.org/get-started/locally/).

### Train the Model

Skip this step if the pre-trained `models/career_model.pkl` already exists.

```bash
python train_model.py
```

This will:
- Load 15,000 balanced training samples from `data/career_ai_training_dataset.csv`
- Train a `CareerNet` neural network for up to 300 epochs with early stopping
- Save three artifacts to `models/`: `career_model.pkl`, `scaler.pkl`, `label_encoder.pkl`

To regenerate the synthetic training dataset from scratch:

```bash
python generate_improved_dataset.py
```

### Run the Streamlit App

```bash
cd career_ai/backend
streamlit run app.py
```

The interactive UI will open at `http://localhost:8501` in your browser.

**Deploying to Streamlit Cloud:**

1. Push your repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app**, select your repository.
4. Set the **Main file path** to `career_ai/backend/app.py`.
5. Click **Deploy**.

### Run the FastAPI Server

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at:

| URL | Description |
|---|---|
| `http://localhost:8000` | Health check |
| `http://localhost:8000/docs` | Interactive Swagger UI |
| `http://localhost:8000/redoc` | ReDoc documentation |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/api/quiz` | Retrieve all 20 quiz questions grouped by category |
| `GET` | `/api/domains` | List all 10 available career domains |
| `POST` | `/api/predict` | Submit quiz scores → get ranked domain predictions |
| `POST` | `/api/recommend` | Get career recommendations for a specific domain |
| `POST` | `/api/full-result` | One-shot: predictions + recommendations for top 3 domains |
| `POST` | `/api/skill-gap` | Analyse skill gap for a single career |
| `POST` | `/api/skill-gap/bulk` | Analyse skill gap for multiple careers at once |
| `POST` | `/api/roadmap` | Generate a personalised learning roadmap |

### Example: Predict Domain

**Request**

```http
POST /api/predict
Content-Type: application/json
```

```json
{
  "python_interest": 9,
  "java_interest": 3,
  "javascript_interest": 2,
  "cpp_interest": 3,
  "mobile_dev_interest": 1,
  "linear_algebra": 8,
  "statistics_probability": 9,
  "discrete_math": 5,
  "calculus": 8,
  "system_design_interest": 3,
  "networking_interest": 2,
  "security_interest": 1,
  "cloud_interest": 2,
  "design_interest": 1,
  "product_thinking": 2,
  "communication_skill": 6,
  "leadership": 4,
  "research_interest": 8,
  "analytical_skill": 9,
  "curiosity_learning": 9
}
```

**Response**

```json
{
  "top_domains": [
    { "domain": "Artificial Intelligence", "confidence": 85.3 },
    { "domain": "Data Science", "confidence": 78.2 },
    { "domain": "Backend Development", "confidence": 62.1 }
  ],
  "all_domains": [...],
  "student_skills": ["Python", "NumPy", "Statistics", "Linear Algebra", ...]
}
```

---

## Machine Learning Pipeline

### Neural Network Architecture — `CareerNet`

```
Input (23 features)
  │
  ├─ Linear(23 → 512) → BatchNorm → ReLU → Dropout(0.2)
  ├─ Linear(512 → 256) → BatchNorm → ReLU → Dropout(0.2)
  ├─ Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.1)
  ├─ Linear(128 → 64)  → BatchNorm → ReLU → Dropout(0.05)
  │
Output: Linear(64 → 10 domains)
```

### Input Features (23 total)

| Group | Features |
|---|---|
| **Programming** (5) | `python_interest`, `java_interest`, `javascript_interest`, `cpp_interest`, `mobile_dev_interest` |
| **Mathematics** (4) | `linear_algebra`, `statistics_probability`, `discrete_math`, `calculus` |
| **Systems** (4) | `system_design_interest`, `networking_interest`, `security_interest`, `cloud_interest` |
| **Creative & Product** (2) | `design_interest`, `product_thinking` |
| **Soft Skills** (2) | `communication_skill`, `leadership` |
| **Research & Analytical** (3) | `research_interest`, `analytical_skill`, `curiosity_learning` |
| **Derived** (3) | `dsa_skill`, `projects_built`, `build_and_deploy` |

### Output Classes (10 domains)

Artificial Intelligence · Backend Development · Frontend Development · Full Stack Development · Mobile Development · DevOps & Cloud · Cybersecurity · Data Science · Game Development · UI/UX Design

### Training Details

| Parameter | Value |
|---|---|
| Training samples | 15,000 (balanced, 1,500/domain) |
| Train / Val / Test split | 72% / 8% / 20% |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Max epochs | 300 |
| Early stopping patience | 25 epochs |

---

## Career Knowledge Base

The file `data/career_knowledge_dataset.csv` contains **58 careers** across the 10 domains, each with:

- **Required skills** — comma-separated list
- **Roadmap** — arrow-separated (`→`) learning steps

The recommender uses keyword-based skill matching (threshold: score ≥ 7/10 to "have" a skill) and ranks careers by `matched_skills / required_skills × 100`.

The roadmap generator enriches each step with curated resources from platforms including:

> freeCodeCamp · Khan Academy · MIT OpenCourseWare · fast.ai · Coursera · Udemy ·
> LeetCode · Kaggle · TryHackMe · official documentation · and more

---

## Testing

```bash
# End-to-end tests with 6 student personas
python test_examples.py

# Unit tests for individual services
python test_services.py

# Model accuracy evaluation on the test set
python test_accuracy.py
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

> Built with ❤️ for students navigating their career journeys.
