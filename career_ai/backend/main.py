"""
FastAPI — Career AI Backend
Entry point: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.schemas      import QuizScores, RecommendRequest, SkillGapRequest, RoadmapRequest
from routes.quiz         import QUIZ_QUESTIONS
from services.predictor  import predict_domains
from services.recommender import recommend_careers, get_student_skills, get_all_domains
from services.skill_gap  import analyze_gap, bulk_analyze
from services.roadmap    import generate_roadmap

app = FastAPI(
    title="Career AI API",
    description="AI-Powered Career Recommendation System for The Student Hub",
    version="1.0.0",
)

# Allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Career AI API is running 🚀"}

# ── Quiz ──────────────────────────────────────────────────────────────────────
@app.get("/api/quiz", tags=["Quiz"])
def get_quiz():
    """Return all 20 quiz questions grouped by category."""
    categories = {}
    for q in QUIZ_QUESTIONS:
        cat = q["category"]
        categories.setdefault(cat, []).append(q)
    return {"total_questions": len(QUIZ_QUESTIONS), "categories": categories}

# ── Predict domains ───────────────────────────────────────────────────────────
@app.post("/api/predict", tags=["Prediction"])
def predict(scores: QuizScores):
    """
    Submit quiz scores and receive top career domain predictions
    with confidence percentages.
    """
    scores_dict = scores.model_dump()
    domains = predict_domains(scores_dict)
    student_skills = get_student_skills(scores_dict)

    return {
        "top_domains":     domains[:3],
        "all_domains":     domains,
        "student_skills":  student_skills,
    }

# ── Recommend careers ─────────────────────────────────────────────────────────
@app.post("/api/recommend", tags=["Recommendation"])
def recommend(req: RecommendRequest):
    """
    Given a domain and quiz scores, return ranked career recommendations.
    """
    scores_dict = req.quiz_scores.model_dump()
    careers = recommend_careers(req.domain, scores_dict, req.top_n)
    if not careers:
        raise HTTPException(status_code=404, detail=f"No careers found for domain '{req.domain}'")
    return {"domain": req.domain, "careers": careers}

# ── Full result (predict + recommend top 3 domains) ───────────────────────────
@app.post("/api/full-result", tags=["Prediction"])
def full_result(scores: QuizScores):
    """
    One-shot endpoint: returns domain predictions AND top 3 career
    recommendations for each of the top 3 predicted domains.
    """
    scores_dict    = scores.model_dump()
    domains        = predict_domains(scores_dict)
    student_skills = get_student_skills(scores_dict)
    top3_domains   = domains[:3]

    result_domains = []
    for d in top3_domains:
        careers = recommend_careers(d["domain"], scores_dict, top_n=3)
        result_domains.append({
            "domain":     d["domain"],
            "confidence": d["confidence"],
            "careers":    careers,
        })

    return {
        "top_domains":    result_domains,
        "student_skills": student_skills,
    }

# ── Skill gap ─────────────────────────────────────────────────────────────────
@app.post("/api/skill-gap", tags=["Skill Gap"])
def skill_gap(req: SkillGapRequest):
    """
    Analyze the gap between student's current skills and a target career.
    """
    result = analyze_gap(req.career, req.student_skills)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.post("/api/skill-gap/bulk", tags=["Skill Gap"])
def skill_gap_bulk(careers: list[str], student_skills: list[str]):
    """Analyze skill gap for multiple careers at once."""
    return bulk_analyze(careers, student_skills)

# ── Roadmap ───────────────────────────────────────────────────────────────────
@app.post("/api/roadmap", tags=["Roadmap"])
def roadmap(req: RoadmapRequest):
    """
    Generate a personalized step-by-step learning roadmap.
    """
    result = generate_roadmap(req.career, req.missing_skills)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

# ── All domains list ──────────────────────────────────────────────────────────
@app.get("/api/domains", tags=["Knowledge Base"])
def domains():
    return {"domains": get_all_domains()}
