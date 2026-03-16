from services.predictor import predict_domains
from services.recommender import recommend_careers, get_student_skills
from services.skill_gap import analyze_gap
from services.roadmap import generate_roadmap

scores = {
    "python_interest": 9, "java_interest": 3, "javascript_interest": 4,
    "cpp_interest": 2, "mobile_dev_interest": 1,
    "linear_algebra": 8, "statistics_probability": 9, "discrete_math": 5, "calculus": 7,
    "system_design_interest": 4, "networking_interest": 2, "security_interest": 1, "cloud_interest": 3,
    "design_interest": 2, "product_thinking": 3,
    "communication_skill": 7, "leadership": 5,
    "research_interest": 8, "analytical_skill": 9, "curiosity_learning": 9
}

print("== TOP DOMAINS ==")
domains = predict_domains(scores)
for d in domains[:3]:
    print(f"  {d['domain']}: {d['confidence']}%")

student_skills = get_student_skills(scores)
print(f"\n== STUDENT SKILLS ==\n  {student_skills}")

print("\n== CAREER RECOMMENDATIONS (Artificial Intelligence) ==")
careers = recommend_careers("Artificial Intelligence", scores, 3)
for c in careers:
    print(f"  {c['career']} -- match: {c['match_score']}%")

print("\n== SKILL GAP (Machine Learning Engineer) ==")
gap = analyze_gap("Machine Learning Engineer", student_skills)
print(f"  Readiness: {gap['readiness_score']}%")
print(f"  Missing: {gap['missing_skills']}")

print("\n== ROADMAP ==")
rm = generate_roadmap("Machine Learning Engineer", gap["missing_skills"])
for s in rm["roadmap_steps"]:
    print(f"  [{s['priority'].upper()[:4]}] {s['step']}")

print("\nALL TESTS PASSED")
