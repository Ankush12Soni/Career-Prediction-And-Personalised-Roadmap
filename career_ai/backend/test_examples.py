"""
Career AI - Interactive Example Test
Tests 6 detailed student personas end-to-end through all system components:
Prediction → Recommendations → Skill Gap → Roadmap
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.predictor   import predict_domains
from services.recommender import recommend_careers, get_student_skills
from services.skill_gap   import analyze_gap
from services.roadmap     import generate_roadmap

# ── 6 Detailed Student Personas ───────────────────────────────────────────────
STUDENTS = [
    {
        "name":    "Aryan Sharma",
        "bio":     "3rd year CSE student, loves Python and math, wants to work in AI",
        "scores": {
            "python_interest": 9, "java_interest": 3, "javascript_interest": 2,
            "cpp_interest": 3, "mobile_dev_interest": 1,
            "linear_algebra": 8, "statistics_probability": 9, "discrete_math": 5, "calculus": 8,
            "system_design_interest": 3, "networking_interest": 2, "security_interest": 1, "cloud_interest": 2,
            "design_interest": 1, "product_thinking": 2,
            "communication_skill": 6, "leadership": 4,
            "research_interest": 8, "analytical_skill": 9, "curiosity_learning": 9
        }
    },
    {
        "name":    "Priya Nair",
        "bio":     "2nd year IT student, loves making apps and UI, active on Instagram",
        "scores": {
            "python_interest": 3, "java_interest": 2, "javascript_interest": 9,
            "cpp_interest": 1, "mobile_dev_interest": 7,
            "linear_algebra": 2, "statistics_probability": 3, "discrete_math": 2, "calculus": 1,
            "system_design_interest": 4, "networking_interest": 2, "security_interest": 1, "cloud_interest": 3,
            "design_interest": 9, "product_thinking": 7,
            "communication_skill": 8, "leadership": 6,
            "research_interest": 2, "analytical_skill": 5, "curiosity_learning": 8
        }
    },
    {
        "name":    "Rahul Verma",
        "bio":     "Final year student, interested in hacking, networks and Linux",
        "scores": {
            "python_interest": 6, "java_interest": 4, "javascript_interest": 3,
            "cpp_interest": 6, "mobile_dev_interest": 2,
            "linear_algebra": 3, "statistics_probability": 4, "discrete_math": 6, "calculus": 3,
            "system_design_interest": 6, "networking_interest": 9, "security_interest": 9, "cloud_interest": 5,
            "design_interest": 2, "product_thinking": 2,
            "communication_skill": 5, "leadership": 6,
            "research_interest": 5, "analytical_skill": 8, "curiosity_learning": 8
        }
    },
    {
        "name":    "Sneha Kulkarni",
        "bio":     "MBA student, strong communicator, interested in product strategy",
        "scores": {
            "python_interest": 3, "java_interest": 1, "javascript_interest": 2,
            "cpp_interest": 1, "mobile_dev_interest": 2,
            "linear_algebra": 2, "statistics_probability": 6, "discrete_math": 2, "calculus": 2,
            "system_design_interest": 5, "networking_interest": 2, "security_interest": 2, "cloud_interest": 3,
            "design_interest": 7, "product_thinking": 9,
            "communication_skill": 9, "leadership": 9,
            "research_interest": 5, "analytical_skill": 7, "curiosity_learning": 7
        }
    },
    {
        "name":    "Dev Patel",
        "bio":     "Engineering student, enjoys AWS, Docker, and setting up servers",
        "scores": {
            "python_interest": 7, "java_interest": 5, "javascript_interest": 4,
            "cpp_interest": 4, "mobile_dev_interest": 2,
            "linear_algebra": 3, "statistics_probability": 4, "discrete_math": 5, "calculus": 3,
            "system_design_interest": 8, "networking_interest": 8, "security_interest": 5, "cloud_interest": 9,
            "design_interest": 2, "product_thinking": 4,
            "communication_skill": 6, "leadership": 7,
            "research_interest": 3, "analytical_skill": 7, "curiosity_learning": 8
        }
    },
    {
        "name":    "Ananya Roy",
        "bio":     "Science student, loves statistics, research papers, and data",
        "scores": {
            "python_interest": 7, "java_interest": 2, "javascript_interest": 2,
            "cpp_interest": 3, "mobile_dev_interest": 1,
            "linear_algebra": 8, "statistics_probability": 9, "discrete_math": 7, "calculus": 8,
            "system_design_interest": 3, "networking_interest": 2, "security_interest": 1, "cloud_interest": 2,
            "design_interest": 2, "product_thinking": 3,
            "communication_skill": 6, "leadership": 5,
            "research_interest": 9, "analytical_skill": 9, "curiosity_learning": 9
        }
    },
]

SEP  = "=" * 72
SEP2 = "-" * 72
SEP3 = "." * 50

def run_student(student):
    name   = student["name"]
    bio    = student["bio"]
    scores = student["scores"]

    print(f"\n{SEP}")
    print(f"  STUDENT : {name}")
    print(f"  BIO     : {bio}")
    print(SEP)

    # ── Step 1: Domain Prediction ─────────────────────────────────────────────
    print("\n  [STEP 1] DOMAIN PREDICTION")
    print(f"  {SEP3}")
    domains = predict_domains(scores)
    for rank, d in enumerate(domains[:5], 1):
        bar   = "#" * int(d["confidence"] / 5)
        medal = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
        print(f"  {medal:<4}  {d['domain']:<28}  {d['confidence']:>5}%  {bar}")

    top_domain = domains[0]["domain"]

    # ── Step 2: Student Skills ────────────────────────────────────────────────
    print(f"\n  [STEP 2] DERIVED STUDENT SKILLS")
    print(f"  {SEP3}")
    student_skills = get_student_skills(scores)
    for i, s in enumerate(student_skills):
        end = "\n" if (i + 1) % 4 == 0 else "  "
        print(f"  + {s:<24}", end=end)
    print()

    # ── Step 3: Career Recommendations ───────────────────────────────────────
    print(f"\n  [STEP 3] TOP CAREER RECOMMENDATIONS  ({top_domain})")
    print(f"  {SEP3}")
    careers = recommend_careers(top_domain, scores, top_n=3)
    for i, c in enumerate(careers, 1):
        print(f"  {i}. {c['career']}")
        print(f"     Match Score  : {c['match_score']}%")
        print(f"     Has Skills   : {', '.join(c['matched_skills']) if c['matched_skills'] else 'None yet'}")
        print(f"     Missing      : {', '.join(c['missing_skills']) if c['missing_skills'] else 'None!'}")

    # ── Step 4: Skill Gap for top career ─────────────────────────────────────
    top_career = careers[0]["career"] if careers else None
    if top_career:
        print(f"\n  [STEP 4] SKILL GAP ANALYSIS  ->  {top_career}")
        print(f"  {SEP3}")
        gap = analyze_gap(top_career, student_skills)
        readiness = gap["readiness_score"]
        bar_len   = int(readiness / 5)
        bar_fill  = "#" * bar_len
        bar_empty = "-" * (20 - bar_len)
        print(f"  Readiness  : [{bar_fill}{bar_empty}]  {readiness}%")
        print(f"  Have       : {', '.join(gap['matched_skills']) if gap['matched_skills'] else 'None yet'}")
        print(f"  Need       : {', '.join(gap['missing_skills']) if gap['missing_skills'] else 'All covered!'}")

        # ── Step 5: Personalized Roadmap ─────────────────────────────────────
        print(f"\n  [STEP 5] PERSONALIZED LEARNING ROADMAP")
        print(f"  {SEP3}")
        roadmap = generate_roadmap(top_career, gap["missing_skills"])
        print(f"  Career     : {roadmap['career']}")
        print(f"  Progress   : {roadmap['completed_steps']}/{roadmap['total_steps']} steps already covered\n")

        ICONS = {"done": "[DONE]", "high": "[TODO]", "milestone": "[GOAL]"}
        for step in roadmap["roadmap_steps"]:
            icon = ICONS.get(step["priority"], "[    ]")
            res  = f"  -> {step['resources'][0]}" if step.get("resources") else ""
            print(f"  {icon}  {step['step']}")
            if res:
                print(f"         {res}")

    print(f"\n{SEP2}\n")


# ── Run all students ──────────────────────────────────────────────────────────
print(SEP)
print("   CAREER AI SYSTEM  --  END-TO-END EXAMPLE TEST")
print("   6 Student Profiles | 5 Pipeline Stages Each")
print(SEP)

for student in STUDENTS:
    run_student(student)

print(SEP)
print("  ALL EXAMPLES COMPLETE")
print(SEP)
