"""
Career Recommendation Engine
Given a predicted domain and student quiz scores, returns ranked careers.
"""

import os
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "career_knowledge_dataset.csv")

# Mapping quiz feature → implied skill keywords (matched against career skills column)
SKILL_MAP = {
    "python_interest":          ["Python", "NumPy", "Pandas", "Scikit-learn", "PyTorch", "TensorFlow",
                                  "MLflow", "Airflow", "dbt", "FastAPI"],
    "java_interest":            ["Java", "Kotlin", "Spring Boot", "Android SDK", "Jetpack Compose"],
    "javascript_interest":      ["JavaScript", "TypeScript", "Node.js", "React", "Express",
                                  "React Native", "Web3.js", "Expo"],
    "cpp_interest":             ["C++", "C", "Unreal Engine", "Game Physics", "Shaders", "Assembly",
                                  "Embedded C", "ROS", "Microcontrollers"],
    "mobile_dev_interest":      ["Android SDK", "iOS SDK", "Swift", "SwiftUI", "UIKit", "Kotlin",
                                  "Flutter", "Dart", "React Native", "Firebase", "Core Data",
                                  "Room Database", "Jetpack Compose"],
    "linear_algebra":           ["Linear Algebra", "Mathematics", "3D Math"],
    "statistics_probability":   ["Statistics", "Probability", "Financial Mathematics", "Time Series",
                                  "Risk Modeling", "A/B Testing", "Experiment Design", "Econometrics"],
    "discrete_math":            ["Discrete Math", "Data Structures", "Algorithms", "Cryptography",
                                  "Graph Theory"],
    "calculus":                 ["Calculus", "Mathematics", "Aerodynamics", "Thermodynamics",
                                  "Control Systems", "Signal Processing", "Kinematics"],
    "system_design_interest":   ["SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "System Design",
                                  "Databases", "Query Optimisation", "Indexing", "Data Modelling",
                                  "Normalization", "Stored Procedures", "Triggers", "ORM",
                                  "ETL Pipelines", "Data Pipelines", "Backup & Recovery",
                                  "Replication", "Performance Tuning", "NoSQL", "Snowflake",
                                  "Apache Spark", "Kafka", "REST APIs", "API Design",
                                  "GraphQL", "Message Queues", "Solidity", "Smart Contracts"],
    "networking_interest":      ["Networking", "TCP/IP", "HTTP", "DNS", "REST APIs",
                                  "Authentication", "Rate Limiting", "OpenAPI"],
    "security_interest":        ["Security", "Kali Linux", "Penetration Testing", "Encryption",
                                  "Metasploit", "Burp Suite", "OWASP", "Vulnerability Assessment",
                                  "SIEM Tools", "Threat Detection", "Incident Response",
                                  "Security Architecture", "IAM", "Zero Trust", "SAST/DAST",
                                  "Firewall Configuration", "Reverse Engineering", "YARA Rules",
                                  "CompTIA Security+", "CEH", "CSPM", "Compliance",
                                  "Windows Internals", "Sandboxing", "Cloud Security",
                                  "Kubernetes Security"],
    "cloud_interest":           ["AWS", "Azure", "GCP", "Cloud", "Docker", "Kubernetes",
                                  "Terraform", "CI/CD", "Helm", "Serverless", "IAM",
                                  "Monitoring", "Prometheus", "Grafana", "Linux", "Bash Scripting",
                                  "Automation", "DVC", "MLflow", "Kubeflow", "MLOps",
                                  "Observability", "SLO", "Incident Management"],
    "design_interest":          ["Figma", "UI/UX", "Adobe Photoshop", "Adobe Illustrator",
                                  "Design Principles", "Typography", "Colour Theory", "Wireframing",
                                  "Component Libraries", "Prototyping", "User Research",
                                  "Usability Testing", "Information Architecture", "Design Systems",
                                  "Accessibility", "After Effects", "Cinema 4D", "Lottie",
                                  "Motion Graphics", "Animation Principles", "Storyboarding",
                                  "Brand Identity", "Layout Design"],
    "product_thinking":         ["Product Strategy", "Product Thinking", "User Stories", "Agile",
                                  "Jira", "Roadmapping", "Stakeholder Management",
                                  "Business Strategy", "Requirements Gathering", "Process Mapping",
                                  "BPMN", "Operations", "Supply Chain", "ERP Systems", "Lean",
                                  "Marketing Analytics", "SEO", "Google Analytics", "Funnel Analysis",
                                  "Email Marketing", "Paid Ads", "Branding", "Copywriting"],
    "communication_skill":      ["Communication", "Stakeholder Communication", "Stakeholder Engagement",
                                  "Report Writing", "Technical Writing", "Teaching",
                                  "Curriculum Design", "Essay Writing", "Copywriting"],
    "leadership":               ["Leadership", "Teamwork", "Management", "Grant Writing",
                                  "Physical Fitness", "Strategy"],
    "research_interest":        ["Research Methods", "Research", "LaTeX", "Academic Writing",
                                  "Paper Writing", "Publications", "PhD", "Experiment Design",
                                  "Survey Methods", "Causal Inference", "Policy Analysis",
                                  "Benchmarking"],
    "analytical_skill":         ["Data Analysis", "Analytical", "PowerBI", "Tableau", "Dashboard Design",
                                  "DAX", "Star Schema", "Data Visualization", "Excel",
                                  "General Studies", "Economics", "Public Administration",
                                  "Policy Analysis", "CAD", "FEA", "MATLAB", "Structural Analysis",
                                  "Aerodynamics", "CFD", "Circuit Design", "Power Systems",
                                  "PCB Design", "Manufacturing Processes"],
    "curiosity_learning":       ["Curiosity", "Self-learning"],
    "dsa_skill":                ["Data Structures", "Algorithms", "LeetCode", "Problem Solving",
                                  "Competitive Programming", "OOP", "System Design"],
    "projects_built":           ["GitHub", "Portfolio", "Open Source", "Build Projects",
                                  "Side Projects", "Hackathon"],
    "build_and_deploy":         ["Docker", "Deployment", "CI/CD", "AWS", "Vercel", "Netlify",
                                  "App Store Deployment", "Play Store", "Firebase", "FastAPI",
                                  "REST APIs", "Model Deployment", "MLflow", "Build Projects",
                                  "Publish App", "GitHub Pages", "Heroku", "Railway"],
}

SKILL_THRESHOLD = 7  # score >= this means student "has" the skill (raised to reduce noise)

_df = None

def _load():
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)

# How many skills to surface per quiz feature (keeps the badge list readable)
_MAX_SKILLS_PER_FEATURE = 5

def get_student_skills(quiz_scores: dict) -> list[str]:
    """Convert quiz scores to a list of skill keywords the student possesses.
    Only features scored >= SKILL_THRESHOLD contribute, capped at _MAX_SKILLS_PER_FEATURE
    skills each so the badge list stays readable.
    """
    skills = set()
    for feature, score in quiz_scores.items():
        if score >= SKILL_THRESHOLD:
            # Weight: higher score → show more skills from that feature
            cap = _MAX_SKILLS_PER_FEATURE if score <= 8 else len(SKILL_MAP.get(feature, []))
            for skill in SKILL_MAP.get(feature, [])[:cap]:
                skills.add(skill)
    return sorted(skills)

def recommend_careers(domain: str, quiz_scores: dict, top_n: int = 5) -> list[dict]:
    """
    Return top-N career recommendations for a domain,
    ranked by skill match with the student's profile.
    """
    _load()

    domain_df = _df[_df["domain"] == domain].copy()
    if domain_df.empty:
        return []

    student_skills = set(s.lower() for s in get_student_skills(quiz_scores))

    results = []
    for _, row in domain_df.iterrows():
        required = [s.strip() for s in str(row["skills"]).split(",")]
        req_lower = set(s.lower() for s in required)

        matched = req_lower & student_skills
        missing = req_lower - student_skills

        match_pct = round(len(matched) / len(req_lower) * 100, 1) if req_lower else 0

        results.append({
            "career":        row["career"],
            "domain":        row["domain"],
            "required_skills": required,
            "matched_skills": [s for s in required if s.lower() in student_skills],
            "missing_skills": [s for s in required if s.lower() not in student_skills],
            "match_score":   match_pct,
            "roadmap":       row["roadmap"],
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results[:top_n]

def get_all_domains() -> list[str]:
    _load()
    return sorted(_df["domain"].unique().tolist())
