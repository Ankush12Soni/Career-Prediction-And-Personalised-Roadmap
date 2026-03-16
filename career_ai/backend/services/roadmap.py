"""
Personalized Learning Roadmap Generator
Builds an adaptive step-by-step roadmap based on missing skills.
"""

import os
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "career_knowledge_dataset.csv")

# Detailed sub-resources for common skills
SKILL_RESOURCES = {
    # Core Programming
    "python":               {"title": "Python Programming",         "resources": ["freeCodeCamp Python Course", "Automate the Boring Stuff (free)", "Python.org Docs"]},
    "java":                 {"title": "Java Programming",           "resources": ["Java Brains YouTube", "MOOC.fi Java Course", "Head First Java"]},
    "javascript":           {"title": "JavaScript Fundamentals",    "resources": ["javascript.info (free)", "MDN Web Docs", "freeCodeCamp JS"]},
    "typescript":           {"title": "TypeScript",                 "resources": ["TypeScript Official Docs", "Matt Pocock TS Course", "Execute Program TS"]},
    "c++":                  {"title": "C++ Programming",            "resources": ["LearnCpp.com (free)", "The C++ Programming Language (Stroustrup)", "Competitive Programming 3"]},
    "c ":                   {"title": "C Programming",              "resources": ["CS50 Harvard (free)", "Beej's Guide to C", "The C Programming Language (K&R)"]},
    "kotlin":               {"title": "Kotlin for Android",         "resources": ["Kotlin Official Docs", "Android Basics in Kotlin (Google)", "Udacity Android Kotlin"]},
    "swift":                {"title": "Swift for iOS",              "resources": ["Swift.org Docs", "100 Days of Swift (free)", "Stanford CS193p"]},
    "dart":                 {"title": "Dart & Flutter",             "resources": ["Flutter Official Docs", "The Complete Flutter Dev Bootcamp (Udemy)", "Flutter YouTube Channel"]},
    "solidity":             {"title": "Solidity & Smart Contracts", "resources": ["CryptoZombies (free)", "Solidity Docs", "Hardhat Tutorial"]},
    # Math & Theory
    "statistics":           {"title": "Statistics & Probability",   "resources": ["Khan Academy Statistics (free)", "StatQuest YouTube (free)", "Think Stats (free book)"]},
    "linear algebra":       {"title": "Linear Algebra",             "resources": ["3Blue1Brown Essence of LA (free)", "MIT OCW 18.06 (free)", "Khan Academy LA"]},
    "calculus":             {"title": "Calculus",                   "resources": ["3Blue1Brown Essence of Calculus (free)", "Khan Academy Calculus", "MIT OCW 18.01"]},
    "discrete math":        {"title": "Discrete Mathematics",       "resources": ["MIT OCW 6.042J (free)", "Discrete Math (Rosen)", "TrevTutor YouTube"]},
    "data structures":      {"title": "Data Structures & Algorithms","resources": ["NeetCode.io (free)", "LeetCode 75 Study Plan", "Algorithms by Sedgewick (free)"]},
    "algorithms":           {"title": "Algorithms",                 "resources": ["NeetCode.io", "CLRS Introduction to Algorithms", "Codeforces Practice"]},
    # ML / AI
    "machine learning":     {"title": "Machine Learning",           "resources": ["Andrew Ng ML Specialization (Coursera)", "Hands-On ML (Géron)", "fast.ai Practical ML"]},
    "deep learning":        {"title": "Deep Learning",              "resources": ["deeplearning.ai Specialization", "fast.ai (free)", "PyTorch Tutorials (free)"]},
    "tensorflow":           {"title": "TensorFlow / Keras",         "resources": ["TensorFlow Official Docs", "Coursera TF Developer Cert", "Kaggle Courses"]},
    "pytorch":              {"title": "PyTorch Framework",          "resources": ["PyTorch Official Tutorials (free)", "fast.ai", "Zero to Mastery PyTorch"]},
    "scikit-learn":         {"title": "Scikit-learn",               "resources": ["Scikit-learn User Guide (free)", "Hands-On ML Ch. 2-3", "Kaggle ML Courses (free)"]},
    "huggingface":          {"title": "HuggingFace Transformers",   "resources": ["HuggingFace Course (free)", "HuggingFace Docs", "BERT Paper"]},
    "transformers":         {"title": "Transformers & BERT/GPT",    "resources": ["HuggingFace Course (free)", "Attention Is All You Need paper", "The Illustrated Transformer blog"]},
    "langchain":            {"title": "LangChain for LLM Apps",     "resources": ["LangChain Docs", "LangChain Cookbook GitHub", "DeepLearning.ai LangChain Course (free)"]},
    "vector databases":     {"title": "Vector Databases",           "resources": ["Pinecone Docs", "Weaviate Academy (free)", "ChromaDB GitHub"]},
    "mlflow":               {"title": "MLflow Experiment Tracking", "resources": ["MLflow Official Docs", "MLflow Quickstart Tutorial", "Databricks MLflow Course"]},
    "dvc":                  {"title": "Data Version Control (DVC)", "resources": ["DVC Official Docs", "Iterative.ai Tutorials", "DVC YouTube Channel"]},
    "kubeflow":             {"title": "Kubeflow MLOps Platform",    "resources": ["Kubeflow Docs", "Google Cloud Kubeflow Tutorial", "MLOps Specialization (Coursera)"]},
    # Databases
    "sql":                  {"title": "SQL & Databases",            "resources": ["SQLZoo (free)", "Mode Analytics SQL Tutorial (free)", "PostgreSQL Official Docs"]},
    "postgresql":           {"title": "PostgreSQL",                 "resources": ["PostgreSQL Tutorial (free)", "Use The Index Luke (free)", "PGExercises.com (free)"]},
    "mongodb":              {"title": "MongoDB (NoSQL)",            "resources": ["MongoDB University (free)", "MongoDB Docs", "Mongoose.js Docs"]},
    "redis":                {"title": "Redis Caching",              "resources": ["Redis Official Docs", "Redis University (free)", "Redis in Action (book)"]},
    "apache spark":         {"title": "Apache Spark",               "resources": ["Spark: The Definitive Guide", "DataBricks Community (free)", "Spark Official Docs"]},
    "kafka":                {"title": "Apache Kafka",               "resources": ["Confluent Kafka Tutorials (free)", "Kafka: The Definitive Guide", "Conduktor Academy"]},
    "airflow":              {"title": "Apache Airflow",             "resources": ["Airflow Official Docs", "Astronomer Academy (free)", "Airflow: The Hands-On Guide"]},
    "dbt":                  {"title": "dbt (Data Build Tool)",      "resources": ["dbt Learn (free)", "dbt Official Docs", "Analytics Engineering Bootcamp"]},
    "snowflake":            {"title": "Snowflake Data Warehouse",   "resources": ["Snowflake University (free)", "Snowflake Docs", "dbt + Snowflake Tutorial"]},
    "stored procedures":    {"title": "Stored Procedures & PL/pgSQL","resources": ["PostgreSQL PL/pgSQL Docs (free)", "SQLZoo Stored Procs", "Oracle PL/SQL Tutorial"]},
    "query optimisation":   {"title": "Query Optimisation & Indexing","resources": ["Use The Index Luke (free)", "Explain Analyze Docs", "High Performance MySQL"]},
    # Web & APIs
    "html":                 {"title": "HTML & CSS",                 "resources": ["MDN HTML (free)", "freeCodeCamp Responsive Web", "The Odin Project (free)"]},
    "react":                {"title": "React.js",                   "resources": ["React Official Docs", "Full Stack Open (free)", "Scrimba React Course"]},
    "node.js":              {"title": "Node.js & Express",          "resources": ["Node.js Docs", "The Odin Project Node (free)", "Full Stack Open Backend (free)"]},
    "rest api":             {"title": "REST API Design",            "resources": ["RESTful Web APIs (Richardson)", "REST API Design Rulebook", "OpenAPI Spec Docs"]},
    "graphql":              {"title": "GraphQL",                    "resources": ["GraphQL Official Docs (free)", "HowToGraphQL (free)", "Apollo GraphQL Tutorials"]},
    "authentication":       {"title": "Auth: JWT, OAuth 2.0",       "resources": ["Auth0 Docs (free)", "JWT.io Introduction", "OAuth 2.0 Simplified (free)"]},
    # Cloud & DevOps
    "docker":               {"title": "Docker Containers",          "resources": ["Docker Official Docs (free)", "TechWorld with Nana YouTube", "Play with Docker (free)"]},
    "kubernetes":           {"title": "Kubernetes",                 "resources": ["Kubernetes.io Docs (free)", "KodeKloud (free tier)", "Certified Kubernetes (CKAD)"]},
    "aws":                  {"title": "AWS Cloud Services",         "resources": ["AWS Free Tier + Docs", "A Cloud Guru", "AWS Skill Builder (free)"]},
    "terraform":            {"title": "Terraform IaC",              "resources": ["Terraform Official Docs (free)", "HashiCorp Learn (free)", "Terraform Up and Running"]},
    "ci/cd":                {"title": "CI/CD Pipelines",            "resources": ["GitHub Actions Docs (free)", "GitLab CI Docs", "Jenkins Official Docs"]},
    "helm":                 {"title": "Helm (Kubernetes Package Manager)","resources": ["Helm Official Docs (free)", "Artifact Hub Charts", "Helm Chart Best Practices"]},
    "linux":                {"title": "Linux Command Line",         "resources": ["Linux Journey (free)", "The Linux Command Line (free book)", "OverTheWire Bandit (free)"]},
    "monitoring":           {"title": "Monitoring: Prometheus & Grafana","resources": ["Prometheus Docs (free)", "Grafana Docs (free)", "TechWorld with Nana Monitoring"]},
    "bash":                 {"title": "Bash Scripting",             "resources": ["Bash Guide (free)", "ShellCheck Tool", "The Linux Command Line"]},
    # Security
    "networking":           {"title": "Computer Networking",        "resources": ["Professor Messer CompTIA N+ (free)", "Computer Networks (Tanenbaum)", "Cisco NetAcad"]},
    "kali linux":           {"title": "Kali Linux & Pentesting",    "resources": ["TryHackMe (free tier)", "Kali Linux Docs", "Offensive Security Docs"]},
    "metasploit":           {"title": "Metasploit Framework",       "resources": ["Offensive Security Metasploit Docs (free)", "TryHackMe Metasploit", "Metasploit Unleashed"]},
    "burp suite":           {"title": "Burp Suite Web Testing",     "resources": ["PortSwigger Web Academy (free)", "Burp Suite Docs", "TryHackMe OWASP"]},
    "owasp":                {"title": "OWASP Top 10",               "resources": ["OWASP.org (free)", "PortSwigger Web Academy (free)", "OWASP Testing Guide"]},
    "encryption":           {"title": "Cryptography & Encryption",  "resources": ["Cryptography I - Coursera (Stanford, free audit)", "Serious Cryptography (book)", "Crypto101 (free)"]},
    # Design
    "figma":                {"title": "Figma UI/UX Design",         "resources": ["Figma Official Tutorials (free)", "DesignCourse YouTube", "Figma Community"]},
    "adobe photoshop":      {"title": "Adobe Photoshop",            "resources": ["Adobe Tutorials (free)", "Phlearn YouTube", "Photoshop Essentials"]},
    "adobe illustrator":    {"title": "Adobe Illustrator",          "resources": ["Adobe Illustrator Tutorials (free)", "Illustrator How-To YouTube", "Skillshare Illustrator"]},
    "user research":        {"title": "UX Research Methods",        "resources": ["Nielsen Norman Group Articles (free)", "UX Beginner's Guide (free)", "Just Enough Research (book)"]},
    "after effects":        {"title": "Adobe After Effects",        "resources": ["Adobe AE Tutorials (free)", "Motion Array YouTube", "School of Motion"]},
    # Research & Academic
    "research methods":     {"title": "Research Methods",           "resources": ["Google Scholar (free)", "Research Methods in CS", "How to Read a Paper (Keshav)"]},
    "latex":                {"title": "LaTeX for Academic Writing", "resources": ["Overleaf Learn LaTeX (free)", "LaTeX Tutorial (free)", "ACM/IEEE Templates"]},
    "r":                    {"title": "R for Statistics",           "resources": ["R for Data Science (free book)", "Swirl R Course (free)", "RStudio Primers (free)"]},
    # Engineering
    "matlab":               {"title": "MATLAB / Simulink",          "resources": ["MATLAB OnRamp (free)", "MIT OCW MATLAB Tutorials", "MATLAB Central Community"]},
    "cad":                  {"title": "CAD (SolidWorks / AutoCAD)", "resources": ["SolidWorks Tutorials (free)", "AutoCAD LT Docs", "GrabCAD Community"]},
    "ros":                  {"title": "ROS (Robot Operating System)","resources": ["ROS Official Tutorials (free)", "The Construct ROS Courses", "ETH Zurich Programming for Robotics"]},
    "opencv":               {"title": "OpenCV Computer Vision",     "resources": ["OpenCV Official Docs (free)", "PyImageSearch Blog", "OpenCV Python Tutorials"]},
}

_df = None

def _load():
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)

def _enrich_step(step: str) -> dict:
    """Attach resource links to a roadmap step where possible."""
    step_lower = step.lower()
    resources = []
    for keyword, info in SKILL_RESOURCES.items():
        if keyword in step_lower:
            resources = info["resources"]
            break
    return {"step": step, "resources": resources}

def generate_roadmap(career_name: str, missing_skills: list[str]) -> dict:
    """
    Generate a personalized roadmap:
    - Pulls base roadmap from knowledge base
    - Prioritizes missing skill steps first
    - Appends project / internship milestones

    Returns:
        {career, domain, total_steps, completed_steps, roadmap_steps: [{step, resources, priority}]}
    """
    _load()

    row = _df[_df["career"].str.lower() == career_name.lower()]
    if row.empty:
        return {"error": f"Career '{career_name}' not found."}

    row = row.iloc[0]
    raw = str(row["roadmap"])
    base_steps = [s.strip() for s in raw.replace("→", "->").split("->")]

    missing_lower = set(s.lower() for s in missing_skills)

    roadmap_steps = []
    for step in base_steps:
        step_lower = step.lower()
        is_missing = any(m in step_lower for m in missing_lower)
        enriched = _enrich_step(step)
        enriched["priority"] = "high" if is_missing else "done"
        roadmap_steps.append(enriched)

    # Append standard milestones if not already present
    milestones = [
        {"step": f"Build 2-3 {row['career']} Projects", "resources": ["GitHub", "Kaggle", "Devpost"], "priority": "milestone"},
        {"step": "Contribute to Open Source",            "resources": ["GitHub Explore", "First Contributions"], "priority": "milestone"},
        {"step": "Apply for Internships / Jobs",         "resources": ["LinkedIn", "Internshala", "AngelList"],  "priority": "milestone"},
    ]
    existing_steps_lower = {s["step"].lower() for s in roadmap_steps}
    for m in milestones:
        if m["step"].lower() not in existing_steps_lower:
            roadmap_steps.append(m)

    completed = sum(1 for s in roadmap_steps if s["priority"] == "done")

    return {
        "career":          row["career"],
        "domain":          row["domain"],
        "total_steps":     len(roadmap_steps),
        "completed_steps": completed,
        "roadmap_steps":   roadmap_steps,
    }
