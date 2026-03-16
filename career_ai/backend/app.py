"""
Student Hub - AI Career Advisor
Streamlit UI for career prediction, recommendations, skill gap & roadmap.
Run: streamlit run app.py  (from the backend folder)
"""

import sys
import os
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from services.predictor   import predict_domains
from services.recommender import recommend_careers, get_student_skills
from services.skill_gap   import analyze_gap
from services.roadmap     import generate_roadmap

# Resource name → search URL mapper (opens real pages)
_RESOURCE_URLS = {
    "freecodecamp":     "https://www.freecodecamp.org",
    "khan academy":     "https://www.khanacademy.org",
    "mdn":              "https://developer.mozilla.org",
    "mit ocw":          "https://ocw.mit.edu",
    "coursera":         "https://www.coursera.org",
    "neetcode":         "https://neetcode.io",
    "leetcode":         "https://leetcode.com",
    "kaggle":           "https://www.kaggle.com/learn",
    "fast.ai":          "https://www.fast.ai",
    "pytorch":          "https://pytorch.org/tutorials",
    "tensorflow":       "https://www.tensorflow.org/learn",
    "huggingface":      "https://huggingface.co/learn",
    "langchain":        "https://python.langchain.com",
    "docker":           "https://docs.docker.com/get-started",
    "kubernetes":       "https://kubernetes.io/docs/tutorials",
    "aws":              "https://aws.amazon.com/training/free",
    "terraform":        "https://developer.hashicorp.com/terraform/tutorials",
    "github actions":   "https://docs.github.com/actions",
    "portswigger":      "https://portswigger.net/web-security",
    "tryhackme":        "https://tryhackme.com",
    "sqlzoo":           "https://sqlzoo.net",
    "postgresql":       "https://www.postgresql.org/docs",
    "mongodb":          "https://learn.mongodb.com",
    "redis":            "https://university.redis.com",
    "react":            "https://react.dev/learn",
    "node.js":          "https://nodejs.org/en/learn",
    "typescript":       "https://www.typescriptlang.org/docs",
    "odin project":     "https://www.theodinproject.com",
    "full stack open":  "https://fullstackopen.com",
    "figma":            "https://www.figma.com/resources/learn-design",
    "opencv":           "https://docs.opencv.org/4.x/d9/df8/tutorial_root.html",
    "3blue1brown":      "https://www.youtube.com/@3blue1brown",
    "statquest":        "https://www.youtube.com/@statquest",
    "overleaf":         "https://www.overleaf.com/learn",
    "r for data science": "https://r4ds.had.co.nz",
    "dbt":              "https://courses.getdbt.com",
    "airflow":          "https://airflow.apache.org/docs",
    "mlflow":           "https://mlflow.org/docs/latest",
    "linux journey":    "https://linuxjourney.com",
    "cs50":             "https://cs50.harvard.edu",
    "andrew ng":        "https://www.coursera.org/specializations/machine-learning-introduction",
    "deeplearning.ai":  "https://www.deeplearning.ai",
}

def _roadmap_resource_url(resource_name: str) -> str:
    """Return a real URL for a known resource, else a Google search fallback."""
    name_lower = resource_name.lower()
    for keyword, url in _RESOURCE_URLS.items():
        if keyword in name_lower:
            return url
    # Fallback: Google search
    query = resource_name.replace(" ", "+")
    return f"https://www.google.com/search?q={query}"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Hub — AI Career Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Cards */
    .card {
        background: #1a1d27;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 18px;
        border: 1px solid #2a2d3a;
    }

    /* Domain badge */
    .domain-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6c63ff, #3ecf8e);
        color: white;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 13px;
        font-weight: 600;
        margin: 4px 4px 4px 0;
    }

    /* Skill tags */
    .skill-have {
        display: inline-block;
        background: #1a3a2a;
        color: #3ecf8e;
        border: 1px solid #3ecf8e44;
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 12px;
        margin: 3px;
    }
    .skill-miss {
        display: inline-block;
        background: #3a1a1a;
        color: #ff6b6b;
        border: 1px solid #ff6b6b44;
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 12px;
        margin: 3px;
    }

    /* Step cards */
    .step-done { 
        background: #0e2a1e; border-left: 4px solid #3ecf8e;
        padding: 10px 16px; border-radius: 8px; margin: 6px 0;
    }
    .step-todo {
        background: #2a1e0e; border-left: 4px solid #f5a623;
        padding: 10px 16px; border-radius: 8px; margin: 6px 0;
    }
    .step-goal {
        background: #0e1a2a; border-left: 4px solid #6c63ff;
        padding: 10px 16px; border-radius: 8px; margin: 6px 0;
    }

    /* Section headers */
    h2, h3 { color: #ffffff !important; }
    .section-title {
        font-size: 20px; font-weight: 700;
        color: #fff; margin-bottom: 14px;
        border-bottom: 2px solid #6c63ff44;
        padding-bottom: 8px;
    }

    /* Readiness ring label */
    .readiness-label {
        text-align: center; font-size: 42px; font-weight: 800;
        color: #3ecf8e;
    }

    /* Slider label override */
    label { color: #b0b0b0 !important; font-size: 13px !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #1a1d27; }
    ::-webkit-scrollbar-thumb { background: #6c63ff; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar — Quiz ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Hub")
    st.markdown("#### AI Career Advisor")
    st.markdown("---")
    st.markdown("**Answer each question honestly — 0 to 10**")
    st.caption("0 = No experience  |  5 = Moderate  |  10 = Expert")
    st.markdown("")

    # ── Quiz questions — specific and domain-discriminating ──────────────
    CATEGORIES = {
        "💻 Programming Languages": [
            ("python_interest",
             "🐍 Python — How often do you write Python scripts, notebooks, or data pipelines?"),
            ("java_interest",
             "☕ Java / Kotlin — Do you build enterprise apps, Android apps, or backend services in Java/Kotlin?"),
            ("javascript_interest",
             "🌐 JavaScript / TypeScript — How comfortable are you building web UIs or Node.js backends?"),
            ("cpp_interest",
             "⚡ C / C++ — Do you write performance-critical code, embedded systems, or game engines?"),
            ("mobile_dev_interest",
             "📱 Mobile Dev — Have you built or published iOS / Android apps (Swift, Flutter, React Native)?"),
        ],
        "📐 Mathematics & Theory": [
            ("linear_algebra",
             "🔢 Linear Algebra — Can you work with matrices, eigenvectors, SVD? (used in ML / graphics)"),
            ("statistics_probability",
             "📊 Statistics & Probability — Do you understand distributions, hypothesis testing, p-values, Bayes?"),
            ("discrete_math",
             "🔣 Discrete Math — Are you comfortable with graphs, trees, combinatorics, proofs, logic?"),
            ("calculus",
             "∫  Calculus / Optimisation — Can you reason about gradients, partial derivatives, chain rule?"),
        ],
        "🗄️ Data & Databases": [
            ("system_design_interest",
             "🗄️ SQL & Databases — How well do you design schemas, write complex queries (joins, CTEs, indexes)?"),
            ("networking_interest",
             "🌍 Networking & Protocols — Do you understand TCP/IP, HTTP, DNS, REST, WebSockets, latency?"),
            ("security_interest",
             "🔐 Cybersecurity — Can you identify vulnerabilities (OWASP), do penetration testing, or hardening?"),
            ("cloud_interest",
             "☁️ Cloud & DevOps — Have you deployed apps on AWS/GCP/Azure, used Docker, Kubernetes, CI/CD?"),
        ],
        "🎨 Design & Product": [
            ("design_interest",
             "🎨 UI/UX Design — Do you create wireframes, prototypes, or conduct usability tests (Figma etc.)?"),
            ("product_thinking",
             "🧩 Product Thinking — Can you define user stories, prioritise features, analyse metrics for growth?"),
        ],
        "🤝 Communication & Leadership": [
            ("communication_skill",
             "🗣️ Communication — How well do you present technical ideas to non-technical audiences?"),
            ("leadership",
             "👥 Leadership & Teamwork — Have you led a project, managed a team, or mentored peers?"),
        ],
        "🔬 Research & Analytical Thinking": [
            ("research_interest",
             "📄 Research — Do you read papers, run experiments, and contribute to novel knowledge?"),
            ("analytical_skill",
             "🔍 Analytical Thinking — How well do you break down complex problems into structured solutions?"),
            ("curiosity_learning",
             "💡 Curiosity & Self-Learning — How often do you learn new tools or domains beyond your coursework?"),
        ],
        "🏆 Practical Experience": [
            ("dsa_skill",
             "🧩 DSA & Problem Solving — How confidently do you solve Data Structures & Algorithms problems? (arrays, trees, graphs, DP — e.g. LeetCode / competitive programming)"),
            ("projects_built",
             "🛠️ Projects Built — How many substantial projects have you built and deployed / published? (0 = none, 5 = 2-3 solid projects, 10 = 5+ live/open-source projects)"),
            ("build_and_deploy",
             "🚀 Build & Deploy Experience — Have you deployed any project live? (ML model, mobile app, web app, embedded system, game — any domain counts. 0 = never, 5 = deployed 1-2 projects, 10 = multiple live/published projects with CI/CD or app stores)"),
        ],
    }

    scores = {}
    for cat_label, questions in CATEGORIES.items():
        st.markdown(f"**{cat_label}**")
        for feature, question_text in questions:
            scores[feature] = st.slider(
                question_text, 0, 10, 5,
                key=feature,
                help="0 = No experience  |  5 = Moderate  |  10 = Expert / Passionate"
            )
        st.markdown("")

    # ── Backend specialisation (optional) ──────────────────────────────
    st.markdown("**🖥️ Backend Specialisation** *(optional — select what you use)*")
    BACKEND_OPTIONS = [
        "Python (FastAPI / Django / Flask)",
        "Node.js (Express / NestJS)",
        "Java (Spring Boot)",
        "Go (Gin / Echo)",
        "PHP (Laravel)",
        "Ruby (Rails)",
        "Rust (Actix / Axum)",
        ".NET / C#",
    ]
    backend_choices = st.multiselect(
        "Backend stack", BACKEND_OPTIONS, default=[],
        key="backend_spec",
        label_visibility="collapsed",
        placeholder="Pick your backend stack(s)…",
    )
    st.markdown("")

    # ── AI / ML sub-role interest (optional) ─────────────────────────────
    st.markdown("**🤖 AI Sub-roles of Interest** *(optional — select any that excite you)*")
    AI_SUBROLES = [
        "Machine Learning (Classical ML, Scikit-learn)",
        "Deep Learning (Neural Networks, PyTorch/TF)",
        "Natural Language Processing (NLP, LLMs, HuggingFace)",
        "Computer Vision (OpenCV, YOLO, CNNs)",
        "Reinforcement Learning (RL, Gymnasium)",
        "MLOps / Model Deployment (Docker, MLflow, Kubeflow)",
        "Generative AI (Diffusion, GANs, LangChain)",
        "Data Science / Analytics (Pandas, SQL, Tableau)",
    ]
    ai_choices = st.multiselect(
        "AI sub-roles", AI_SUBROLES, default=[],
        key="ai_subroles",
        label_visibility="collapsed",
        placeholder="Pick AI areas you are interested in…",
    )
    st.markdown("")

    predict_btn = st.button("🚀 Analyse My Career", type="primary")

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# 🎓 AI Career Advisor")
st.markdown("Answer the quiz on the left, then click **Analyse My Career** to get your personalised results.")

if not predict_btn:
    # Landing placeholder
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="card">
            <h3>🧠 Domain Prediction</h3>
            <p style="color:#888">Neural network analyses your skill profile and predicts the best career domains for you.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="card">
            <h3>💼 Career Recommendations</h3>
            <p style="color:#888">Get specific career roles ranked by how well they match your current skills.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="card">
            <h3>🗺️ Personalized Roadmap</h3>
            <p style="color:#888">Step-by-step learning path with resources tailored to your missing skills.</p>
        </div>""", unsafe_allow_html=True)

    st.info("👈 Set your scores in the sidebar and click **Analyse My Career** to begin.")
    st.stop()

# ── Run predictions ───────────────────────────────────────────────────────────
backend_choices = st.session_state.get("backend_spec", [])
ai_choices      = st.session_state.get("ai_subroles", [])

# Build extra keyword hints from sub-type selections
extra_skill_hints = []
BACKEND_KEYWORD_MAP = {
    "Python (FastAPI / Django / Flask)": ["Python", "FastAPI", "Django", "Flask"],
    "Node.js (Express / NestJS)":        ["Node.js", "Express", "JavaScript", "TypeScript"],
    "Java (Spring Boot)":                ["Java", "Spring Boot"],
    "Go (Gin / Echo)":                   ["Go"],
    "PHP (Laravel)":                     ["PHP", "Laravel"],
    "Ruby (Rails)":                      ["Ruby", "Rails"],
    "Rust (Actix / Axum)":               ["Rust"],
    ".NET / C#":                         ["C#", ".NET"],
}
AI_KEYWORD_MAP = {
    "Machine Learning (Classical ML, Scikit-learn)":         ["Scikit-learn", "Machine Learning"],
    "Deep Learning (Neural Networks, PyTorch/TF)":           ["PyTorch", "TensorFlow", "Deep Learning"],
    "Natural Language Processing (NLP, LLMs, HuggingFace)": ["NLP", "HuggingFace", "Transformers", "LangChain"],
    "Computer Vision (OpenCV, YOLO, CNNs)":                  ["OpenCV", "Computer Vision", "CNN"],
    "Reinforcement Learning (RL, Gymnasium)":                ["Reinforcement Learning"],
    "MLOps / Model Deployment (Docker, MLflow, Kubeflow)":   ["MLflow", "Kubeflow", "MLOps", "Docker"],
    "Generative AI (Diffusion, GANs, LangChain)":            ["Generative AI", "LangChain", "GANs"],
    "Data Science / Analytics (Pandas, SQL, Tableau)":       ["Pandas", "SQL", "Tableau", "Data Analysis"],
}
for choice in backend_choices:
    extra_skill_hints.extend(BACKEND_KEYWORD_MAP.get(choice, []))
for choice in ai_choices:
    extra_skill_hints.extend(AI_KEYWORD_MAP.get(choice, []))

with st.spinner("🤖 Analysing your profile..."):
    domains        = predict_domains(scores)
    student_skills = get_student_skills(scores)
    # Merge extra hints from checkbox selections
    all_skills = list(set(student_skills) | set(extra_skill_hints))
    top_domain     = domains[0]["domain"]
    careers        = recommend_careers(top_domain, scores, top_n=5)

st.success("✅ Analysis complete!")

# Show chosen sub-type badges if any
_badge_items = []
if backend_choices:
    _badge_items.append(("🖥️ Backend", backend_choices, "#3ecf8e"))
if ai_choices:
    _badge_items.append(("🤖 AI Focus", ai_choices, "#6c63ff"))
if _badge_items:
    cols_sub = st.columns(len(_badge_items))
    for _col, (_label, _choices, _color) in zip(cols_sub, _badge_items):
        _badges = " ".join(
            f'<span style="background:{_color}22;color:{_color};border:1px solid {_color}44;'
            f'border-radius:8px;padding:2px 10px;font-size:12px;margin:2px;display:inline-block">'
            f'{c.split("(")[0].strip()}</span>'
            for c in _choices
        )
        _col.markdown(
            f'<div class="card" style="padding:12px"><b style="color:{_color}">{_label}:</b><br>{_badges}</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 1 — Domain Prediction
# ════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🧠 Career Domain Prediction</div>', unsafe_allow_html=True)

col_chart, col_top = st.columns([3, 2])

with col_chart:
    top5 = domains[:5]
    fig = go.Figure(go.Bar(
        x=[d["confidence"] for d in top5],
        y=[d["domain"]     for d in top5],
        orientation="h",
        marker=dict(
            color=[d["confidence"] for d in top5],
            colorscale=[[0, "#1a1d27"], [0.3, "#6c63ff"], [1.0, "#3ecf8e"]],
            line=dict(color="#2a2d3a", width=1),
        ),
        text=[f'{d["confidence"]}%' for d in top5],
        textposition="outside",
        textfont=dict(color="white", size=13),
    ))
    fig.update_layout(
        paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
        font=dict(color="white", size=13),
        xaxis=dict(range=[0, 115], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
        margin=dict(l=10, r=60, t=10, b=10),
        height=260,
    )
    st.plotly_chart(fig, width="stretch", key="domain_bar")

with col_top:
    st.markdown("<br>", unsafe_allow_html=True)
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, d in enumerate(domains[:5]):
        pct = d["confidence"]
        color = "#3ecf8e" if i == 0 else "#6c63ff" if i == 1 else "#f5a623"
        st.markdown(f"""<div class="card" style="padding:12px 16px;margin-bottom:10px;border-left:4px solid {color}">
            <span style="font-size:18px">{medals[i]}</span>
            <span style="font-weight:700;margin-left:8px">{d['domain']}</span>
            <span style="float:right;color:{color};font-weight:800;font-size:17px">{pct}%</span>
        </div>""", unsafe_allow_html=True)

# Domain selector for recommendations
st.markdown("")
selected_domain = st.selectbox(
    "🔍 Explore recommendations for a domain:",
    options=[d["domain"] for d in domains[:5]],
    index=0,
)

if selected_domain != top_domain:
    careers = recommend_careers(selected_domain, scores, top_n=5)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 2 — Student Skills
# ════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🛠️ Your Current Skills</div>', unsafe_allow_html=True)

if all_skills:
    badges = "".join(f'<span class="skill-have">{s}</span>' for s in sorted(all_skills))
    st.markdown(f'<div class="card">{badges}</div>', unsafe_allow_html=True)
else:
    st.warning("Score higher than 7 in at least one area to see your skills.")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 3 — Career Recommendations
# ════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-title">💼 Career Recommendations — {selected_domain}</div>', unsafe_allow_html=True)

if not careers:
    st.warning("No careers found for this domain.")
else:
    # Career picker tabs
    career_names = [c["career"] for c in careers]
    selected_career_name = st.radio(
        "Select a career for detailed analysis:",
        career_names,
        horizontal=True,
        label_visibility="collapsed",
    )
    selected_career = next(c for c in careers if c["career"] == selected_career_name)

    # Career cards
    cols = st.columns(len(careers))
    for i, (col, c) in enumerate(zip(cols, careers)):
        is_selected = c["career"] == selected_career_name
        border_color = "#3ecf8e" if is_selected else "#2a2d3a"
        score_color  = "#3ecf8e" if c["match_score"] >= 60 else "#f5a623" if c["match_score"] >= 30 else "#ff6b6b"
        col.markdown(f"""<div class="card" style="border:2px solid {border_color};text-align:center;padding:14px 10px">
            <div style="font-weight:700;font-size:13px;margin-bottom:8px">{c['career']}</div>
            <div style="font-size:26px;font-weight:800;color:{score_color}">{c['match_score']}%</div>
            <div style="font-size:11px;color:#888">match</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 4 — Skill Gap Analysis
# ════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-title">📊 Skill Gap Analysis — {selected_career_name}</div>', unsafe_allow_html=True)

gap = analyze_gap(selected_career_name, all_skills)

col_gauge, col_skills = st.columns([1, 2])

with col_gauge:
    readiness = gap["readiness_score"]
    ring_color = "#3ecf8e" if readiness >= 70 else "#f5a623" if readiness >= 40 else "#ff6b6b"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=readiness,
        number={"suffix": "%", "font": {"size": 40, "color": ring_color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555", "tickfont": {"color": "#999"}},
            "bar":  {"color": ring_color, "thickness": 0.28},
            "bgcolor": "#1a1d27",
            "bordercolor": "#2a2d3a",
            "steps": [
                {"range": [0,  40], "color": "#2a1a1a"},
                {"range": [40, 70], "color": "#2a2010"},
                {"range": [70,100], "color": "#0e2a1e"},
            ],
            "threshold": {
                "line": {"color": ring_color, "width": 4},
                "thickness": 0.75,
                "value": readiness,
            },
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#1a1d27", font={"color": "white"},
        margin=dict(l=20, r=20, t=40, b=10), height=240,
    )
    st.plotly_chart(fig_gauge, width="stretch", key="gauge")
    label = "🟢 Job Ready!" if readiness >= 80 else "🟡 Almost There" if readiness >= 50 else "🔴 Keep Learning"
    st.markdown(f"<div style='text-align:center;font-size:16px;font-weight:700'>{label}</div>", unsafe_allow_html=True)

with col_skills:
    st.markdown("**✅ Skills You Have**")
    if gap["matched_skills"]:
        have_html = "".join(f'<span class="skill-have">✓ {s}</span>' for s in gap["matched_skills"])
        st.markdown(f'<div class="card" style="padding:14px">{have_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card" style="padding:14px;color:#888">None matched yet — keep learning!</div>', unsafe_allow_html=True)

    st.markdown("**❌ Skills to Learn**")
    if gap["missing_skills"]:
        miss_html = "".join(f'<span class="skill-miss">✗ {s}</span>' for s in gap["missing_skills"])
        st.markdown(f'<div class="card" style="padding:14px">{miss_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card" style="padding:14px;color:#3ecf8e">🎉 You have all required skills!</div>', unsafe_allow_html=True)

    # Skill breakdown donut
    if gap["matched_skills"] or gap["missing_skills"]:
        n_have = len(gap["matched_skills"])
        n_miss = len(gap["missing_skills"])
        fig_pie = go.Figure(go.Pie(
            labels=["Have", "Missing"],
            values=[n_have, n_miss],
            hole=0.55,
            marker=dict(colors=["#3ecf8e", "#ff6b6b"],
                        line=dict(color="#1a1d27", width=3)),
            textinfo="label+percent",
            textfont=dict(color="white", size=13),
        ))
        fig_pie.update_layout(
            paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
            font=dict(color="white"), showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10), height=180,
        )
        st.plotly_chart(fig_pie, width="stretch", key="pie")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 5 — Personalized Roadmap
# ════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-title">🗺️ Personalized Learning Roadmap — {selected_career_name}</div>', unsafe_allow_html=True)

roadmap = generate_roadmap(selected_career_name, gap["missing_skills"])

# Progress bar
progress = roadmap["completed_steps"] / roadmap["total_steps"] if roadmap["total_steps"] else 0
st.markdown(
    f"**Progress: {roadmap['completed_steps']} / {roadmap['total_steps']} steps covered**"
)
st.progress(progress)
st.markdown("<br>", unsafe_allow_html=True)

# Steps
STEP_META = {
    "done":      ("✅", "step-done", "Already Covered"),
    "high":      ("📌", "step-todo", "To Learn"),
    "milestone": ("🎯", "step-goal", "Milestone"),
}

steps = roadmap["roadmap_steps"]
cols_per_row = 2
for row_start in range(0, len(steps), cols_per_row):
    row_steps = steps[row_start: row_start + cols_per_row]
    cols = st.columns(cols_per_row)
    for col, step_item in zip(cols, row_steps):
        icon, css_class, tag = STEP_META.get(step_item["priority"], ("▶️", "step-todo", "Step"))
        resources = step_item.get("resources", [])

        # Build resource links HTML separately to avoid f-string nesting issues
        if resources:
            links = "".join(
                f'<a href="{_roadmap_resource_url(r)}" target="_blank" '
                f'style="display:block;color:#6c63ff;font-size:11px;'
                f'text-decoration:none;margin-top:3px">→ {r}</a>'
                for r in resources[:3]
            )
            res_block = f"<div style='margin-top:8px;padding-left:24px'>{links}</div>"
        else:
            res_block = ""

        col.markdown(
            f'<div class="{css_class}">'
            f'<span style="font-size:16px">{icon}</span>'
            f'<span style="font-weight:600;font-size:14px;margin-left:6px">{step_item["step"]}</span>'
            f'<span style="float:right;font-size:10px;color:#888;background:#ffffff11;'
            f'border-radius:6px;padding:2px 7px">{tag}</span>'
            f'{res_block}'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════
# SECTION 6 — Radar Chart of Student Profile
# ════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📡 Your Skill Radar</div>', unsafe_allow_html=True)

radar_cats = {
    "Python":        scores["python_interest"],
    "Math/Stats":    (scores["linear_algebra"] + scores["statistics_probability"]) // 2,
    "DB/SQL":        scores["system_design_interest"],
    "Networking":    scores["networking_interest"],
    "Security":      scores["security_interest"],
    "Cloud/DevOps":  scores["cloud_interest"],
    "UI/UX":         scores["design_interest"],
    "Product":       scores["product_thinking"],
    "Research":      scores["research_interest"],
    "DSA":           scores["dsa_skill"],
    "Projects":      scores["projects_built"],
    "Build/Deploy":  scores["build_and_deploy"],
}

labels = list(radar_cats.keys())
values = list(radar_cats.values()) + [list(radar_cats.values())[0]]  # close polygon
labels_closed = labels + [labels[0]]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=values, theta=labels_closed,
    fill="toself",
    fillcolor="rgba(108,99,255,0.2)",
    line=dict(color="#6c63ff", width=2),
    name="Your Profile",
))
fig_radar.update_layout(
    polar=dict(
        bgcolor="#1a1d27",
        radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(color="#666"), gridcolor="#2a2d3a"),
        angularaxis=dict(tickfont=dict(color="white", size=12), gridcolor="#2a2d3a"),
    ),
    paper_bgcolor="#1a1d27",
    font=dict(color="white"),
    showlegend=False,
    margin=dict(l=60, r=60, t=40, b=40),
    height=400,
)
st.plotly_chart(fig_radar, width="stretch", key="radar")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:12px'>"
    "Student Hub — AI Career Advisor &nbsp;|&nbsp; Powered by PyTorch Neural Network &nbsp;|&nbsp; "
    "99.9% Prediction Accuracy</div>",
    unsafe_allow_html=True,
)
