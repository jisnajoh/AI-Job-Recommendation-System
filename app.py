import re
import io
from pathlib import Path

import requests
import pandas as pd
import streamlit as st
import pdfplumber
import docx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# CONFIG
# =========================
DAILY_DATA_DIR = "Data Sets/daily"
FALLBACK_DATA_PATH = "Data Sets/cleaned/postings_clean.csv"
SHOW_TOP_SKILLS = 12

# Put your own Adzuna credentials here
APP_ID = "44463cba"
APP_KEY = "37d12d7510bd6a36375704d1a2f0dfbc"

COMMON_JOB_SKILLS = {
    "communication", "reporting", "analysis", "documentation",
    "problem solving", "excel", "sql", "power bi",
    "tableau", "customer service", "project management",
    "dashboards", "data visualization", "leadership",
    "business analysis", "stakeholder management"
}


# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="AI Job Recommendation System",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
.hero {
    padding: 1.25rem 1.4rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #eff6ff 0%, #f5f3ff 50%, #ecfeff 100%);
    border: 1px solid #e5e7eb;
    margin-bottom: 1rem;
}
.hero h1 {
    margin: 0;
    font-size: 2.15rem;
}
.hero p {
    margin: 0.45rem 0 0 0;
    color: #4b5563;
    font-size: 1rem;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 0.9rem 1rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.03);
}
.metric-title {
    font-size: 0.9rem;
    color: #6b7280;
    margin-bottom: 0.2rem;
}
.metric-value {
    font-size: 1.95rem;
    font-weight: 700;
    color: #111827;
}
.badge {
    display: inline-block;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
}
.badge-blue { background: #dbeafe; color: #1d4ed8; }
.badge-green { background: #dcfce7; color: #166534; }
.badge-purple { background: #ede9fe; color: #6d28d9; }
.badge-orange { background: #ffedd5; color: #c2410c; }
.badge-pink { background: #fce7f3; color: #be185d; }
.badge-gray { background: #f3f4f6; color: #374151; }
.section-title {
    font-size: 1.55rem;
    font-weight: 700;
    margin-top: 0.2rem;
    margin-bottom: 0.8rem;
}
.job-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 1rem 1.1rem;
    box-shadow: 0 5px 16px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}
.job-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.5rem;
}
.job-meta {
    font-size: 1rem;
    color: #374151;
    margin-bottom: 0.45rem;
}
.small-muted {
    color: #6b7280;
    font-size: 0.92rem;
}
.sidebar-note {
    color: #6b7280;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>💼 AI Job Recommendation System</h1>
  <p>Upload a resume, enter skills, or choose a target role to discover matching jobs, missing skills, and job-market trends across multiple industries.</p>
</div>
""", unsafe_allow_html=True)


# =========================
# MULTI-INDUSTRY SKILL ONTOLOGY
# =========================
SKILL_ONTOLOGY = {
    "Data & IT": {
        "python", "sql", "excel", "power bi", "tableau", "pandas", "numpy",
        "scikit-learn", "machine learning", "deep learning", "nlp",
        "data analysis", "data analytics", "data visualization", "data mining",
        "statistics", "matplotlib", "seaborn", "aws", "azure", "gcp",
        "cloud computing", "flask", "django", "html", "css", "javascript",
        "java", "c++", "git", "docker", "linux", "api", "mongodb",
        "mysql", "postgresql", "sqlite", "spark", "hadoop", "etl",
        "business intelligence", "reporting", "dashboards"
    },
    "Healthcare": {
        "patient care", "clinical assessment", "triage", "diagnosis",
        "treatment planning", "care coordination", "patient education",
        "medical records", "ehr", "emr", "epic", "cerner", "hipaa",
        "infection control", "primary care", "acute care", "pharmacology",
        "nurse practitioner", "registered nurse", "healthcare administration",
        "patient scheduling", "vital signs", "clinical documentation",
        "bcls", "bls", "acls", "cpr", "telehealth", "case management",
        "medical terminology"
    },
    "Accounting & Finance": {
        "accounting", "bookkeeping", "financial reporting", "forecasting",
        "budgeting", "audit", "auditing", "gaap", "accounts payable",
        "accounts receivable", "bank reconciliation", "payroll",
        "financial analysis", "tax preparation", "quickbooks", "sap",
        "oracle", "erp", "general ledger", "balance sheet", "income statement",
        "cash flow", "variance analysis", "financial modeling", "invoice processing",
        "reconciliation", "excel"
    },
    "Business & Operations": {
        "business analysis", "stakeholder management", "project management",
        "requirements gathering", "process improvement", "operations management",
        "reporting", "documentation", "workflow optimization", "strategic planning",
        "crm", "salesforce", "leadership", "communication", "problem solving",
        "cross-functional collaboration", "kpi tracking", "scheduling",
        "vendor management", "supply chain", "operations analysis",
        "customer service", "administration", "planning", "coordination"
    },
    "Marketing": {
        "seo", "sem", "google analytics", "social media", "campaign management",
        "content marketing", "content creation", "email marketing", "brand strategy",
        "market research", "advertising", "copywriting", "crm", "hubspot",
        "digital marketing", "branding", "lead generation", "analytics"
    },
    "Human Resources": {
        "recruitment", "talent acquisition", "employee relations", "onboarding",
        "benefits administration", "payroll", "performance management",
        "hris", "workday", "policy compliance", "training", "interviewing",
        "candidate screening", "staffing", "human resources", "labor relations"
    },
    "Education": {
        "teaching", "curriculum development", "lesson planning", "student engagement",
        "classroom management", "assessment", "grading", "academic advising",
        "instruction", "training", "education technology", "student support"
    }
}

GENERIC_NOT_SKILLS = {
    "experience", "knowledge", "skills", "ability", "responsible", "responsibilities",
    "team", "teamwork", "organized", "organization", "development", "project",
    "projects", "management", "tools", "tool", "work", "working", "support",
    "operations", "business", "computer", "university", "education", "training",
    "certification", "certifications", "role", "roles", "candidate", "job",
    "position", "positions", "strong", "excellent", "preferred", "required",
    "ability to", "must have", "should have"
}

SKILL_ALIASES = {
    "powerbi": "power bi",
    "microsoft power bi": "power bi",
    "structured query language": "sql",
    "aws s3": "aws",
    "amazon web services": "aws",
    "scikit learn": "scikit-learn",
    "electronic health record": "ehr",
    "electronic health records": "ehr",
    "electronic medical record": "emr",
    "electronic medical records": "emr",
    "search engine optimization": "seo",
    "search engine marketing": "sem",
    "customer relationship management": "crm",
    "human resource information system": "hris",
    "ap": "accounts payable",
    "ar": "accounts receivable",
    "electronic charting": "ehr",
    "electronic charting systems": "ehr",
    "bi": "business intelligence",
    "ms excel": "excel",
    "microsoft excel": "excel"
}

INDUSTRY_RULES = {
    "Healthcare": [
        "hospital", "clinic", "patient", "nurse", "physician", "medical",
        "health", "ehr", "emr", "hipaa", "clinical", "care coordination",
        "nurse practitioner", "registered nurse"
    ],
    "Accounting & Finance": [
        "bank", "finance", "financial", "accounting", "audit", "auditing",
        "tax", "payroll", "gaap", "bookkeeping", "accounts payable",
        "accounts receivable", "quickbooks"
    ],
    "Marketing": [
        "marketing", "seo", "sem", "content", "campaign", "brand",
        "social media", "advertising", "digital marketing"
    ],
    "Business & Operations": [
        "business", "operations", "stakeholder", "requirements",
        "process improvement", "project management", "workflow",
        "reporting", "coordination", "administration"
    ],
    "Human Resources": [
        "recruitment", "talent", "hr", "onboarding", "employee relations",
        "hris", "staffing", "human resources"
    ],
    "Data & IT": [
        "software", "developer", "cloud", "aws", "azure", "sql", "python",
        "analytics", "machine learning", "data scientist", "data engineer",
        "api", "database", "power bi", "tableau"
    ],
    "Education": [
        "university", "school", "teacher", "student", "education", "curriculum"
    ],
    "Retail": [
        "retail", "store", "merchandise", "inventory", "customer"
    ]
}


# =========================
# HELPERS
# =========================
def normalize_text(x) -> str:
    x = "" if pd.isna(x) else str(x)
    x = x.replace("\xa0", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def parse_skill_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(s).strip().lower() for s in x if str(s).strip()]

    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        s = re.sub(r"[\[\]\'\"]", "", s)
    return [p.strip().lower() for p in s.split(",") if p.strip()]


def looks_like_skill(s: str) -> bool:
    s = (s or "").strip().lower()
    if not (2 <= len(s) <= 50):
        return False
    if s in GENERIC_NOT_SKILLS:
        return False
    if len(s.split()) > 5:
        return False
    if not re.search(r"[a-z0-9]", s):
        return False
    return True


def get_skill_dictionary(selected_industry="All"):
    if selected_industry == "All":
        combined = set()
        for skills in SKILL_ONTOLOGY.values():
            combined.update(skills)
        return combined
    return SKILL_ONTOLOGY.get(selected_industry, set())


def infer_target_industry_from_user_text(user_text: str):
    t = (user_text or "").lower()
    for ind, kws in INDUSTRY_RULES.items():
        if any(k in t for k in kws):
            return ind
    return None


def extract_requirement_phrases(text: str):
    text = text.lower()
    candidates = set()

    patterns = [
        r"(?:experience with|proficiency in|knowledge of|expertise in|skilled in)\s+([a-zA-Z0-9\+\#\/\-\s,]{3,200})",
        r"(?:required skills?|preferred skills?|qualifications?|requirements?)[:\s]+([a-zA-Z0-9\+\#\/\-\s,]{3,220})",
        r"(?:familiarity with|ability to use|hands[- ]on experience with)\s+([a-zA-Z0-9\+\#\/\-\s,]{3,200})",
        r"(?:must have|should have|nice to have)\s+([a-zA-Z0-9\+\#\/\-\s,]{3,200})"
    ]

    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        for m in matches:
            parts = re.split(r",|/|;|\band\b|\bor\b", m)
            for p in parts:
                p = p.strip(" .:-").lower()
                if looks_like_skill(p):
                    candidates.add(p)

    EXTRA_JOB_TERMS = {
        "sql", "python", "excel", "power bi", "tableau", "reporting",
        "dashboards", "data visualization", "statistics", "communication",
        "problem solving", "stakeholder management", "business analysis",
        "documentation", "project management", "data mining",
        "forecasting", "budgeting", "patient care", "ehr", "emr",
        "quickbooks", "accounting", "financial analysis"
    }

    for term in EXTRA_JOB_TERMS:
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        if re.search(pattern, text):
            candidates.add(term)

    return sorted(candidates)


def extract_skills_from_text(text_in: str, selected_industry="All"):
    text = (text_in or "").lower()
    text = re.sub(r"\s+", " ", text)
    found = set()

    skill_dict = get_skill_dictionary(selected_industry)

    for sk in sorted(skill_dict, key=len, reverse=True):
        pattern = r"(?<!\w)" + re.escape(sk) + r"(?!\w)"
        if re.search(pattern, text):
            found.add(sk)

    for alias, canon in SKILL_ALIASES.items():
        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        if re.search(pattern, text):
            found.add(canon)

    if len(found) < 6:
        phrase_candidates = extract_requirement_phrases(text)
        for p in phrase_candidates:
            if 1 <= len(p.split()) <= 4 and len(p) <= 40:
                found.add(p)

    return sorted([s for s in found if looks_like_skill(s)])


def read_resume(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if name.endswith(".txt"):
        return normalize_text(file_bytes.decode("utf-8", errors="ignore"))

    if name.endswith(".docx"):
        d = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join([p.text for p in d.paragraphs if p.text])
        return normalize_text(text)

    if name.endswith(".pdf"):
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages[:10]:
                t = page.extract_text() or ""
                if t.strip():
                    text_parts.append(t)
        return normalize_text("\n".join(text_parts))

    return ""


def skill_match(user_skills: list[str], job_skills_list: list[str]):
    us = set([s.lower().strip() for s in user_skills if s.strip()])
    js = set([s.lower().strip() for s in job_skills_list if s.strip()])

    matched = sorted(list(us.intersection(js)))
    missing = sorted(list(js - us))
    extra_user_skills = sorted(list(us - js))

    score = 0.0 if len(js) == 0 else (len(matched) / len(js))
    return score, matched, missing, extra_user_skills


def build_job_text(df):
    return (
        df["job_title"].fillna("") + " | " +
        df["job_description"].fillna("") + " | " +
        df["job_skills"].fillna("") + " | " +
        df["company"].fillna("") + " | " +
        df["job_location"].fillna("")
    )


def get_latest_csv(folder: str) -> str:
    p = Path(folder)
    if not p.exists():
        return FALLBACK_DATA_PATH
    csvs = sorted([x for x in p.glob("*.csv")])
    return str(csvs[-1]) if csvs else FALLBACK_DATA_PATH


@st.cache_data
def load_data(path, selected_industry="All"):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "job level" in df.columns and "job_level" not in df.columns:
        df = df.rename(columns={"job level": "job_level"})

    required = [
        "job_title", "company", "job_location", "job_link",
        "search_country", "job_level", "job_type",
        "job_description", "job_skills"
    ]
    for c in required:
        if c not in df.columns:
            df[c] = ""

    for c in required:
        df[c] = df[c].apply(normalize_text)

    if "job_skills_list" in df.columns:
        df["job_skills_list"] = df["job_skills_list"].apply(parse_skill_list)
    else:
        df["job_skills_list"] = df["job_skills"].apply(parse_skill_list)

    df["job_skills_list"] = df.apply(
        lambda row: row["job_skills_list"] if len(row["job_skills_list"]) > 0
        else extract_skills_from_text(
            f"{row['job_title']} {row['job_description']}",
            selected_industry="All"
        ),
        axis=1
    )

    df["job_skills"] = df["job_skills_list"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else ""
    )

    df["job_text"] = build_job_text(df)

    if "job_link" in df.columns:
        df = df.drop_duplicates(subset=["job_link"])

    return df.reset_index(drop=True)


@st.cache_data
def fit_tfidf(job_texts: list[str]):
    vec = TfidfVectorizer(stop_words="english", max_features=40000, ngram_range=(1, 2))
    X = vec.fit_transform(job_texts)
    return vec, X


def infer_industry(row) -> str:
    text = f"{row.get('job_title','')} {row.get('job_description','')}".lower()
    for ind, kws in INDUSTRY_RULES.items():
        if any(k in text for k in kws):
            return ind
    return "Other"


def infer_work_mode(row) -> str:
    text = f"{row.get('job_type','')} {row.get('job_description','')} {row.get('job_title','')}".lower()
    if "remote" in text or "work from home" in text or "wfh" in text:
        return "Remote"
    if "hybrid" in text:
        return "Hybrid"
    return "Onsite"


def normalize_level(x: str) -> str:
    t = str(x).lower()
    if "intern" in t:
        return "Intern"
    if "entry" in t or "junior" in t:
        return "Entry"
    if "senior" in t or "lead" in t or "manager" in t:
        return "Senior"
    if t.strip() == "" or t == "nan":
        return "Mid"
    return "Mid"


# =========================
# LIVE JOB API
# =========================
def fetch_live_jobs(query, selected_industry="All"):
    url = "https://api.adzuna.com/v1/api/jobs/us/search/1"
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": 50,
        "what": query,
        "content-type": "application/json"
    }

    response = requests.get(url, params=params, timeout=30)
    data = response.json()

    jobs = []
    for job in data.get("results", []):
        company_name = ""
        if isinstance(job.get("company"), dict):
            company_name = job["company"].get("display_name", "")

        location_name = ""
        if isinstance(job.get("location"), dict):
            location_name = job["location"].get("display_name", "")

        job_title = normalize_text(job.get("title", ""))
        job_description = normalize_text(job.get("description", ""))

        extracted_job_skills = extract_skills_from_text(
            f"{job_title} {job_description}",
            selected_industry="All"
        )

        jobs.append({
            "job_title": job_title,
            "company": normalize_text(company_name),
            "job_location": normalize_text(location_name),
            "job_description": job_description,
            "job_link": normalize_text(job.get("redirect_url", "")),
            "job_skills": ", ".join(extracted_job_skills),
            "job_skills_list": extracted_job_skills,
            "search_country": "United States",
            "job_level": "",
            "job_type": "",
        })

    df = pd.DataFrame(jobs)

    if len(df) == 0:
        for c in [
            "job_title", "company", "job_location", "job_link",
            "search_country", "job_level", "job_type",
            "job_description", "job_skills", "job_skills_list"
        ]:
            if c not in df.columns:
                df[c] = ""

    df["job_text"] = build_job_text(df)
    return df


# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## 🔎 Filters")
st.sidebar.markdown(
    '<div class="sidebar-note">Choose your data source, target role, industry, and preferences.</div>',
    unsafe_allow_html=True
)

data_source = st.sidebar.selectbox(
    "🌐 Data Source",
    ["Dataset", "Live Jobs API"]
)

target_industry_ui = st.sidebar.selectbox(
    "🧭 Target Industry",
    [
        "All",
        "Data & IT",
        "Healthcare",
        "Accounting & Finance",
        "Business & Operations",
        "Marketing",
        "Human Resources",
        "Education"
    ]
)

desired_role = st.sidebar.text_input(
    "🎯 Target role or profession",
    value="",
    placeholder="Example: nurse practitioner, accountant, business analyst, marketing manager, data analyst"
)

only_usa = st.sidebar.checkbox("🇺🇸 Only USA jobs", value=True)
top_k = st.sidebar.slider("📌 Results to show", 5, 30, 10)

nlp_weight = st.sidebar.slider("⚖️ Weight: Resume Similarity", 0.0, 1.0, 0.45, 0.05)
skill_weight = st.sidebar.slider("⚖️ Weight: Skill Match", 0.0, 1.0, 0.45, 0.05)
role_weight = max(0.0, 1.0 - nlp_weight - skill_weight)

if (nlp_weight + skill_weight) > 1.0:
    skill_weight = 1.0 - nlp_weight
    role_weight = 0.0

st.sidebar.caption(
    f"Final score = {nlp_weight:.2f} × similarity + {skill_weight:.2f} × skills + {role_weight:.2f} × role relevance"
)


# =========================
# LOAD DATA
# =========================
DATA_PATH = get_latest_csv(DAILY_DATA_DIR)

if data_source == "Dataset":
    df = load_data(DATA_PATH, selected_industry=target_industry_ui)
else:
    st.sidebar.info("Fetching live jobs from API...")
    query = desired_role.strip() if desired_role.strip() else "jobs"
    df = fetch_live_jobs(query, selected_industry=target_industry_ui)


# =========================
# PREP DATA
# =========================
work = df.copy()

if "work_mode" not in work.columns:
    work["work_mode"] = work.apply(infer_work_mode, axis=1)

if "industry" not in work.columns:
    work["industry"] = work.apply(infer_industry, axis=1)

work["job_level_norm"] = work["job_level"].apply(normalize_level)

if only_usa:
    work = work[work["search_country"].str.lower().str.contains("united states|usa|us", na=False)]


# =========================
# INPUTS
# =========================
left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    st.markdown('<div class="section-title">📄 Upload Resume</div>', unsafe_allow_html=True)
    resume_file = st.file_uploader("PDF / DOCX / TXT", type=["pdf", "docx", "txt"])
    resume_text = read_resume(resume_file)

with right_col:
    st.markdown('<div class="section-title">✍️ Or Type Skills</div>', unsafe_allow_html=True)
    manual_skills = st.text_input(
        "Example: sql, patient care, quickbooks, business analysis, power bi",
        ""
    )

typed_skills = [s.strip().lower() for s in manual_skills.split(",") if s.strip()]
resume_skills = extract_skills_from_text(resume_text, target_industry_ui) if resume_text.strip() else []
manual_detected_skills = extract_skills_from_text(manual_skills, target_industry_ui) if manual_skills.strip() else []

if target_industry_ui == "All":
    resume_skills = sorted(set(resume_skills + extract_skills_from_text(resume_text, "All")))
    manual_detected_skills = sorted(set(manual_detected_skills + extract_skills_from_text(manual_skills, "All")))

user_skills = sorted(set(typed_skills + manual_detected_skills + resume_skills))

user_text_for_industry = f"{desired_role} {manual_skills} {resume_text}"
target_industry_from_text = infer_target_industry_from_user_text(user_text_for_industry)


# =========================
# FILTERS WITH SESSION STATE
# =========================
st.sidebar.markdown("---")

work_mode_options = ["All", "Remote", "Hybrid", "Onsite"]
career_level_options = ["All", "Intern", "Entry", "Mid", "Senior"]

if "work_mode_option" not in st.session_state:
    st.session_state["work_mode_option"] = "All"

if "career_level_option" not in st.session_state:
    st.session_state["career_level_option"] = "All"

inds = ["All"] + sorted(work["industry"].dropna().unique().tolist())

if "industry_option" not in st.session_state:
    default_industry = "All"
    if target_industry_ui != "All" and target_industry_ui in inds:
        default_industry = target_industry_ui
    elif target_industry_from_text in inds:
        default_industry = target_industry_from_text
    st.session_state["industry_option"] = default_industry

st.sidebar.selectbox(
    "🏠 Work Mode",
    work_mode_options,
    key="work_mode_option"
)

st.sidebar.selectbox(
    "🎓 Career Level",
    career_level_options,
    key="career_level_option"
)

if st.session_state["industry_option"] not in inds:
    st.session_state["industry_option"] = "All"

st.sidebar.selectbox(
    "🏭 Industry",
    inds,
    key="industry_option"
)

work_mode_option = st.session_state["work_mode_option"]
career_level_option = st.session_state["career_level_option"]
industry_option = st.session_state["industry_option"]

if work_mode_option != "All":
    work = work[work["work_mode"] == work_mode_option]

if career_level_option != "All":
    work = work[work["job_level_norm"] == career_level_option]

if industry_option != "All":
    work = work[work["industry"] == industry_option]

st.sidebar.caption(f"Jobs after filters: {len(work):,}")


# =========================
# SHOW SKILLS
# =========================
if resume_file and resume_text.strip():
    st.success("Resume loaded successfully ✅")
elif resume_file and not resume_text.strip():
    st.warning("Resume uploaded, but no text could be extracted. Try DOCX or TXT if the PDF is scanned.")

if target_industry_ui != "All":
    st.caption(f"Target industry selected: {target_industry_ui}")

if user_skills:
    st.markdown('<div class="section-title">✅ Skills Detected / Provided</div>', unsafe_allow_html=True)
    st.write(", ".join(user_skills[:SHOW_TOP_SKILLS]))
    with st.expander("Show all detected skills"):
        st.write(", ".join(user_skills))
else:
    st.info("Upload a resume, type skills, or use the target role to get job recommendations.")


# =========================
# MATCHING
# =========================
st.markdown('<div class="section-title">🎯 Job Recommendations</div>', unsafe_allow_html=True)
run = st.button("Find Matching Jobs", type="primary")

if run:
    if len(work) == 0:
        st.warning("No jobs match the selected filters. Try broader options such as All for work mode, career level, or industry.")
        st.stop()

    if not user_skills and not resume_text.strip() and not desired_role.strip():
        st.warning("Please upload a resume, type skills, or enter a target role first.")
        st.stop()

    vec, X_jobs = fit_tfidf(work["job_text"].fillna("").astype(str).tolist())

    if resume_text.strip():
        query_text = f"{desired_role} {resume_text}".strip()
    elif manual_skills.strip():
        query_text = f"{desired_role} {manual_skills}".strip()
    else:
        query_text = desired_role

    X_query = vec.transform([query_text])
    nlp_scores = cosine_similarity(X_query, X_jobs).flatten()

    skill_scores = []
    matched_all = []
    missing_all = []
    extra_all = []
    role_bonus = []
    expanded_job_skills_all = []

    desired_role_lower = desired_role.strip().lower()

    for _, row in work.iterrows():
        js = row["job_skills_list"] if isinstance(row["job_skills_list"], list) else []

        if len(js) < 8:
            expanded_js = extract_skills_from_text(
                f"{row.get('job_title', '')} {row.get('job_description', '')}",
                selected_industry="All"
            )
            js = sorted(set(js + expanded_js))

        desc_text = str(row.get("job_description", "")).lower()
        for cskill in COMMON_JOB_SKILLS:
            if cskill in desc_text and cskill not in js:
                js.append(cskill)

        req_phrases = extract_requirement_phrases(desc_text)
        for rp in req_phrases:
            if rp not in js and looks_like_skill(rp):
                js.append(rp)

        js = sorted(set(js))

        s, matched, missing, extra_user = skill_match(user_skills, js)

        title_text = str(row.get("job_title", "")).lower()

        bonus = 0.0
        if desired_role_lower:
            if desired_role_lower in title_text:
                bonus = 1.0
            elif desired_role_lower in desc_text:
                bonus = 0.7
            else:
                role_words = [w for w in desired_role_lower.split() if len(w) > 2]
                overlap = sum(1 for w in role_words if w in title_text or w in desc_text)
                if role_words:
                    bonus = min(overlap / len(role_words), 0.5)

        skill_scores.append(s)
        matched_all.append(matched)
        missing_all.append(missing)
        extra_all.append(extra_user)
        role_bonus.append(bonus)
        expanded_job_skills_all.append(js)

    results = work.copy()
    results["nlp_score"] = nlp_scores
    results["skill_score"] = skill_scores
    results["role_bonus"] = role_bonus
    results["expanded_job_skills"] = expanded_job_skills_all
    results["final_score"] = (
        (nlp_weight * results["nlp_score"]) +
        (skill_weight * results["skill_score"]) +
        (role_weight * results["role_bonus"])
    ).clip(upper=1.0)

    results["matched_skills"] = matched_all
    results["missing_skills"] = missing_all
    results["extra_user_skills"] = extra_all
    results = results.sort_values("final_score", ascending=False).head(top_k)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-title">Jobs searched</div>
          <div class="metric-value">{len(work):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        best_match = f"{results['final_score'].max() * 100:.1f}%" if len(results) else "0.0%"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-title">Best overall match</div>
          <div class="metric-value">{best_match}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-title">Skills used</div>
          <div class="metric-value">{len(user_skills):,}</div>
        </div>
        """, unsafe_allow_html=True)

    # Score meaning box must be OUTSIDE the metric cards
    st.markdown("""
    <div style="
    background:#f8fafc;
    border:1px solid #e5e7eb;
    padding:16px 18px;
    border-radius:14px;
    margin-top:10px;
    margin-bottom:18px;
    ">
    <h4 style="margin:0 0 10px 0;">ℹ️ Score Meaning</h4>

    <p><b>Overall Match</b> – Final combined score calculated using:</p>
    <ul>
    <li>Resume similarity</li>
    <li>Skill match</li>
    <li>Role relevance</li>
    </ul>

    <p><b>Resume Similarity</b> – How similar your resume text is to the job description.</p>
    <p><b>Skill Match</b> – Percentage of job skills you already have.</p>
    <p><b>Role Relevance</b> – How closely the job title matches the role you searched.</p>

    <p><b>Guide:</b></p>
    <ul>
    <li>80%+ → Excellent match</li>
    <li>60–79% → Good match</li>
    <li>40–59% → Moderate match</li>
    <li>Below 40% → Low match</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    download_cols = [
        "job_title", "company", "job_location", "work_mode", "industry",
        "job_type", "job_level_norm", "job_skills", "final_score",
        "nlp_score", "skill_score", "role_bonus", "job_link"
    ]
    csv_bytes = results[download_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Results (CSV)",
        data=csv_bytes,
        file_name="job_recommendations.csv",
        mime="text/csv"
    )

    st.divider()

    st.markdown('<div class="section-title">🧠 Resume Improvement Suggestions</div>', unsafe_allow_html=True)
    top_missing = []
    for ms in results["missing_skills"].tolist():
        top_missing.extend(ms)

    missing_counts = pd.Series(top_missing).value_counts().head(10) if len(top_missing) > 0 else pd.Series(dtype=int)

    if len(missing_counts):
        st.write("Based on the top matched jobs, consider adding or learning these skills:")
        st.write(", ".join(missing_counts.index.tolist()))
    else:
        st.success("No major missing skills detected for the top results.")

    # Top 5 skills to learn
    if len(missing_counts):
        st.markdown("### ⭐ Top 5 Skills You Should Learn Next")
        top5 = missing_counts.head(5).index.tolist()
        for i, skill in enumerate(top5, 1):
            st.write(f"{i}. {skill}")

    # Suggested career paths
    st.markdown("### 🧠 Suggested Career Paths")
    career_map = {
        "data analyst": ["business analyst", "bi analyst", "reporting analyst", "data visualization analyst"],
        "business analyst": ["operations analyst", "product analyst", "project coordinator"],
        "accountant": ["financial analyst", "audit analyst", "budget analyst"],
        "nurse practitioner": ["clinical specialist", "care coordinator", "healthcare administrator"],
        "marketing analyst": ["digital marketing analyst", "campaign analyst", "market research analyst"]
    }

    role_key = desired_role.strip().lower()
    suggested_roles = career_map.get(role_key, [])

    if suggested_roles:
        st.write("Based on your selected role, you may also explore:")
        st.write(", ".join(suggested_roles))
    else:
        st.write("Try related analyst, coordinator, specialist, or manager roles based on your skills.")

    # Salary insight
    st.markdown("### 💰 Salary Insight")
    if "salary_min" in results.columns and "salary_max" in results.columns:
        salary_df = results[["salary_min", "salary_max"]].dropna()
        if not salary_df.empty:
            avg_min = salary_df["salary_min"].mean()
            avg_max = salary_df["salary_max"].mean()
            st.write(f"Estimated salary range from matched jobs: ${avg_min:,.0f} - ${avg_max:,.0f}")
        else:
            st.write("Salary data is not available for the current matched jobs.")
    else:
        st.write("Salary data is not available for the current matched jobs.")

    st.divider()

    st.markdown('<div class="section-title">📊 Job Trends (Top Results)</div>', unsafe_allow_html=True)
    tc1, tc2 = st.columns(2)

    with tc1:
        skill_bag = []
        for js in results["missing_skills"].tolist():
            if isinstance(js, list):
                skill_bag.extend([s for s in js if looks_like_skill(s)])
        if len(skill_bag) == 0:
            for js in results["expanded_job_skills"].tolist():
                if isinstance(js, list):
                    skill_bag.extend([s for s in js if looks_like_skill(s)])

        top_sk = pd.Series(skill_bag).value_counts().head(10) if len(skill_bag) > 0 else pd.Series(dtype=int)
        if len(top_sk):
            st.caption("Top skills in matched jobs")
            st.bar_chart(top_sk)
        else:
            st.info("No skill trends available.")

    with tc2:
        wm = results["work_mode"].value_counts()
        st.caption("Work mode distribution")
        st.bar_chart(wm)

    st.divider()

    for _, row in results.iterrows():
        matched = row["matched_skills"] or []
        missing = row["missing_skills"] or []
        extra_user = row["extra_user_skills"] or []

        left_html = f"""
        <div class="job-card">
            <div class="job-title">{row['job_title']}</div>
            <div class="job-meta"><strong>Company:</strong> {row['company']}</div>
            <div class="job-meta"><strong>Location:</strong> {row['job_location']}</div>
            <div class="job-meta"><span class="badge badge-blue">{row['work_mode']}</span>
            <span class="badge badge-purple">{row['industry']}</span>
            <span class="badge badge-orange">{row['job_level_norm']}</span></div>
            <div class="small-muted">Type / Level: {row['job_type']} / {row['job_level_norm']}</div>
        </div>
        """
        st.markdown(left_html, unsafe_allow_html=True)

        c_left, c_right = st.columns([2.15, 1])

        with c_left:
            st.write("✅ **Matched skills for this job:**", ", ".join(matched[:10]) if matched else "—")
            st.write("❌ **Missing skills for this job:**", ", ".join(missing[:10]) if missing else "—")
            if extra_user:
                st.write("➕ **Additional skills you have:**", ", ".join(extra_user[:8]))

            why = []
            if row["nlp_score"] >= 0.15:
                why.append("Strong resume and description similarity")
            else:
                why.append("Lower resume and description similarity")

            if row["skill_score"] >= 0.35:
                why.append("Good job-skill coverage")
            else:
                why.append("Job-skill coverage is low")

            if row["role_bonus"] >= 0.7:
                why.append("Role title is highly relevant")
            elif row["role_bonus"] > 0:
                why.append("Role is partially relevant")

            st.info("**Match explanation:** " + " • ".join(why))

            with st.expander("View job details"):
                st.write("**Original Extracted Job Skills:**")
                st.write(row["job_skills"] if row["job_skills"] else "Not available")

                st.write("**Expanded Job Skills Used for Matching:**")
                st.write(", ".join(row["expanded_job_skills"]) if row["expanded_job_skills"] else "Not available")

                st.write("**Job Description (preview):**")
                desc = row["job_description"]
                st.write(desc[:2200] + ("..." if len(desc) > 2200 else ""))

                if isinstance(row.get("job_link", ""), str) and row["job_link"].strip():
                    st.link_button("Open Job Posting", row["job_link"])

        with c_right:
            st.metric("Overall Match", f"{row['final_score'] * 100:.1f}%")
            st.metric("Resume Similarity", f"{row['nlp_score'] * 100:.1f}%")
            st.metric("Skill Match", f"{row['skill_score'] * 100:.1f}%")
            st.metric("Role Relevance", f"{row['role_bonus'] * 100:.1f}%")

            with st.expander("What do these scores mean?"):
                st.write("**Overall Match**: Final combined score using resume similarity, skill match, and role relevance.")
                st.write("**Resume Similarity**: How close your resume text is to the job description text.")
                st.write("**Skill Match**: How many extracted job skills are already in your resume or entered skills.")
                st.write("**Role Relevance**: How closely your target role matches the job title or description.")