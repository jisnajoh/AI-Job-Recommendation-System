import re
import io
import math
from urllib.parse import urlparse

import requests
import pandas as pd
import streamlit as st
import pdfplumber
import docx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Job Recommendation System",
    page_icon="💼",
    layout="wide"
)

# =========================================================
# CONFIG
# =========================================================
MAX_ADZUNA_PAGES = 12
RESULTS_PER_PAGE_API = 50
DEFAULT_PAGE_SIZE = 10
SHOW_TOP_SKILLS = 15

APP_ID = st.secrets.get("ADZUNA_APP_ID", None)
APP_KEY = st.secrets.get("ADZUNA_APP_KEY", None)

USA_PATTERNS = r"united states|usa|us"

BAD_KEYWORDS = {
    "staffing", "recruiter", "recruitment", "placement", "talent solutions",
    "training program", "bench sales", "consultancy", "corp to corp", "c2c",
    "w2 only", "no job", "marketing partner", "immediate joiners",
    "third party", "implementation partner"
}

BAD_TITLE_KEYWORDS = {
    "training", "program", "bench", "walk-in", "walk in"
}

BAD_DOMAINS = {
    "monster.com",
    "ziprecruiter.com",
    "dice.com",
    "careerbuilder.com",
    "jobrapido.com",
    "lensa.com",
    "talent.com",
    "jooble.org",
    "learn4good.com",
    "grabjobs.co"
}

COMMON_JOB_SKILLS = {
    "communication", "reporting", "analysis", "documentation",
    "problem solving", "excel", "sql", "power bi",
    "tableau", "customer service", "project management",
    "dashboards", "data visualization", "leadership",
    "business analysis", "stakeholder management",
    "python", "java", "aws", "pandas", "numpy", "statistics",
    "machine learning", "nlp", "scikit-learn", "data analysis",
    "data analytics", "mysql", "postgresql", "sqlite",
    "html", "css", "javascript", "flask", "django",
    "cloud computing", "api", "git", "docker", "linux",
    "financial analysis", "accounting", "quickbooks",
    "ehr", "emr", "patient care", "powerpoint",
    "snowflake", "dbt", "looker", "sas", "r", "databricks",
    "bigquery", "redshift", "etl", "elt", "data warehousing",
    "data warehouse", "ssis", "ssrs", "ssms", "ssas",
    "kpi", "data modeling", "xlookup", "vlookup", "pivot tables",
    "excel pivot tables", "ad hoc reporting", "data governance",
    "forecasting", "a/b testing", "ab testing", "google sheets",
    "oracle", "sap", "erp", "crm", "salesforce", "workday",
    "epic", "cerner", "hipaa", "rest api", "streamlit",
    "jupyter", "mongodb", "spark", "hadoop", "airflow",
    "azure", "gcp", "tensorflow", "pytorch", "scala",
    "alteryx", "knime", "qlik", "qlik sense", "microstrategy",
    "informatica", "sap hana", "report development",
    "dashboard development", "structured data", "unstructured data",
    "data profiling", "data quality", "data integration",
    "data pipelines", "predictive modeling", "reporting tools",
    "visualization tools", "business intelligence"
}

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
.hero {
    padding: 1.2rem 1.3rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #eff6ff 0%, #f5f3ff 50%, #ecfeff 100%);
    border: 1px solid #e5e7eb;
    margin-bottom: 1rem;
}
.hero h1 {
    margin: 0;
    font-size: 2rem;
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
    font-size: 1.75rem;
    font-weight: 700;
    color: #111827;
}
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 0.25rem;
    margin-bottom: 0.8rem;
}
.job-list-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 0.85rem 0.95rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.job-detail-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 3px 12px rgba(0,0,0,0.03);
}
.job-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.3rem;
}
.job-detail-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.4rem;
}
.job-meta {
    color: #374151;
    font-size: 0.94rem;
    margin-bottom: 0.35rem;
}
.small-muted {
    color: #6b7280;
    font-size: 0.9rem;
}
.badge {
    display: inline-block;
    padding: 0.22rem 0.62rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.3rem;
    margin-bottom: 0.3rem;
}
.badge-blue { background: #dbeafe; color: #1d4ed8; }
.badge-green { background: #dcfce7; color: #166534; }
.badge-purple { background: #ede9fe; color: #6d28d9; }
.badge-orange { background: #ffedd5; color: #c2410c; }
.badge-pink { background: #fce7f3; color: #be185d; }
.score-box {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 0.8rem 0.9rem;
    margin-bottom: 0.7rem;
}
.saved-pill {
    display: inline-block;
    background: #fef3c7;
    color: #92400e;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>💼 AI Job Recommendation System</h1>
  <p>Live job recommendations using updated API postings, skill matching, role relevance, score explanations, saved jobs, and paginated results.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SKILL ONTOLOGY
# =========================================================
SKILL_ONTOLOGY = {
    "Data & IT": {
        "python", "sql", "excel", "power bi", "tableau", "pandas", "numpy",
        "scikit-learn", "machine learning", "deep learning", "nlp",
        "data analysis", "data analytics", "data visualization", "statistics",
        "aws", "azure", "gcp", "cloud computing", "flask", "django",
        "html", "css", "javascript", "java", "git", "docker", "linux",
        "api", "rest api", "mongodb", "mysql", "postgresql", "sqlite",
        "spark", "hadoop", "etl", "business intelligence", "reporting",
        "dashboards", "streamlit", "jupyter", "data cleaning", "database",
        "snowflake", "dbt", "looker", "sas", "r", "databricks",
        "bigquery", "redshift", "data warehousing", "data warehouse",
        "data modeling", "ssis", "ssrs", "ssms", "ssas", "airflow",
        "tensorflow", "pytorch", "alteryx", "knime", "qlik", "qlik sense",
        "microstrategy", "informatica", "data pipelines", "data quality",
        "structured data", "unstructured data", "report development",
        "dashboard development"
    },
    "Healthcare": {
        "patient care", "clinical assessment", "triage", "diagnosis",
        "treatment planning", "care coordination", "patient education",
        "medical records", "ehr", "emr", "epic", "cerner", "hipaa",
        "infection control", "telehealth", "case management", "medical terminology",
        "registered nurse", "nurse practitioner"
    },
    "Accounting & Finance": {
        "accounting", "bookkeeping", "financial reporting", "forecasting",
        "budgeting", "audit", "auditing", "gaap", "accounts payable",
        "accounts receivable", "bank reconciliation", "payroll",
        "financial analysis", "tax preparation", "quickbooks", "sap",
        "oracle", "erp", "general ledger", "cash flow", "reconciliation", "excel"
    },
    "Business & Operations": {
        "business analysis", "stakeholder management", "project management",
        "requirements gathering", "process improvement", "operations management",
        "reporting", "documentation", "workflow optimization", "crm",
        "salesforce", "leadership", "communication", "problem solving",
        "kpi tracking", "vendor management", "supply chain", "customer service"
    },
    "Marketing": {
        "seo", "sem", "google analytics", "social media", "campaign management",
        "content marketing", "content creation", "email marketing", "brand strategy",
        "market research", "advertising", "copywriting", "crm",
        "digital marketing", "lead generation", "analytics"
    },
    "Human Resources": {
        "recruitment", "talent acquisition", "employee relations", "onboarding",
        "benefits administration", "payroll", "performance management",
        "hris", "workday", "policy compliance", "training", "interviewing",
        "candidate screening", "staffing", "human resources"
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
    "ability to", "must have", "should have", "plus", "etc", "using",
    "including", "preferred qualifications", "basic", "advanced", "benefits",
    "salary", "pay range", "responsibility"
}

SKILL_ALIASES = {
    "powerbi": "power bi",
    "microsoft power bi": "power bi",
    "ms power bi": "power bi",
    "structured query language": "sql",
    "sql server": "sql",
    "ms sql": "sql",
    "aws s3": "aws",
    "amazon web services": "aws",
    "scikit learn": "scikit-learn",
    "electronic health record": "ehr",
    "electronic medical record": "emr",
    "search engine optimization": "seo",
    "search engine marketing": "sem",
    "customer relationship management": "crm",
    "human resource information system": "hris",
    "ms excel": "excel",
    "microsoft excel": "excel",
    "py": "python",
    "js": "javascript",
    "restful api": "rest api",
    "application programming interface": "api",
    "bi": "business intelligence",
    "bi tools": "business intelligence",
    "business intelligence tools": "business intelligence",
    "google bigquery": "bigquery",
    "amazon redshift": "redshift",
    "g sheets": "google sheets",
    "a/b tests": "a/b testing",
    "ab tests": "ab testing",
    "alteryx designer": "alteryx",
    "knime analytics platform": "knime",
    "dashboarding": "dashboard development",
    "reporting tools": "reporting"
}

INDUSTRY_RULES = {
    "Healthcare": [
        "hospital", "clinic", "patient", "nurse", "physician", "medical",
        "health", "ehr", "emr", "hipaa", "clinical"
    ],
    "Accounting & Finance": [
        "bank", "finance", "financial", "accounting", "audit", "tax",
        "payroll", "gaap", "bookkeeping", "accounts payable", "quickbooks"
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
        "api", "database", "power bi", "tableau", "java", "javascript",
        "alteryx", "knime", "qlik", "informatica"
    ],
    "Education": [
        "university", "school", "teacher", "student", "education", "curriculum"
    ]
}

# =========================================================
# HELPERS
# =========================================================
def normalize_text(x) -> str:
    x = "" if pd.isna(x) else str(x)
    x = x.replace("\xa0", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def clean_for_matching(text: str) -> str:
    text = normalize_text(text).lower()
    text = text.replace("&", " and ")
    text = text.replace("/", " / ")
    text = re.sub(r"[\|\(\)\[\]\{\}:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    parts = re.split(r",|;|\|", s)
    return [p.strip().lower() for p in parts if p.strip()]


def looks_like_skill(s: str) -> bool:
    s = (s or "").strip().lower()
    if not (2 <= len(s) <= 60):
        return False
    if s in GENERIC_NOT_SKILLS:
        return False
    if len(s.split()) > 6:
        return False
    if not re.search(r"[a-z0-9\+#]", s):
        return False
    if s.isdigit():
        return False
    return True


def get_skill_dictionary(selected_industry="All"):
    if selected_industry == "All":
        combined = set()
        for skills in SKILL_ONTOLOGY.values():
            combined.update(skills)
        combined.update(COMMON_JOB_SKILLS)
        return combined
    return SKILL_ONTOLOGY.get(selected_industry, set()).union(COMMON_JOB_SKILLS)


def apply_skill_aliases(skills):
    normalized = set()
    for s in skills:
        s = str(s).strip().lower()
        if not s:
            continue
        normalized.add(SKILL_ALIASES.get(s, s))
    return sorted([s for s in normalized if looks_like_skill(s)])


def infer_target_industry_from_user_text(user_text: str):
    t = clean_for_matching(user_text)
    for ind, kws in INDUSTRY_RULES.items():
        if any(k in t for k in kws):
            return ind
    return None


def extract_requirement_phrases(text: str):
    text = clean_for_matching(text)
    candidates = set()

    patterns = [
        r"(?:experience with|proficiency in|knowledge of|expertise in|skilled in|hands[- ]on experience with|experience using|working knowledge of|familiarity with)\s+([a-zA-Z0-9\+\#\/\-\s,]{3,300})",
        r"(?:required skills?|preferred skills?|basic qualifications?|preferred qualifications?|qualifications?|requirements?|technical skills?)[:\s]+([a-zA-Z0-9\+\#\/\-\s,]{3,350})",
        r"(?:must have|should have|nice to have)\s+([a-zA-Z0-9\+\#\/\-\s,]{3,250})",
        r"(?:tools?|technologies?|stack|platforms?)[:\s]+([a-zA-Z0-9\+\#\/\-\s,]{3,250})",
        r"(?:using|use of|build using|develop using|reports using|apps using)\s+([a-zA-Z0-9\+\#\/\-\s,]{3,250})"
    ]

    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        for m in matches:
            parts = re.split(r",|;|/|\band\b|\bor\b", m)
            for p in parts:
                p = p.strip(" .:-").lower()
                if looks_like_skill(p):
                    candidates.add(SKILL_ALIASES.get(p, p))

    return sorted(candidates)


def extract_bullet_like_terms(text: str):
    raw_text = str(text)
    raw_text = raw_text.replace("•", "\n").replace("▪", "\n").replace("·", "\n").replace("–", "\n").replace("*", "\n")

    parts = re.split(r"\n|\.", raw_text)
    out = set()

    skill_dict = get_skill_dictionary("All")

    for part in parts:
        piece = clean_for_matching(part)
        if not piece:
            continue

        for sk in sorted(skill_dict, key=len, reverse=True):
            pattern = r"(?<!\w)" + re.escape(sk) + r"(?!\w)"
            if re.search(pattern, piece):
                out.add(sk)

        for alias, canon in SKILL_ALIASES.items():
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            if re.search(pattern, piece):
                out.add(canon)

        phrase_parts = re.split(r",|;|/|\band\b|\bor\b", piece)
        for p in phrase_parts:
            p = p.strip(" .:-").lower()
            if looks_like_skill(p):
                out.add(SKILL_ALIASES.get(p, p))

    return sorted(out)


def extract_skills_from_text(text_in: str, selected_industry="All"):
    text = clean_for_matching(text_in)
    found = set()
    skill_dict = get_skill_dictionary(selected_industry if selected_industry != "All" else "All")

    for sk in sorted(skill_dict, key=len, reverse=True):
        pattern = r"(?<!\w)" + re.escape(sk) + r"(?!\w)"
        if re.search(pattern, text):
            found.add(sk)

    for alias, canon in SKILL_ALIASES.items():
        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        if re.search(pattern, text):
            found.add(canon)

    for p in extract_requirement_phrases(text_in):
        if looks_like_skill(p):
            found.add(p)

    for b in extract_bullet_like_terms(text_in):
        if looks_like_skill(b):
            found.add(b)

    return apply_skill_aliases(found)


def extract_exact_job_skills(job_title: str, job_description: str):
    text_raw = f"{job_title}\n{job_description}"
    text = clean_for_matching(text_raw)
    found = set()

    skill_dict = get_skill_dictionary("All")

    # 1. Direct dictionary matches
    for sk in sorted(skill_dict, key=len, reverse=True):
        pattern = r"(?<!\w)" + re.escape(sk) + r"(?!\w)"
        if re.search(pattern, text):
            found.add(sk)

    # 2. Alias matches
    for alias, canon in SKILL_ALIASES.items():
        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        if re.search(pattern, text):
            found.add(canon)

    # 3. Requirement phrase extraction
    found.update(extract_requirement_phrases(text_raw))

    # 4. Bullet / line scanning
    found.update(extract_bullet_like_terms(text_raw))

    # 5. Direct scan for "using X, Y, Z"
    using_matches = re.findall(
        r"(?:using|use of|experience with|proficient in)\s+([a-zA-Z0-9\+\#\/\-\s,]{3,300})",
        text,
        flags=re.IGNORECASE
    )
    for match in using_matches:
        parts = re.split(r",|;|/|\band\b|\bor\b", match)
        for p in parts:
            p = p.strip(" .:-").lower()
            if looks_like_skill(p):
                found.add(SKILL_ALIASES.get(p, p))

    # 6. Fallback broader extraction
    if not found:
        found.update(extract_skills_from_text(text_raw, "All"))

    return sorted(set([s for s in found if looks_like_skill(s)]))


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


def extract_resume_sections(resume_text: str):
    text = normalize_text(resume_text)
    low = text.lower()

    skills_text = ""
    experience_text = ""
    summary_text = ""

    skills_match = re.search(
        r"(skills|technical skills|core skills)(.*?)(experience|projects|education|certifications|$)",
        low,
        flags=re.IGNORECASE | re.DOTALL
    )
    if skills_match:
        skills_text = skills_match.group(2)

    exp_match = re.search(
        r"(experience|work experience|projects|project experience)(.*?)(education|certifications|skills|$)",
        low,
        flags=re.IGNORECASE | re.DOTALL
    )
    if exp_match:
        experience_text = exp_match.group(2)

    summary_match = re.search(
        r"(summary|profile|objective)(.*?)(skills|experience|education|$)",
        low,
        flags=re.IGNORECASE | re.DOTALL
    )
    if summary_match:
        summary_text = summary_match.group(2)

    return {
        "skills_text": normalize_text(skills_text),
        "experience_text": normalize_text(experience_text),
        "summary_text": normalize_text(summary_text)
    }


def build_focus_text(desired_role: str, user_skills: list, resume_sections: dict):
    skills_part = " ".join(user_skills[:40])
    experience_part = resume_sections.get("experience_text", "")[:1200]
    summary_part = resume_sections.get("summary_text", "")[:500]
    return normalize_text(f"{desired_role} {skills_part} {summary_part} {experience_part}")


def build_job_text(df):
    return (
        df["job_title"].fillna("") + " | " +
        df["job_description"].fillna("") + " | " +
        df["job_skills"].fillna("") + " | " +
        df["company"].fillna("") + " | " +
        df["job_location"].fillna("")
    )


def deduplicate_jobs(df):
    if df.empty:
        return df

    temp = df.copy()
    temp["job_title_clean"] = temp["job_title"].fillna("").astype(str).str.lower().str.strip()
    temp["company_clean"] = temp["company"].fillna("").astype(str).str.lower().str.strip()
    temp["location_clean"] = temp["job_location"].fillna("").astype(str).str.lower().str.strip()

    temp["job_title_clean"] = temp["job_title_clean"].str.replace(r"\s+", " ", regex=True)
    temp["company_clean"] = temp["company_clean"].str.replace(r"\s+", " ", regex=True)
    temp["location_clean"] = temp["location_clean"].str.replace(r"\s+", " ", regex=True)

    if "job_link" in temp.columns:
        temp["job_link_clean"] = temp["job_link"].fillna("").astype(str).str.lower().str.strip()
        temp = temp.drop_duplicates(subset=["job_link_clean"], keep="first")

    temp = temp.drop_duplicates(
        subset=["job_title_clean", "company_clean", "location_clean"],
        keep="first"
    )

    return temp.drop(
        columns=["job_title_clean", "company_clean", "location_clean", "job_link_clean"],
        errors="ignore"
    ).reset_index(drop=True)


def infer_industry(row) -> str:
    text = clean_for_matching(f"{row.get('job_title', '')} {row.get('job_description', '')}")
    for ind, kws in INDUSTRY_RULES.items():
        if any(k in text for k in kws):
            return ind
    return "Other"


def infer_work_mode(row) -> str:
    text = clean_for_matching(
        f"{row.get('job_type', '')} {row.get('job_description', '')} {row.get('job_title', '')}"
    )
    if "remote" in text or "work from home" in text or "wfh" in text:
        return "Remote"
    if "hybrid" in text:
        return "Hybrid"
    return "Onsite"


def normalize_level(x: str, title: str = "", desc: str = "") -> str:
    t = clean_for_matching(f"{x} {title} {desc}")
    if "intern" in t or "internship" in t:
        return "Intern"
    if "entry" in t or "junior" in t or "jr " in t or "recent graduate" in t or "new grad" in t:
        return "Entry"
    if "senior" in t or "lead" in t or "manager" in t or "principal" in t or "director" in t:
        return "Senior"
    return "Mid"


def normalize_job_type(x: str, title: str = "", desc: str = "") -> str:
    t = clean_for_matching(f"{x} {title} {desc}")
    if "intern" in t or "internship" in t:
        return "Internship"
    if "part-time" in t or "part time" in t:
        return "Part-time"
    if "contract" in t or "temporary" in t:
        return "Contract"
    if "full-time" in t or "full time" in t:
        return "Full-time"
    return "Other"


def is_direct_company_job(link, description, title):
    link = normalize_text(link).lower()
    description = normalize_text(description).lower()
    title = normalize_text(title).lower()

    full_text = f"{title} {description}"

    if not link:
        return False

    domain = urlparse(link).netloc.replace("www.", "").lower()

    if any(bad_domain in domain for bad_domain in BAD_DOMAINS):
        return False

    if any(word in full_text for word in BAD_KEYWORDS):
        return False

    if any(word in title for word in BAD_TITLE_KEYWORDS):
        return False

    return True


def skill_match(user_skills, job_skills_list):
    us = set(apply_skill_aliases([s.lower().strip() for s in user_skills if str(s).strip()]))
    js = set(apply_skill_aliases([s.lower().strip() for s in job_skills_list if str(s).strip()]))

    matched = sorted(list(us.intersection(js)))
    missing = sorted(list(js - us))
    extra_user_skills = sorted([
        s for s in (us - js)
        if s not in {"communication", "leadership", "documentation", "reporting"}
    ])

    score = 0.0 if len(js) == 0 else (len(matched) / len(js))
    return score, matched, missing, extra_user_skills


def build_match_reason(row):
    matched_count = len(row["matched_skills"]) if isinstance(row["matched_skills"], list) else 0
    total_job_skills = len(row["expanded_job_skills"]) if isinstance(row["expanded_job_skills"], list) else 0

    reasons = []
    if total_job_skills > 0:
        reasons.append(f"Matched {matched_count} of {total_job_skills} extracted job skills")
    else:
        reasons.append("Limited skill extraction")

    if row["role_bonus"] >= 0.85:
        reasons.append("Strong target role match")
    elif row["role_bonus"] >= 0.45:
        reasons.append("Moderate target role match")
    else:
        reasons.append("Weak target role match")

    if row["nlp_score"] >= 0.22:
        reasons.append("Good profile similarity")
    elif row["nlp_score"] >= 0.10:
        reasons.append("Moderate profile similarity")
    else:
        reasons.append("Low profile similarity")

    return " | ".join(reasons)


def build_page_numbers(current_page, total_pages, max_visible=7):
    if total_pages <= max_visible:
        return list(range(1, total_pages + 1))

    pages = {1, total_pages, current_page}
    for p in range(current_page - 1, current_page + 2):
        if 1 <= p <= total_pages:
            pages.add(p)

    return sorted(pages)


@st.cache_data(show_spinner=False)
def fit_tfidf(job_texts):
    vec = TfidfVectorizer(stop_words="english", max_features=40000, ngram_range=(1, 2))
    X = vec.fit_transform(job_texts)
    return vec, X


# =========================================================
# LIVE JOB API
# =========================================================
def fetch_live_jobs(query, max_pages=MAX_ADZUNA_PAGES):
    jobs = []

    if not APP_ID or not APP_KEY:
        st.error("Missing Adzuna API credentials. Add ADZUNA_APP_ID and ADZUNA_APP_KEY in .streamlit/secrets.toml")
        return pd.DataFrame()

    for page in range(1, max_pages + 1):
        url = f"https://api.adzuna.com/v1/api/jobs/us/search/{page}"
        params = {
            "app_id": APP_ID,
            "app_key": APP_KEY,
            "results_per_page": RESULTS_PER_PAGE_API,
            "what": query,
            "content-type": "application/json"
        }

        try:
            response = requests.get(url, params=params, timeout=25)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            st.warning(f"Could not fetch page {page}: {e}")
            continue

        page_results = data.get("results", [])
        if not page_results:
            break

        for job in page_results:
            company_name = ""
            if isinstance(job.get("company"), dict):
                company_name = job["company"].get("display_name", "")

            location_name = ""
            if isinstance(job.get("location"), dict):
                location_name = job["location"].get("display_name", "")

            job_title = normalize_text(job.get("title", ""))
            job_description = normalize_text(job.get("description", ""))
            job_link = normalize_text(job.get("redirect_url", ""))

            extracted_job_skills = extract_exact_job_skills(job_title, job_description)

            contract_type = normalize_text(job.get("contract_type", ""))
            contract_time = normalize_text(job.get("contract_time", ""))
            job_type_text = " ".join([x for x in [contract_type, contract_time] if x]).strip()

            jobs.append({
                "job_title": job_title,
                "company": normalize_text(company_name),
                "job_location": normalize_text(location_name),
                "job_description": job_description,
                "job_link": job_link,
                "job_skills": ", ".join(extracted_job_skills),
                "job_skills_list": extracted_job_skills,
                "search_country": "United States",
                "job_level": "",
                "job_type": job_type_text,
                "salary_min": job.get("salary_min", None),
                "salary_max": job.get("salary_max", None),
                "source_page": page
            })

    df = pd.DataFrame(jobs)

    if df.empty:
        return df

    df = deduplicate_jobs(df)
    df["job_text"] = build_job_text(df)
    return df.reset_index(drop=True)


# =========================================================
# SESSION STATE
# =========================================================
if "results_all_ranked" not in st.session_state:
    st.session_state["results_all_ranked"] = None

if "page_no" not in st.session_state:
    st.session_state["page_no"] = 1

if "jobs_analyzed_count" not in st.session_state:
    st.session_state["jobs_analyzed_count"] = 0

if "selected_job_idx" not in st.session_state:
    st.session_state["selected_job_idx"] = 0

if "saved_jobs" not in st.session_state:
    st.session_state["saved_jobs"] = []

if "last_query_signature" not in st.session_state:
    st.session_state["last_query_signature"] = None

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 🔎 Search & Filters")
st.sidebar.success("Live Jobs API enabled")

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
    value="data analyst",
    placeholder="Example: data analyst, nurse practitioner, accountant"
)

manual_skills = st.sidebar.text_area(
    "✍️ Skills (comma-separated)",
    value="sql, python, excel, power bi",
    height=90
)

remove_low_quality = st.sidebar.checkbox("🧹 Remove low-quality / recruiter postings", value=True)
only_usa = st.sidebar.checkbox("🇺🇸 Only USA jobs", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚖️ Scoring Weights")

nlp_weight = st.sidebar.slider("Profile Similarity", 0.0, 1.0, 0.20, 0.05)
skill_weight = st.sidebar.slider("Required Skills Match", 0.0, 1.0, 0.50, 0.05)
role_weight = st.sidebar.slider("Target Role Match", 0.0, 1.0, 0.30, 0.05)

total_weight = nlp_weight + skill_weight + role_weight
if total_weight == 0:
    nlp_weight, skill_weight, role_weight = 0.20, 0.50, 0.30
    total_weight = 1.0

nlp_weight /= total_weight
skill_weight /= total_weight
role_weight /= total_weight

st.sidebar.caption(
    f"Final score = {nlp_weight:.2f} × profile similarity + {skill_weight:.2f} × skills + {role_weight:.2f} × role match"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛 Result Settings")

display_page_size = st.sidebar.selectbox("Jobs per page", [10, 20, 25, 50], index=0)

sort_option = st.sidebar.selectbox(
    "Sort results by",
    [
        "Best Match",
        "Skills Match",
        "Role Match",
        "Newest Fetched",
        "Job Title (A-Z)",
        "Company (A-Z)"
    ]
)

work_mode_option = st.sidebar.selectbox("🏠 Work Mode", ["All", "Remote", "Hybrid", "Onsite"])
career_level_option = st.sidebar.selectbox("🎓 Career Level", ["All", "Intern", "Entry", "Mid", "Senior"])
employment_type_option = st.sidebar.selectbox("💼 Employment Type", ["All", "Full-time", "Part-time", "Internship", "Contract", "Other"])

resume_file = st.sidebar.file_uploader("📄 Upload Resume", type=["pdf", "docx", "txt"])

run_search = st.sidebar.button("Find Matching Jobs", type="primary")

# =========================================================
# INPUT PREP
# =========================================================
resume_text = read_resume(resume_file)
resume_sections = extract_resume_sections(resume_text) if resume_text.strip() else {
    "skills_text": "", "experience_text": "", "summary_text": ""
}

typed_skills = [s.strip().lower() for s in re.split(r",|;|\|", manual_skills) if s.strip()]
resume_skills = extract_skills_from_text(resume_text, target_industry_ui) if resume_text.strip() else []
manual_detected_skills = extract_skills_from_text(manual_skills, target_industry_ui) if manual_skills.strip() else []

if target_industry_ui == "All":
    resume_skills = sorted(set(resume_skills + extract_skills_from_text(resume_text, "All")))
    manual_detected_skills = sorted(set(manual_detected_skills + extract_skills_from_text(manual_skills, "All")))

user_skills = apply_skill_aliases(sorted(set(typed_skills + manual_detected_skills + resume_skills)))
focused_user_text = build_focus_text(desired_role, user_skills, resume_sections)

user_text_for_industry = f"{desired_role} {manual_skills} {resume_text}"
target_industry_from_text = infer_target_industry_from_user_text(user_text_for_industry)

# =========================================================
# TOP INFO
# =========================================================
if resume_file and resume_text.strip():
    st.success("Resume loaded successfully ✅")
elif resume_file and not resume_text.strip():
    st.warning("Resume uploaded, but text could not be extracted clearly. DOCX or TXT usually works better.")

if user_skills:
    st.markdown('<div class="section-title">✅ Detected Candidate Skills</div>', unsafe_allow_html=True)
    st.write(", ".join(user_skills[:SHOW_TOP_SKILLS]))
    with st.expander("Show all detected skills"):
        st.write(", ".join(user_skills))
else:
    st.info("Upload a resume, type skills, or enter a target role to improve recommendations.")

# =========================================================
# SKILL GAP ANALYSIS
# =========================================================
st.markdown('<div class="section-title">🔍 Skill Gap Analysis</div>', unsafe_allow_html=True)
st.markdown(
    "Paste a job posting below to see exactly which skills you have, which are missing, "
    "and get a clear gap analysis.",
)

job_posting_text = st.text_area(
    "Paste the job posting text here",
    height=200,
    placeholder=(
        "Paste the full job posting or just the requirements/qualifications section here...\n\n"
        "Example: We are looking for a Data Analyst with skills in Python, SQL, "
        "Power BI, Excel, Data Analysis, Machine Learning, and NLP..."
    ),
)

analyze_gap = st.button("Analyze Skill Gap", type="secondary")

if analyze_gap and job_posting_text.strip():
    job_posting_skills = extract_skills_from_text(job_posting_text, target_industry_ui)

    req_phrases = extract_requirement_phrases(job_posting_text)
    for rp in req_phrases:
        if rp not in job_posting_skills and looks_like_skill(rp):
            job_posting_skills.append(rp)

    bullet_terms = extract_bullet_like_terms(job_posting_text)
    for bt in bullet_terms:
        if bt not in job_posting_skills and looks_like_skill(bt):
            job_posting_skills.append(bt)

    job_posting_skills = sorted(set(apply_skill_aliases(job_posting_skills)))

    if not job_posting_skills:
        st.warning("No skills could be extracted from the job posting. Try pasting a longer description.")
    elif not user_skills:
        st.warning("Please type your skills or upload a resume first so we can compare.")
    else:
        gap_score, matched_skills, missing_skills, extra_skills = skill_match(
            user_skills, job_posting_skills
        )

        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-title">Job Posting Skills</div>
              <div class="metric-value">{len(job_posting_skills)}</div>
            </div>
            """, unsafe_allow_html=True)
        with g2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-title">Your Matching Skills</div>
              <div class="metric-value">{len(matched_skills)}</div>
            </div>
            """, unsafe_allow_html=True)
        with g3:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-title">Skills You're Missing</div>
              <div class="metric-value">{len(missing_skills)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg,
                #dcfce7 0%,
                #dcfce7 {gap_score * 100:.0f}%,
                #fecaca {gap_score * 100:.0f}%,
                #fecaca 100%);
            border-radius: 12px;
            padding: 14px 18px;
            margin: 12px 0 18px 0;
            border: 1px solid #e5e7eb;
            text-align: center;
            font-weight: 700;
            font-size: 1.1rem;
        ">
            Skill Match: {gap_score * 100:.0f}% — You have {len(matched_skills)} of {len(job_posting_skills)} required skills
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Skills Found in the Job Posting")
        user_skills_lower = [s.lower().strip() for s in user_skills]
        posting_badges = ""
        for sk in job_posting_skills:
            if sk in user_skills_lower:
                posting_badges += f'<span class="badge badge-green">{sk}</span>'
            else:
                posting_badges += f'<span class="badge badge-orange">{sk}</span>'
        st.markdown(posting_badges, unsafe_allow_html=True)
        st.caption("Green = you have it, Orange = missing")

        col_match, col_miss = st.columns(2)

        with col_match:
            st.markdown("#### Matched Skills")
            if matched_skills:
                for sk in matched_skills:
                    st.write(f"- {sk}")
            else:
                st.write("No matching skills found.")

        with col_miss:
            st.markdown("#### Missing Skills")
            if missing_skills:
                for sk in missing_skills:
                    st.write(f"- {sk}")
            else:
                st.success("You have all the required skills!")

        if extra_skills:
            st.markdown("#### Extra Skills You Have (not in this posting)")
            extra_badges = " ".join(
                [f'<span class="badge badge-blue">{sk}</span>' for sk in extra_skills]
            )
            st.markdown(extra_badges, unsafe_allow_html=True)
            st.caption(
                "These skills from your resume aren't listed in this job posting, "
                "but they may still add value to your application."
            )

        if missing_skills:
            st.markdown("#### Recommended Next Steps")
            st.write(
                "To improve your chances for this role, consider learning or "
                "highlighting these missing skills:"
            )
            for i, sk in enumerate(missing_skills[:5], 1):
                st.write(f"**{i}. {sk}**")
            if len(missing_skills) > 5:
                st.write(
                    f"...and {len(missing_skills) - 5} more. "
                    "See the full list above."
                )

elif analyze_gap and not job_posting_text.strip():
    st.warning("Please paste a job posting text above before analyzing.")

st.divider()

# =========================================================
# SEARCH EXECUTION
# =========================================================
query_signature = {
    "role": desired_role,
    "skills": manual_skills,
    "industry": target_industry_ui,
    "resume_present": bool(resume_text.strip()),
    "remove_low_quality": remove_low_quality,
    "only_usa": only_usa,
    "work_mode": work_mode_option,
    "career_level": career_level_option,
    "employment_type": employment_type_option,
    "weights": (nlp_weight, skill_weight, role_weight)
}

if run_search:
    if not desired_role.strip() and not user_skills and not resume_text.strip():
        st.warning("Please enter a target role, skills, or upload a resume first.")
        st.stop()

    with st.spinner("Fetching live jobs and ranking matches..."):
        query = desired_role.strip() if desired_role.strip() else "jobs"
        raw_df = fetch_live_jobs(query=query, max_pages=MAX_ADZUNA_PAGES)

        if raw_df.empty:
            st.warning("No jobs were fetched from the API.")
            st.stop()

        fetched_jobs_count = len(raw_df)
        work = raw_df.copy()

        if remove_low_quality:
            work = work[
                work.apply(
                    lambda row: is_direct_company_job(
                        row.get("job_link", ""),
                        row.get("job_description", ""),
                        row.get("job_title", "")
                    ),
                    axis=1
                )
            ].copy()

        cleaned_jobs_count = len(work)

        if work.empty:
            st.warning("All fetched jobs were removed by the cleaner. Try turning off the low-quality filter.")
            st.stop()

        work["work_mode"] = work.apply(infer_work_mode, axis=1)
        work["industry"] = work.apply(infer_industry, axis=1)
        work["job_level_norm"] = work.apply(
            lambda row: normalize_level(row.get("job_level", ""), row.get("job_title", ""), row.get("job_description", "")),
            axis=1
        )
        work["employment_type_norm"] = work.apply(
            lambda row: normalize_job_type(row.get("job_type", ""), row.get("job_title", ""), row.get("job_description", "")),
            axis=1
        )

        if only_usa:
            work = work[work["search_country"].astype(str).str.lower().str.contains(USA_PATTERNS, na=False)].copy()

        if work_mode_option != "All":
            work = work[work["work_mode"] == work_mode_option].copy()

        if career_level_option != "All":
            work = work[work["job_level_norm"] == career_level_option].copy()

        if employment_type_option != "All":
            work = work[work["employment_type_norm"] == employment_type_option].copy()

        if target_industry_ui != "All":
            work = work[work["industry"] == target_industry_ui].copy()
        elif target_industry_from_text:
            pass

        if work.empty:
            st.warning("No jobs match your current filters. Try broader filters.")
            st.stop()

        vec, X_jobs = fit_tfidf(work["job_text"].fillna("").astype(str).tolist())
        query_text = focused_user_text if focused_user_text.strip() else desired_role.strip()
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
            js = row["job_skills_list"] if isinstance(row["job_skills_list"], list) else parse_skill_list(row.get("job_skills", ""))
            expanded_js = extract_exact_job_skills(row.get("job_title", ""), row.get("job_description", ""))

            js = sorted(set(js + expanded_js))
            js = apply_skill_aliases(js)

            if not js:
                js = extract_skills_from_text(
                    f"{row.get('job_title', '')} {row.get('job_description', '')}",
                    "All"
                )

            s, matched, missing, extra_user = skill_match(user_skills, js)

            title_text = clean_for_matching(str(row.get("job_title", "")))
            desc_text = clean_for_matching(str(row.get("job_description", "")))

            bonus = 0.0
            if desired_role_lower:
                if desired_role_lower in title_text:
                    bonus = 1.0
                elif desired_role_lower in desc_text:
                    bonus = 0.75
                else:
                    role_words = [w for w in desired_role_lower.split() if len(w) > 2]
                    overlap = sum(1 for w in role_words if w in title_text or w in desc_text)
                    if role_words:
                        bonus = min(overlap / len(role_words), 0.55)

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
        results["match_reason"] = results.apply(build_match_reason, axis=1)

        results = deduplicate_jobs(results).reset_index(drop=True)
        results["fetch_order"] = range(1, len(results) + 1)

        if sort_option == "Best Match":
            results = results.sort_values("final_score", ascending=False).reset_index(drop=True)
        elif sort_option == "Skills Match":
            results = results.sort_values("skill_score", ascending=False).reset_index(drop=True)
        elif sort_option == "Role Match":
            results = results.sort_values("role_bonus", ascending=False).reset_index(drop=True)
        elif sort_option == "Newest Fetched":
            results = results.sort_values(["source_page", "fetch_order"], ascending=[False, True]).reset_index(drop=True)
        elif sort_option == "Job Title (A-Z)":
            results = results.sort_values("job_title", ascending=True).reset_index(drop=True)
        elif sort_option == "Company (A-Z)":
            results = results.sort_values("company", ascending=True).reset_index(drop=True)

        results.attrs["fetched_jobs_count"] = fetched_jobs_count
        results.attrs["cleaned_jobs_count"] = cleaned_jobs_count

        st.session_state["results_all_ranked"] = results
        st.session_state["jobs_analyzed_count"] = len(work)
        st.session_state["page_no"] = 1
        st.session_state["selected_job_idx"] = 0
        st.session_state["last_query_signature"] = query_signature

# =========================================================
# DISPLAY RESULTS
# =========================================================
results_all_ranked = st.session_state.get("results_all_ranked")

if results_all_ranked is not None and not results_all_ranked.empty:
    fetched_jobs_count = results_all_ranked.attrs.get("fetched_jobs_count", len(results_all_ranked))
    cleaned_jobs_count = results_all_ranked.attrs.get("cleaned_jobs_count", len(results_all_ranked))
    jobs_analyzed_count = st.session_state.get("jobs_analyzed_count", len(results_all_ranked))

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-title">Fetched jobs</div>
          <div class="metric-value">{fetched_jobs_count:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-title">Jobs after cleaning</div>
          <div class="metric-value">{cleaned_jobs_count:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-title">Filtered jobs analyzed</div>
          <div class="metric-value">{jobs_analyzed_count:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        best_match = f"{results_all_ranked['final_score'].max() * 100:.1f}%"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-title">Top recommendation score</div>
          <div class="metric-value">{best_match}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="score-box">
    <b>Score Meaning</b><br>
    Overall Match = Profile Similarity + Required Skills Match + Target Role Match.<br>
    Profile Similarity compares your focused role/skills/resume text with the job posting.<br>
    Required Skills Match checks how many extracted job skills you already have.<br>
    Target Role Match checks how closely the job title or description matches your selected role.
    </div>
    """, unsafe_allow_html=True)

    total_results = len(results_all_ranked)
    total_pages = max(1, math.ceil(total_results / display_page_size))

    if st.session_state["page_no"] > total_pages:
        st.session_state["page_no"] = total_pages

    current_page = st.session_state["page_no"]
    start_idx = (current_page - 1) * display_page_size
    end_idx = min(start_idx + display_page_size, total_results)

    page_df = results_all_ranked.iloc[start_idx:end_idx].reset_index(drop=True)

    st.markdown('<div class="section-title">📄 Results</div>', unsafe_allow_html=True)
    st.caption(f"Showing jobs {start_idx + 1} to {end_idx} of {total_results}")

    nav1, nav2, nav3 = st.columns([1, 4, 1])

    with nav1:
        if st.button("⬅ Prev", disabled=(current_page == 1), key="prev_page"):
            st.session_state["page_no"] = max(1, current_page - 1)
            st.session_state["selected_job_idx"] = 0
            st.rerun()

    with nav2:
        page_numbers = build_page_numbers(current_page, total_pages, max_visible=7)
        page_cols = st.columns(len(page_numbers))
        for i, p in enumerate(page_numbers):
            label = f"[{p}]" if p == current_page else str(p)
            if page_cols[i].button(label, key=f"page_btn_{p}"):
                st.session_state["page_no"] = p
                st.session_state["selected_job_idx"] = 0
                st.rerun()

    with nav3:
        if st.button("Next ➡", disabled=(current_page == total_pages), key="next_page"):
            st.session_state["page_no"] = min(total_pages, current_page + 1)
            st.session_state["selected_job_idx"] = 0
            st.rerun()

    left_col, right_col = st.columns([1.05, 1.35], gap="large")

    with left_col:
        st.markdown("### Job List")

        for idx, row in page_df.iterrows():
            global_idx = start_idx + idx
            is_saved = global_idx in st.session_state["saved_jobs"]

            st.markdown(f"""
            <div class="job-list-card">
                <div class="job-title">{row['job_title']}</div>
                <div class="job-meta"><strong>{row['company']}</strong></div>
                <div class="job-meta">{row['job_location']}</div>
                <div class="job-meta">
                    <span class="badge badge-blue">{row['work_mode']}</span>
                    <span class="badge badge-purple">{row['industry']}</span>
                    <span class="badge badge-orange">{row['job_level_norm']}</span>
                    <span class="badge badge-green">{row['employment_type_norm']}</span>
                </div>
                <div class="small-muted">Overall Match: {row['final_score'] * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("View Details", key=f"view_{global_idx}"):
                    st.session_state["selected_job_idx"] = idx
                    st.rerun()
            with c2:
                if is_saved:
                    if st.button("Unsave", key=f"unsave_{global_idx}"):
                        st.session_state["saved_jobs"].remove(global_idx)
                        st.rerun()
                else:
                    if st.button("Save Job", key=f"save_{global_idx}"):
                        st.session_state["saved_jobs"].append(global_idx)
                        st.rerun()

        if st.session_state["saved_jobs"]:
            st.markdown("### ⭐ Saved Jobs")
            st.write(f"Saved jobs count: {len(st.session_state['saved_jobs'])}")

    with right_col:
        if len(page_df) > 0:
            selected_idx = min(st.session_state["selected_job_idx"], len(page_df) - 1)
            selected_row = page_df.iloc[selected_idx]
            selected_global_idx = start_idx + selected_idx
            is_saved_selected = selected_global_idx in st.session_state["saved_jobs"]

            st.markdown('<div class="job-detail-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="job-detail-title">{selected_row["job_title"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="job-meta"><strong>Company:</strong> {selected_row["company"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="job-meta"><strong>Location:</strong> {selected_row["job_location"]}</div>', unsafe_allow_html=True)

            if is_saved_selected:
                st.markdown('<div class="saved-pill">Saved</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="job-meta">
                <span class="badge badge-blue">{selected_row['work_mode']}</span>
                <span class="badge badge-purple">{selected_row['industry']}</span>
                <span class="badge badge-orange">{selected_row['job_level_norm']}</span>
                <span class="badge badge-green">{selected_row['employment_type_norm']}</span>
            </div>
            """, unsafe_allow_html=True)

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Overall Match", f"{selected_row['final_score'] * 100:.1f}%")
            with s2:
                st.metric("Profile Similarity", f"{selected_row['nlp_score'] * 100:.1f}%")
            with s3:
                st.metric("Skills Match", f"{selected_row['skill_score'] * 100:.1f}%")
            with s4:
                st.metric("Role Match", f"{selected_row['role_bonus'] * 100:.1f}%")

            st.info(f"**Why this job ranked here:** {selected_row['match_reason']}")

            matched = selected_row["matched_skills"] if isinstance(selected_row["matched_skills"], list) else []
            missing = selected_row["missing_skills"] if isinstance(selected_row["missing_skills"], list) else []
            extra_user = selected_row["extra_user_skills"] if isinstance(selected_row["extra_user_skills"], list) else []

            st.write("### ✅ Matched Skills")
            st.write(", ".join(matched[:15]) if matched else "—")

            st.write("### ❌ Missing Skills")
            st.write(", ".join(missing[:15]) if missing else "—")

            if extra_user:
                st.write("### ➕ Additional Skills You Have")
                st.write(", ".join(extra_user[:15]))

            st.write("### 🛠 Extracted Job Skills")
            expanded_skills = selected_row["expanded_job_skills"] if isinstance(selected_row["expanded_job_skills"], list) else []
            st.write(", ".join(expanded_skills[:25]) if expanded_skills else "Not available")

            st.write("### 📄 Job Description")
            desc = selected_row["job_description"]
            st.write(desc[:3500] + ("..." if len(desc) > 3500 else ""))

            if pd.notna(selected_row.get("salary_min")) or pd.notna(selected_row.get("salary_max")):
                sal_min = selected_row.get("salary_min")
                sal_max = selected_row.get("salary_max")

                if pd.notna(sal_min) and pd.notna(sal_max):
                    st.write(f"### 💰 Salary Range\n${float(sal_min):,.0f} - ${float(sal_max):,.0f}")
                elif pd.notna(sal_min):
                    st.write(f"### 💰 Salary Minimum\n${float(sal_min):,.0f}")
                elif pd.notna(sal_max):
                    st.write(f"### 💰 Salary Maximum\n${float(sal_max):,.0f}")

            if isinstance(selected_row.get("job_link", ""), str) and selected_row["job_link"].strip():
                st.link_button("Open Job Posting", selected_row["job_link"])

            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-title">📊 Job Market Summary</div>', unsafe_allow_html=True)
    sm1, sm2 = st.columns(2)

    with sm1:
        st.write("**Top Job Titles**")
        top_titles = results_all_ranked["job_title"].value_counts().head(5)
        if not top_titles.empty:
            st.dataframe(top_titles.rename_axis("Job Title").reset_index(name="Count"), use_container_width=True)

        st.write("**Top Companies**")
        top_companies = results_all_ranked["company"].value_counts().head(5)
        if not top_companies.empty:
            st.dataframe(top_companies.rename_axis("Company").reset_index(name="Count"), use_container_width=True)

    with sm2:
        st.write("**Industry Distribution**")
        industry_counts = results_all_ranked["industry"].value_counts()
        if not industry_counts.empty:
            st.bar_chart(industry_counts)

        st.write("**Career Level Distribution**")
        level_counts = results_all_ranked["job_level_norm"].value_counts()
        if not level_counts.empty:
            st.bar_chart(level_counts)

    st.divider()

    st.markdown('<div class="section-title">🧠 Resume Improvement Suggestions</div>', unsafe_allow_html=True)
    top_missing = []
    for ms in results_all_ranked.head(25)["missing_skills"].tolist():
        if isinstance(ms, list):
            top_missing.extend(ms)

    missing_counts = pd.Series(top_missing).value_counts().head(10) if len(top_missing) > 0 else pd.Series(dtype=int)

    if len(missing_counts):
        st.write("Based on the top ranked jobs, consider learning or highlighting these skills:")
        st.write(", ".join(missing_counts.index.tolist()))
    else:
        st.success("No major missing skills detected in the top ranked jobs.")

    st.markdown("### Suggested Career Paths")
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
        st.write(", ".join(suggested_roles))
    else:
        st.write("You can also explore related analyst, coordinator, specialist, or associate roles.")

    st.markdown("### 💰 Salary Insight")
    salary_df = results_all_ranked[["salary_min", "salary_max"]].dropna(how="all")
    if not salary_df.empty:
        valid_min = pd.to_numeric(salary_df["salary_min"], errors="coerce").dropna()
        valid_max = pd.to_numeric(salary_df["salary_max"], errors="coerce").dropna()

        if not valid_min.empty or not valid_max.empty:
            avg_min = valid_min.mean() if not valid_min.empty else None
            avg_max = valid_max.mean() if not valid_max.empty else None

            if avg_min is not None and avg_max is not None:
                st.write(f"Estimated average salary range from ranked jobs: ${avg_min:,.0f} - ${avg_max:,.0f}")
            elif avg_min is not None:
                st.write(f"Estimated average minimum salary: ${avg_min:,.0f}")
            elif avg_max is not None:
                st.write(f"Estimated average maximum salary: ${avg_max:,.0f}")
        else:
            st.write("Salary data is not available for the current ranked jobs.")
    else:
        st.write("Salary data is not available for the current ranked jobs.")

    st.divider()

    st.markdown('<div class="section-title">⬇️ Download Results</div>', unsafe_allow_html=True)
    download_cols = [
        "job_title", "company", "job_location", "work_mode", "industry",
        "employment_type_norm", "job_type", "job_level_norm", "job_skills",
        "final_score", "nlp_score", "skill_score", "role_bonus",
        "salary_min", "salary_max", "job_link"
    ]

    csv_all = results_all_ranked[download_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download All Ranked Results (CSV)",
        data=csv_all,
        file_name="all_ranked_job_recommendations.csv",
        mime="text/csv"
    )

    csv_page = page_df[download_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Current Page Results (CSV)",
        data=csv_page,
        file_name=f"job_recommendations_page_{current_page}.csv",
        mime="text/csv"
    )

else:
    st.info("Set your filters and click **Find Matching Jobs** to load live jobs.")
