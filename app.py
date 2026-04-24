"""
AI Job Recommendation System
============================
A Streamlit web app that:
  1. Reads the user's resume (PDF / DOCX / TXT)
  2. Fetches live job postings from the Adzuna API
  3. Scrapes the FULL job posting (BeautifulSoup) when the user clicks
     "View Details" - because the Adzuna free-tier snippet is only ~150
     characters and gives incorrect skill extraction
  4. Extracts skills from both resume and jobs (longest-skill-first,
     with aliases) and produces honest, verifiable match scores

Every number shown is the actual computed value. If skill extraction
is incomplete (<4 skills) we cap the displayed skill score at 50% and
show a warning, so the user is never misled.

Run locally:
    streamlit run app.py
"""

import re
import io
import math
import time
import html
from urllib.parse import urlparse

import requests
import pandas as pd
import streamlit as st
import pdfplumber
import docx

from bs4 import BeautifulSoup


def safe(x) -> str:
    """HTML-escape any dynamic value before it goes into an unsafe_allow_html block."""
    return html.escape(str(x if x is not None else ""))


def format_pct(value, decimals=1) -> str:
    """Format score ratios as UI percentages with one shared precision."""
    try:
        pct = float(value) * 100
    except (TypeError, ValueError):
        pct = 0.0
    if not math.isfinite(pct):
        pct = 0.0
    return f"{pct:.{decimals}f}%"

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

# Minimum number of extracted job skills before a 100% match is considered
# trustworthy. Below this we cap the skill score at 50% (honesty rule).
MIN_JOB_SKILLS_FOR_FULL_SCORE = 4

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# BUG 6 FIX — alternate UA used on retry when the first fetch fails.
ALTERNATE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.0 Safari/605.1.15"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

BAD_KEYWORDS = {
    "staffing", "recruiter", "recruitment", "placement", "talent solutions",
    "training program", "bench sales", "consultancy", "corp to corp", "c2c",
    "w2 only", "no job", "marketing partner", "immediate joiners",
    "third party", "implementation partner"
}

BAD_TITLE_KEYWORDS = {
    "training", "bench", "walk-in", "walk in"
}

BAD_DOMAINS = {
    "monster.com", "ziprecruiter.com", "dice.com", "careerbuilder.com",
    "jobrapido.com", "lensa.com", "talent.com", "jooble.org",
    "learn4good.com", "grabjobs.co"
}

# =========================================================
# SKILL DICTIONARY
# =========================================================
COMMON_JOB_SKILLS = {
    # ---- Data & Analytics ----
    "sql", "python", "excel", "power bi", "tableau", "pandas", "numpy",
    "matplotlib", "seaborn", "scikit-learn", "machine learning", "nlp",
    "statistics", "data analysis", "data analytics", "data visualization",
    "data science", "data management", "data cleaning", "data validation",
    "dashboards", "business intelligence", "eda", "predictive modeling",
    "tf-idf", "streamlit", "reporting", "deep learning",
    "feature engineering", "data modeling", "data warehousing",
    "data warehouse", "etl", "elt", "kpi", "metrics",
    "ad hoc reporting", "data governance", "forecasting",
    "a/b testing", "regression analysis", "predictive analytics",
    "natural language processing", "computer vision",
    "structured data", "unstructured data", "data profiling",
    "data quality", "data integration", "data pipelines",
    "data mining", "etl pipelines", "data storytelling", "storytelling",
    "data reliability", "schema design", "scope management",
    "requirements negotiation", "data quality testing",
    "data presentation", "insight generation",
    # BUG 3/4/5 FIX — skills flagged as missing by professor
    "data interpretation", "presentation skills", "report writing",

    # ---- Databases ----
    "mysql", "postgresql", "sqlite", "sql server", "mongodb",
    "oracle", "nosql", "snowflake", "bigquery", "redshift",
    "stored procedures", "transact-sql", "t-sql",

    # ---- Cloud & DevOps ----
    "aws", "azure", "gcp", "cloud computing", "docker", "kubernetes",
    "terraform", "ansible", "git", "ci/cd", "devops", "linux",
    "jenkins", "airflow", "databricks",

    # ---- Web & Programming ----
    "flask", "django", "html", "css", "javascript", "typescript",
    "java", "c++", "c#", "ruby", "go", "rust", "kotlin", "swift",
    "node.js", "react", "angular", "vue", "rest api", "api",
    "graphql", "spring boot",

    # ---- BI / Reporting tools ----
    "ssis", "ssrs", "ssms", "ssas", "looker", "qlik", "qlik sense",
    "microstrategy", "alteryx", "knime", "informatica", "sas", "r",
    "spss", "google sheets", "google analytics", "power query",
    "pivot tables", "vlookup", "xlookup",
    "gis", "gis tools", "arcgis", "qgis",
    "data integrity", "data accuracy",
    "ad hoc", "performance metrics", "kpi tracking",
    "stakeholder communication", "cross-functional", "project tracking",

    # ---- Office / Productivity ----
    "powerpoint", "word", "outlook", "ms office", "ms access",
    "sharepoint", "ms project", "ms visio", "jira", "confluence",
    "agile", "scrum", "kanban",

    # ---- Soft skills ----
    "communication", "teamwork", "leadership", "documentation",
    "problem solving", "project management", "stakeholder management",
    "critical thinking", "time management", "analytical thinking",
    "collaboration",

    # ---- Healthcare ----
    "patient care", "clinical assessment", "triage", "diagnosis",
    "treatment planning", "care coordination", "patient education",
    "medical records", "ehr", "emr", "epic", "cerner", "hipaa",
    "infection control", "telehealth", "case management",
    "medical terminology",

    # ---- Accounting / Finance ----
    "accounting", "bookkeeping", "financial reporting", "budgeting",
    "audit", "auditing", "gaap", "accounts payable",
    "accounts receivable", "bank reconciliation", "payroll",
    "financial analysis", "tax preparation", "quickbooks",
    "general ledger", "cash flow", "reconciliation",

    # ---- Business / Ops ----
    "business analysis", "requirements gathering", "process improvement",
    "operations management", "workflow optimization", "crm",
    "salesforce", "vendor management", "supply chain",
    "customer service", "sap", "erp", "workday",

    # ---- Marketing ----
    "seo", "sem", "social media", "campaign management",
    "content marketing", "content creation", "email marketing",
    "brand strategy", "market research", "advertising",
    "copywriting", "digital marketing", "lead generation",

    # ---- HR / Education ----
    "recruitment", "talent acquisition", "employee relations",
    "onboarding", "benefits administration", "performance management",
    "hris", "policy compliance", "interviewing", "candidate screening",
    "teaching", "curriculum development", "lesson planning",
    "classroom management", "assessment",
    "academic advising", "instruction",

    # ---- Testing / QA ----
    "unit testing", "integration testing", "qa", "quality assurance",
    "test automation", "selenium",
}

# Map variants to the canonical skill name the rest of the app uses.
SKILL_ALIASES = {
    # Excel
    "ms excel": "excel",
    "microsoft excel": "excel",
    "excel reports": "excel",
    "excel spreadsheets": "excel",
    "excel spreadsheet": "excel",
    "advanced excel": "excel",

    # Power BI
    "powerbi": "power bi",
    "power bi desktop": "power bi",
    "microsoft power bi": "power bi",
    "ms power bi": "power bi",
    "power bi dashboards": "power bi",
    "power bi dashboard": "power bi",
    "power bi reports": "power bi",
    "power bi report": "power bi",

    # Scikit-learn
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",

    # NLP
    "natural language processing": "nlp",

    # SQL variants all collapse to "sql" (per CLAUDE.md)
    "sqlite": "sql",
    "mysql": "sql",
    "postgresql": "sql",
    "postgres": "sql",
    "sql server": "sql",
    "ms sql": "sql",
    "ms sql server": "sql",
    "t-sql": "sql",
    "tsql": "sql",
    "transact-sql": "sql",
    "transact sql": "sql",
    "structured query language": "sql",

    # AWS
    "amazon web services": "aws",
    "aws s3": "aws",

    # TF-IDF
    "tf idf": "tf-idf",
    "tfidf": "tf-idf",

    # Misc
    "py": "python",
    "js": "javascript",
    "restful api": "rest api",
    "application programming interface": "api",
    "bi tools": "business intelligence",
    "business intelligence tools": "business intelligence",
    "google bigquery": "bigquery",
    "amazon redshift": "redshift",
    "g sheets": "google sheets",
    "microsoft office": "ms office",
    "microsoft sharepoint": "sharepoint",
    "ms sharepoint": "sharepoint",
    "microsoft project": "ms project",
    "microsoft visio": "ms visio",
    "electronic health record": "ehr",
    "electronic medical record": "emr",
    "search engine optimization": "seo",
    "search engine marketing": "sem",
    "customer relationship management": "crm",
    "human resource information system": "hris",
    "dashboarding": "dashboards",
    "dashboard development": "dashboards",
    "dashboard": "dashboards",
    "reporting tools": "reporting",
    "reporting platforms": "reporting",
    "ad hoc reporting": "reporting",
    "adhoc reporting": "reporting",
    "ad-hoc reporting": "reporting",
    "data platforms": "database",
    "pivot table": "pivot tables",
    "arc gis": "arcgis",
    "gis software": "gis tools",
    "geographic information system": "gis",
    "geographic information systems": "gis",
    "c programming": "c",
    "big data analytics": "data analytics",
    "statistical computing": "statistics",
    "qlik sense": "qlik",
    "tableau software": "tableau",
    "a/b tests": "a/b testing",
    "ab testing": "a/b testing",
    "ab tests": "a/b testing",
    "c sharp": "c#",
    "csharp": "c#",
    "cplusplus": "c++",

    # Professor-flagged phrase variants (Monday review)
    "visualization": "data visualization",
    "visualizations": "data visualization",
    "data viz": "data visualization",
    "data visualisation": "data visualization",
    "visualization tools": "data visualization",
    "kpis": "kpi",
    "key performance indicators": "kpi",
    "key performance indicator": "kpi",
    "schema": "schema design",
    "schemas": "schema design",
    "database schema": "schema design",
    "schema development": "schema design",
    "storytelling with data": "data storytelling",
    "present insights": "data storytelling",
    "presenting insights": "data storytelling",
    "ensure data accuracy": "data accuracy",
    "ensures data accuracy": "data accuracy",
    "ensuring data accuracy": "data accuracy",
    "ensure accuracy": "data accuracy",
    "ensure data reliability": "data reliability",
    "ensuring data reliability": "data reliability",
    "reliable data": "data reliability",
    "data quality test": "data quality testing",
    "data quality tests": "data quality testing",
    "quality testing": "data quality testing",
    "negotiate requirements": "requirements negotiation",
    "negotiating requirements": "requirements negotiation",
    "etl processes": "etl",
    "etl process": "etl",
    "etl/elt": "etl",
    "extract transform load": "etl",
    "extract, transform, load": "etl",

    # BUG 3 FIX
    "interpretation": "data interpretation",
    # BUG 4 FIX
    "presentation": "presentation skills",
    "presenting": "presentation skills",
    # BUG 5 FIX
    "report development": "report writing",
    "writing reports": "report writing",
}

# PROBLEM 1 FIX — acronyms that may appear ALL-CAPS in JDs (ETL, SQL, KPI...).
# These are scanned case-insensitively against the ORIGINAL raw text so that
# word-boundary matching doesn't depend on lowercasing quirks.
ACRONYM_SKILLS = {
    "etl", "elt", "sql", "kpi", "nlp", "eda", "qa", "ehr", "emr",
    "aws", "gcp", "api", "ci/cd", "seo", "sem", "crm", "hris", "erp",
    "sap", "ssrs", "ssis", "ssas", "ssms", "spss", "sas",
    "hipaa", "oauth", "sso", "icd-10", "cpt", "gaap",
}

# PROBLEM 6 FIX — proper display casing for skills shown in the UI.
DISPLAY_CASE = {
    "etl": "ETL", "elt": "ELT", "sql": "SQL", "kpi": "KPI", "nlp": "NLP",
    "eda": "EDA", "qa": "QA", "ehr": "EHR", "emr": "EMR",
    "aws": "AWS", "gcp": "GCP", "api": "API", "ci/cd": "CI/CD",
    "seo": "SEO", "sem": "SEM", "crm": "CRM", "hris": "HRIS", "erp": "ERP",
    "sap": "SAP", "ssrs": "SSRS", "ssis": "SSIS", "ssas": "SSAS", "ssms": "SSMS",
    "spss": "SPSS", "sas": "SAS", "hipaa": "HIPAA", "oauth": "OAuth", "sso": "SSO",
    "icd-10": "ICD-10", "cpt": "CPT", "gaap": "GAAP",
    "tf-idf": "TF-IDF", "rest api": "REST API", "a/b testing": "A/B Testing",
    "html": "HTML", "css": "CSS", "nosql": "NoSQL",
    "power bi": "Power BI", "ms sql": "MS SQL", "sql server": "SQL Server",
    "ms office": "MS Office", "ms access": "MS Access",
    "ms project": "MS Project", "ms visio": "MS Visio",
    "python": "Python", "java": "Java", "javascript": "JavaScript",
    "typescript": "TypeScript", "ruby": "Ruby", "go": "Go",
    "rust": "Rust", "kotlin": "Kotlin", "swift": "Swift", "r": "R",
    "excel": "Excel", "tableau": "Tableau", "pandas": "Pandas",
    "numpy": "NumPy", "matplotlib": "Matplotlib", "seaborn": "Seaborn",
    "scikit-learn": "Scikit-learn", "streamlit": "Streamlit",
    "jupyter": "Jupyter", "pytorch": "PyTorch", "tensorflow": "TensorFlow",
    "keras": "Keras", "xgboost": "XGBoost",
    "flask": "Flask", "django": "Django", "react": "React",
    "angular": "Angular", "vue": "Vue", "node.js": "Node.js",
    "spring boot": "Spring Boot", "graphql": "GraphQL",
    "docker": "Docker", "kubernetes": "Kubernetes",
    "terraform": "Terraform", "ansible": "Ansible",
    "jenkins": "Jenkins", "airflow": "Airflow", "databricks": "Databricks",
    "snowflake": "Snowflake", "bigquery": "BigQuery", "redshift": "Redshift",
    "mongodb": "MongoDB", "mysql": "MySQL", "postgresql": "PostgreSQL",
    "sqlite": "SQLite", "oracle": "Oracle",
    "git": "Git", "linux": "Linux",
    "jira": "JIRA", "confluence": "Confluence",
    "agile": "Agile", "scrum": "Scrum", "kanban": "Kanban",
    "salesforce": "Salesforce", "workday": "Workday",
    "quickbooks": "QuickBooks", "epic": "Epic", "cerner": "Cerner",
    "selenium": "Selenium",
    "looker": "Looker", "qlik": "Qlik", "alteryx": "Alteryx",
    "knime": "KNIME", "informatica": "Informatica",
    "microstrategy": "MicroStrategy",
    "power query": "Power Query",
    "google analytics": "Google Analytics",
    "google sheets": "Google Sheets", "azure": "Azure",
    "arcgis": "ArcGIS", "qgis": "QGIS", "gis": "GIS",
    "machine learning": "Machine Learning", "deep learning": "Deep Learning",
    "natural language processing": "Natural Language Processing",
    "computer vision": "Computer Vision",
    "data analysis": "Data Analysis", "data analytics": "Data Analytics",
    "data visualization": "Data Visualization", "data science": "Data Science",
    "data cleaning": "Data Cleaning", "data validation": "Data Validation",
    "data management": "Data Management", "data mining": "Data Mining",
    "data integrity": "Data Integrity", "data accuracy": "Data Accuracy",
    "data reliability": "Data Reliability", "data governance": "Data Governance",
    "data modeling": "Data Modeling", "data warehousing": "Data Warehousing",
    "data warehouse": "Data Warehouse", "data profiling": "Data Profiling",
    "data quality": "Data Quality", "data quality testing": "Data Quality Testing",
    "data integration": "Data Integration", "data pipelines": "Data Pipelines",
    "data storytelling": "Data Storytelling",
    "data presentation": "Data Presentation",
    "insight generation": "Insight Generation",
    "etl pipelines": "ETL Pipelines",
    "schema design": "Schema Design", "scope management": "Scope Management",
    "requirements negotiation": "Requirements Negotiation",
    "requirements gathering": "Requirements Gathering",
    "pivot tables": "Pivot Tables", "stored procedures": "Stored Procedures",
    "kpi tracking": "KPI Tracking", "performance metrics": "Performance Metrics",
    "business intelligence": "Business Intelligence",
    "predictive modeling": "Predictive Modeling",
    "predictive analytics": "Predictive Analytics",
    "feature engineering": "Feature Engineering",
    "regression analysis": "Regression Analysis",
    "dashboards": "Dashboards", "reporting": "Reporting",
    "documentation": "Documentation", "statistics": "Statistics",
    "forecasting": "Forecasting", "storytelling": "Storytelling",
    "communication": "Communication", "teamwork": "Teamwork",
    "leadership": "Leadership", "collaboration": "Collaboration",
    "project management": "Project Management",
    "stakeholder management": "Stakeholder Management",
    "problem solving": "Problem Solving",
    "critical thinking": "Critical Thinking",
    "time management": "Time Management",
    "analytical thinking": "Analytical Thinking",
    # BUG 3/4/5 FIX
    "data interpretation": "Data Interpretation",
    "presentation skills": "Presentation Skills",
    "report writing": "Report Writing",
}


def display_skill(skill: str) -> str:
    """PROBLEM 6 FIX — return the proper-case form of a skill for UI display."""
    s = (skill or "").strip().lower()
    if s in DISPLAY_CASE:
        return DISPLAY_CASE[s]
    if " " in s:
        return s.title()
    return s.capitalize() if s.isalpha() else s


def display_skills_list(skills) -> list:
    return [display_skill(s) for s in (skills or [])]


def display_skills_str(skills) -> str:
    return ", ".join(display_skills_list(skills))


# Words that appear in resumes and JDs but are NEVER real skills.
# NOTE: "preferred" and "teamwork" are intentionally NOT here - per CLAUDE.md.
GENERIC_NOT_SKILLS = {
    # Generic role/responsibility words
    "experience", "knowledge", "skills", "ability", "responsible",
    "responsibilities", "tools", "tool", "management", "work",
    "working", "support", "operations", "business", "computer",
    "university", "education", "training", "certification",
    "certifications", "role", "roles", "candidate", "job",
    "position", "positions", "strong", "excellent",
    "required", "ability to", "must have", "should have", "plus",
    "etc", "using", "including", "basic", "advanced", "benefits",
    "salary", "pay range", "responsibility", "duties", "duty",
    "programs", "program", "location",
    "citizen", "citizenship", "clearance", "applicant", "applicants",
    "collaboratively", "selected", "currently", "minimum", "maximum",
    "year", "years", "degree", "bachelor", "master", "phd",
    "associate", "associates",
    "related", "equivalent", "comparable",
    "full time", "part time", "contract", "temporary", "permanent",
    "company", "employer", "department", "division", "group",
    "ideal", "looking for", "seeking", "need", "needs",
    "us citizen", "us citizens", "united states",
    "description", "overview", "summary", "about", "mission",
    "compensation", "equal opportunity", "eoe",
    "multiple", "various", "several", "other", "additional",
    "environment", "fast paced", "fast-paced", "detail oriented",
    "independently",
    "ensure", "maintain", "provide", "assist", "perform",
    "develop", "create", "implement", "coordinate",
    "review", "prepare", "analyze", "evaluate", "monitor",
    "communicate", "participate", "contribute", "handle",
    "formats", "format", "methods", "method", "processes", "process",
    "networks", "network", "systems", "system",

    # EEO / legal boilerplate (these leaked from page footers)
    "color", "religion", "sex", "age", "ancestry", "race",
    "disability", "veteran", "veterans", "genetic", "marital",
    "gender", "orientation", "national origin", "pregnancy",
    "ethnicity", "military", "accommodation", "accommodations",
    "affirmative", "diversity", "inclusion", "protected",
    "qualified", "ordinance", "regulation", "applicable",
    "consideration", "regardless", "action", "gender identity",
    "sexual orientation", "protected veteran", "genetic factors",
    "marital status", "gender identity and expression",

    # Job-board navigation / UI noise
    "back to last search", "apply for this job",
    "create alert", "similar jobs", "receive similar",
    "back to", "sign in", "register", "login",
    "save job", "share", "print", "report", "subscribe",
    "email alerts", "job alerts",

    # Salary stats / page chrome
    "per year", "per hour", "estimated", "national average",
    "salary comparison", "stats for this job",

    # Generic descriptor phrases (NOT skills)
    "attention to detail",
    "written", "verbal", "on-site", "onsite", "hybrid", "remote",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "available", "proficiency", "familiarity",
    "working knowledge", "knowledge of",
    "actionable", "actionable insights", "insights", "trends",
    "accuracy", "consistency", "integrity",
    "data requests",
    "bench", "walk-in",
}

# Words that, if present in a candidate skill phrase, disqualify it as being a skill.
INSTITUTION_WORDS = {
    "college", "university", "academy", "institute", "school",
    "inc", "llc", "corp", "ltd", "gmbh", "co.",
}

# Common US place / state tokens that leak from page headers/footers.
US_LOCATION_WORDS = {
    "atlanta", "georgia", "marietta", "fayetteville", "wellston",
    "san", "ramon", "cobb", "county", "nashville", "tennessee",
    "ga", "tn", "ca", "ny", "tx", "fl", "wa", "or", "nj", "pa",
    "mi", "oh", "il", "in", "az", "co", "md", "va", "nc", "sc",
}

# =========================================================
# ROLE -> TYPICAL RESPONSIBILITIES
# Used for Score 3 (Role Match). We compare the job's text
# against these responsibility keywords so the match is
# meaningful even when the title is generic.
# =========================================================
TYPICAL_ROLE_RESPONSIBILITIES = {
    "data analyst": [
        "sql", "python", "excel", "data analysis", "reporting",
        "dashboards", "data visualization", "power bi", "tableau",
        "insights", "trends", "data cleaning", "data management",
        "business intelligence", "kpi", "metrics", "queries",
        "stakeholder", "etl",
    ],
    "data scientist": [
        "machine learning", "python", "statistics", "modeling",
        "nlp", "scikit-learn", "pandas", "numpy", "data analysis",
        "predictive modeling", "feature engineering", "deep learning",
        "experimentation", "a/b testing", "algorithms",
    ],
    "data engineer": [
        "sql", "python", "etl", "elt", "airflow", "data pipelines",
        "data warehouse", "snowflake", "bigquery", "redshift",
        "spark", "kafka", "aws", "azure", "gcp",
        "data modeling",
    ],
    "business analyst": [
        "requirements gathering", "stakeholder management",
        "process improvement", "business analysis", "workflow",
        "documentation", "reporting", "sql", "excel", "dashboards",
        "user stories", "agile", "jira",
    ],
    "software engineer": [
        "python", "java", "javascript", "git", "api", "rest api",
        "unit testing", "debugging", "algorithms", "data structures",
        "agile", "code review", "ci/cd", "docker",
    ],
    "accountant": [
        "accounting", "bookkeeping", "reconciliation", "general ledger",
        "accounts payable", "accounts receivable", "gaap",
        "financial reporting", "audit", "tax", "excel", "quickbooks",
    ],
    "financial analyst": [
        "financial analysis", "forecasting", "budgeting", "excel",
        "variance analysis", "financial reporting", "modeling",
        "dashboards", "kpi", "metrics", "sql",
    ],
    "nurse practitioner": [
        "patient care", "clinical assessment", "diagnosis",
        "treatment planning", "triage", "care coordination",
        "ehr", "emr", "medical terminology", "patient education",
    ],
    "registered nurse": [
        "patient care", "clinical assessment", "triage",
        "care coordination", "medication administration", "ehr",
        "emr", "patient education", "infection control",
    ],
    "marketing analyst": [
        "google analytics", "seo", "sem", "campaign management",
        "market research", "data analysis", "dashboards", "kpi",
        "content marketing", "email marketing", "crm",
    ],
    "project manager": [
        "project management", "stakeholder management", "agile",
        "scrum", "jira", "risk management", "budgeting",
        "documentation", "timeline", "planning",
    ],
    "hr generalist": [
        "recruitment", "employee relations", "onboarding",
        "benefits administration", "hris", "performance management",
        "policy compliance", "talent acquisition",
    ],
    "teacher": [
        "teaching", "lesson planning", "curriculum development",
        "classroom management", "assessment", "grading",
        "student engagement", "instruction",
    ],
}

INDUSTRY_RULES = {
    "Healthcare": [
        "hospital", "clinic", "patient", "nurse", "physician",
        "medical", "health", "ehr", "emr", "hipaa", "clinical",
    ],
    "Accounting & Finance": [
        "bank", "finance", "financial", "accounting", "audit",
        "tax", "payroll", "gaap", "bookkeeping", "accounts payable",
        "quickbooks",
    ],
    "Marketing": [
        "marketing", "seo", "sem", "content", "campaign", "brand",
        "social media", "advertising", "digital marketing",
    ],
    "Business & Operations": [
        "operations", "stakeholder", "requirements",
        "process improvement", "project management", "workflow",
        "coordination", "administration",
    ],
    "Human Resources": [
        "recruitment", "talent", "hr ", "onboarding",
        "employee relations", "hris", "staffing", "human resources",
    ],
    "Data & IT": [
        "software", "developer", "cloud", "aws", "azure", "sql",
        "python", "analytics", "machine learning", "data scientist",
        "data engineer", "api", "database", "power bi", "tableau",
        "java", "javascript", "alteryx", "knime", "qlik",
        "informatica",
    ],
    "Education": [
        "university", "school", "teacher", "student", "education",
        "curriculum",
    ],
}

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.hero {
    padding: 1.2rem 1.3rem; border-radius: 18px;
    background: linear-gradient(135deg, #eff6ff 0%, #f5f3ff 50%, #ecfeff 100%);
    border: 1px solid #e5e7eb; margin-bottom: 1rem;
}
.hero h1 { margin: 0; font-size: 2rem; }
.hero p { margin: 0.45rem 0 0 0; color: #4b5563; font-size: 1rem; }

.metric-card {
    background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 18px; padding: 0.9rem 1rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.03);
}
.metric-title { font-size: 0.9rem; color: #6b7280; margin-bottom: 0.2rem; }
.metric-value { font-size: 1.75rem; font-weight: 700; color: #111827; }

.section-title {
    font-size: 1.35rem; font-weight: 700;
    margin-top: 0.25rem; margin-bottom: 0.8rem;
}

.job-list-card {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px;
    padding: 0.85rem 0.95rem; margin-bottom: 0.75rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.job-list-card-selected {
    background: #eff6ff; border: 2px solid #2563eb; border-radius: 16px;
    padding: 0.85rem 0.95rem; margin-bottom: 0.75rem;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.15);
}
.viewing-tag {
    display: inline-block; background: #2563eb; color: white;
    padding: 0.15rem 0.55rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; margin-bottom: 0.35rem;
}

.job-detail-card {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 18px;
    padding: 1rem 1.1rem; box-shadow: 0 3px 12px rgba(0,0,0,0.03);
}
.job-title { font-size: 1.1rem; font-weight: 700; color: #111827; margin-bottom: 0.3rem; }
.job-detail-title { font-size: 1.55rem; font-weight: 700; color: #111827; margin-bottom: 0.4rem; }
.job-meta { color: #374151; font-size: 0.94rem; margin-bottom: 0.35rem; }
.small-muted { color: #6b7280; font-size: 0.9rem; }

.badge {
    display: inline-block; padding: 0.22rem 0.62rem;
    border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    margin-right: 0.3rem; margin-bottom: 0.3rem;
}
.badge-blue   { background: #dbeafe; color: #1d4ed8; }
.badge-green  { background: #dcfce7; color: #166534; }
.badge-purple { background: #ede9fe; color: #6d28d9; }
.badge-orange { background: #ffedd5; color: #c2410c; }
.badge-pink   { background: #fce7f3; color: #be185d; }
.badge-red    { background: #fee2e2; color: #991b1b; }

.matched-box {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 12px; padding: 0.75rem 0.9rem; margin-bottom: 0.6rem;
}
.missing-box {
    background: #fef2f2; border: 1px solid #fecaca;
    border-radius: 12px; padding: 0.75rem 0.9rem; margin-bottom: 0.6rem;
}
.required-box {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 12px; padding: 0.75rem 0.9rem; margin-bottom: 0.6rem;
}
.extra-box {
    background: #faf5ff; border: 1px solid #e9d5ff;
    border-radius: 12px; padding: 0.75rem 0.9rem; margin-bottom: 0.6rem;
}

.warning-banner {
    background: #fffbeb; border: 1px solid #fde68a;
    color: #92400e; border-radius: 12px;
    padding: 0.7rem 0.9rem; margin: 0.5rem 0;
    font-weight: 600;
}
.success-banner {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    color: #166534; border-radius: 12px;
    padding: 0.7rem 0.9rem; margin: 0.5rem 0;
    font-weight: 600;
}

.saved-pill {
    display: inline-block; background: #fef3c7; color: #92400e;
    padding: 0.2rem 0.6rem; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>💼 AI Job Recommendation System</h1>
  <p>Live Adzuna postings + full-page scraping + honest skill matching. Every score is verifiable.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# TEXT HELPERS
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


_SENTENCE_WORDS = {
    "shall", "will", "would", "could", "should", "must", "may", "might",
    "the", "this", "that", "these", "those", "their", "our", "your",
    "who", "whom", "which", "where", "when", "what", "how", "why",
    "been", "being", "have", "has", "had", "does", "did",
    "not", "also", "very", "just", "only", "than", "then",
    "from", "into", "with", "within", "without", "between",
    "about", "after", "before", "during", "through",
    "across", "along", "around", "because", "since",
}

_NON_SKILL_PATTERNS = [
    r"\b(?:us citizen|citizen|citizenship|clearance|secret clearance)\b",
    r"\b(?:at least|one or more|years? of|degree in|four year|must be|ability to|required to|responsible for|selected for|able to)\b",
    r"\b(?:shall|will|would|could|should)\s+\w+",
    r"^the\s+",
]

# BUG 2 FIX — single-char programming-language skills (R, C).
SINGLE_CHAR_SKILLS = {"r", "c"}

# Multi-word skills that ARE allowed (pre-approved). Anything else with 2+ words
# must exist in COMMON_JOB_SKILLS to be accepted.
ALLOWED_MULTIWORD_SKILLS = {
    "data analysis", "data analytics", "data visualization",
    "data science", "data cleaning", "data validation",
    "data management", "data mining", "data integrity",
    "data accuracy", "data governance", "data modeling",
    "data warehousing", "data warehouse", "data profiling",
    "data quality", "data integration", "data pipelines",
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "power bi", "power query", "pivot tables",
    "business intelligence", "predictive modeling", "predictive analytics",
    "feature engineering", "a/b testing", "regression analysis",
    "scikit-learn", "sql server", "ms sql", "google analytics",
    "google sheets", "rest api", "cloud computing",
    "project management", "stakeholder management",
    "customer service", "problem solving", "critical thinking",
    "time management", "analytical thinking", "ad hoc reporting",
    "ad hoc", "gis tools", "social media", "content marketing",
    "email marketing", "digital marketing", "lead generation",
    "market research", "campaign management", "brand strategy",
    "financial analysis", "financial reporting", "general ledger",
    "accounts payable", "accounts receivable", "bank reconciliation",
    "tax preparation", "cash flow", "operations management",
    "workflow optimization", "vendor management", "supply chain",
    "patient care", "clinical assessment", "treatment planning",
    "care coordination", "patient education", "medical records",
    "medical terminology", "infection control", "case management",
    "telehealth", "talent acquisition", "employee relations",
    "benefits administration", "performance management",
    "policy compliance", "candidate screening",
    "curriculum development", "lesson planning", "classroom management",
    "academic advising", "unit testing", "integration testing",
    "quality assurance", "test automation", "ms office", "ms access",
    "ms project", "ms visio", "spring boot", "node.js",
    "qlik sense", "structured data", "unstructured data",
    "kpi tracking", "performance metrics", "stakeholder communication",
    "cross-functional", "project tracking",
    "requirements gathering", "process improvement",
    "business analysis",
    # PROBLEM 2 FIX — multi-word skills that need whitelist approval
    "data storytelling", "data reliability", "schema design",
    "scope management", "requirements negotiation",
    "data quality testing", "data presentation", "insight generation",
    "etl pipelines", "stored procedures",
    # BUG 3/4/5 FIX
    "data interpretation", "presentation skills", "report writing",
}


def looks_like_skill(s: str) -> bool:
    s = (s or "").strip().lower()
    # BUG 2 FIX — single-char skills bypass length check
    if s in SINGLE_CHAR_SKILLS:
        return True
    if not (2 <= len(s) <= 40):
        return False
    if s in GENERIC_NOT_SKILLS:
        return False

    # Pre-approved compound skills bypass the generic-word / length filters,
    # since phrases like "project management" legitimately contain "management".
    is_approved = s in ALLOWED_MULTIWORD_SKILLS or s in COMMON_JOB_SKILLS

    # Reject zip-code-like content (starts with 5 digits)
    if re.match(r"^\d{5}", s):
        return False

    # Reject HTML entities, URL fragments, weird symbols
    if "&" in s or "http" in s or "www." in s or "&dollar" in s or "@" in s:
        return False

    # Reject location-only strings and common city/state tokens
    words = s.split()
    if len(words) == 1 and words[0] in US_LOCATION_WORDS:
        return False

    # Reject institution / company-name patterns
    if any(w in INSTITUTION_WORDS for w in words):
        return False

    if not re.search(r"[a-z0-9\+#]", s):
        return False
    if s.isdigit():
        return False

    if is_approved:
        return True

    if len(words) > 3:
        return False
    if len(words) >= 2 and any(w in _SENTENCE_WORDS for w in words):
        return False
    if len(words) >= 2 and any(w in GENERIC_NOT_SKILLS for w in words):
        return False
    # Multi-word phrases not in the whitelist are rejected — stops free-form
    # junk like "ga data analyst marietta" from slipping through.
    if len(words) >= 2:
        return False

    for pat in _NON_SKILL_PATTERNS:
        if re.search(pat, s):
            return False
    return True


def is_known_skill(s: str) -> bool:
    """Whitelist check: only accept if the phrase is in our dictionary or aliases."""
    s = (s or "").strip().lower()
    if not s:
        return False
    s = SKILL_ALIASES.get(s, s)
    return s in COMMON_JOB_SKILLS or s in ALLOWED_MULTIWORD_SKILLS


def _split_slash_variants(tokens):
    """Split 'html/css' into ['html', 'css']; leave 'a/b testing' alone (pre-approved)."""
    out = []
    for t in tokens:
        t = str(t).strip().lower()
        if not t:
            continue
        if t in ALLOWED_MULTIWORD_SKILLS or t in SKILL_ALIASES or t in COMMON_JOB_SKILLS:
            out.append(t)
            continue
        if "/" in t:
            for part in t.split("/"):
                p = part.strip()
                if p:
                    out.append(p)
        else:
            out.append(t)
    return out


def apply_skill_aliases(skills):
    """Resolve aliases, drop junk, return sorted list."""
    out = set()
    for s in _split_slash_variants(skills):
        s = str(s).strip().lower()
        if not s:
            continue
        s = SKILL_ALIASES.get(s, s)
        out.add(s)
    return sorted([s for s in out if looks_like_skill(s)])


# =========================================================
# SKILL EXTRACTION
# =========================================================
def _scan_dictionary(text_lower: str, found: set, raw_text: str = ""):
    """Longest-first scan so 'data analysis' wins over 'analysis'.

    PROBLEM 1 FIX — also scans the raw (original-case) text with IGNORECASE
    for acronym skills (ETL, SQL, KPI...) that may appear all-caps in JDs.
    """
    dict_items = sorted(COMMON_JOB_SKILLS, key=len, reverse=True)
    for sk in dict_items:
        # BUG 2 FIX — single-char skills need stricter matching (list separators only)
        if len(sk) == 1 and sk in SINGLE_CHAR_SKILLS:
            pattern = r'(?:^|[,;:\s/(\[])' + re.escape(sk) + r'(?:$|[,;:\s/)\]])'
            if re.search(pattern, text_lower):
                found.add(sk)
        else:
            if re.search(r"(?<!\w)" + re.escape(sk) + r"(?!\w)", text_lower):
                found.add(sk)

    for alias, canon in SKILL_ALIASES.items():
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text_lower):
            found.add(canon)

    # PROBLEM 1 FIX — case-insensitive acronym pass on ORIGINAL raw text
    if raw_text:
        for ac in ACRONYM_SKILLS:
            if re.search(r"(?<!\w)" + re.escape(ac) + r"(?!\w)", raw_text, flags=re.IGNORECASE):
                canon = SKILL_ALIASES.get(ac, ac)
                if canon in COMMON_JOB_SKILLS or canon in ALLOWED_MULTIWORD_SKILLS:
                    found.add(canon)


def _extract_from_bullet_points(raw_text: str, found: set):
    """PROBLEM 2 FIX — split text on newlines/periods/semicolons/bullet chars
    and run the dictionary scan on each chunk. This catches multi-word skills
    that appear mid-bullet and wouldn't surface in a flat whole-text scan."""
    text = clean_for_matching(str(raw_text))
    # \u2022 = •, \u25cf = ●, \u25cb = ○, \u2013 = –, \u2014 = —
    chunks = re.split(r"[.;\u2022\u25cf\u25cb\u2013\u2014\n\r]", text)
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 5:
            continue
        _scan_dictionary(chunk, found)


def _extract_from_colon_lines(raw_text: str, found: set):
    """
    Parse lines like:
        Programming: Python, SQL, Java
        Data Analysis & Visualization: Pandas, NumPy, Power BI, Excel
    Splits on FIRST colon, treats right side as comma-separated skills.
    Strips parenthetical notes like (S3, IAM, Rekognition).
    Also splits on slash so 'HTML/CSS' becomes 'html' + 'css'.
    """
    for line in str(raw_text).splitlines():
        if ":" not in line:
            continue
        left, _, right = line.partition(":")
        # Drop parenthetical notes
        right = re.sub(r"\(.*?\)", "", right)
        # Split on commas/semicolons/pipes first, then slashes inside each token
        parts = re.split(r",|;|\|", right)
        tokens = []
        for p in parts:
            p = p.strip(" .:-").lower()
            if not p:
                continue
            # Keep pre-approved multi-word or a/b testing intact
            if p in ALLOWED_MULTIWORD_SKILLS or p in COMMON_JOB_SKILLS or p in SKILL_ALIASES:
                tokens.append(p)
                continue
            if "/" in p:
                tokens.extend([x.strip() for x in p.split("/") if x.strip()])
            else:
                tokens.append(p)
        for p in tokens:
            # URL / institution / too-long rejects
            if re.match(r"^https?://", p) or re.match(r"^www\.", p):
                continue
            if any(w in INSTITUTION_WORDS for w in p.split()):
                continue
            p = SKILL_ALIASES.get(p, p)
            if p in COMMON_JOB_SKILLS or p in ALLOWED_MULTIWORD_SKILLS:
                if looks_like_skill(p):
                    found.add(p)


def _extract_from_requirement_phrases(raw_text: str, found: set):
    """Pull skills from 'experience with X, Y, Z' type constructions.

    Strictly whitelisted: only accepts tokens that exist in COMMON_JOB_SKILLS
    or SKILL_ALIASES. Free-form nouns from the page are dropped.
    """
    # Strip parentheticals like "AWS (S3, IAM, Rekognition)" first so the
    # words inside don't leak as fake skills.
    stripped = re.sub(r"\(.*?\)", " ", str(raw_text))
    text = clean_for_matching(stripped)
    patterns = [
        r"(?:experience with|proficiency in|proficient in|knowledge of|"
        r"expertise in|skilled in|hands[- ]on experience with|"
        r"experience using|working knowledge of|familiarity with|"
        r"familiar with)\s+([a-zA-Z0-9\+\#\/\-\s,]{3,300})",
        r"(?:required skills?|preferred skills?|technical skills?|"
        r"core skills?|key skills?)[:\s]+([a-zA-Z0-9\+\#\/\-\s,]{3,350})",
        r"(?:must have|should have|nice to have)\s+([a-zA-Z0-9\+\#\/\-\s,]{3,250})",
        r"(?:tools?|technologies?|stack|platforms?)[:\s]+([a-zA-Z0-9\+\#\/\-\s,]{3,250})",
        r"(?:using|use of|build using|reports using|develop and maintain|"
        r"maintain)\s+([a-zA-Z0-9\+\#\/\-\s,]{3,250})",
    ]
    for pat in patterns:
        for m in re.findall(pat, text, flags=re.IGNORECASE):
            parts = re.split(r",|;|/|\band\b|\bor\b", m)
            for p in parts:
                p = p.strip(" .:-").lower()
                if not p:
                    continue
                p = SKILL_ALIASES.get(p, p)
                # Whitelist: only keep if it's a known skill in the dictionary
                if (p in COMMON_JOB_SKILLS or p in ALLOWED_MULTIWORD_SKILLS) \
                        and looks_like_skill(p):
                    found.add(p)


def extract_skills_from_text(text: str) -> list:
    """
    Resume skill extractor. Runs three methods and merges.
      M1: full text scan against skill dictionary
      M2: colon-format line parser (Programming: X, Y, Z)
      M3: requirement phrase extractor

    Strips URLs, institution names, and multi-word junk before scanning.
    """
    if not text or not str(text).strip():
        return []

    # Strip URLs and www.* addresses so they can't leak as "skills"
    clean_src = re.sub(r"https?://\S+", " ", str(text))
    clean_src = re.sub(r"www\.\S+", " ", clean_src)

    found = set()
    lower = clean_for_matching(clean_src)

    _scan_dictionary(lower, found, raw_text=clean_src)
    _extract_from_colon_lines(clean_src, found)
    _extract_from_requirement_phrases(clean_src, found)
    _extract_from_bullet_points(clean_src, found)  # PROBLEM 2 FIX

    # Final whitelist gate — resume skills MUST be in the known dictionary
    # or pre-approved multi-word set. This kills 'mar augusthinose college',
    # 'toggl track', 'red hat academy', 'html concepts data cleaning', etc.
    clean = set()
    for s in found:
        s = SKILL_ALIASES.get(s, s)
        if (s in COMMON_JOB_SKILLS or s in ALLOWED_MULTIWORD_SKILLS) and looks_like_skill(s):
            clean.add(s)
    return sorted(clean)


def extract_exact_job_skills(title: str, text: str) -> list:
    """
    STRICT whitelist job-side extractor.

    Unlike extract_skills_from_text (which is permissive for resumes), this
    ONLY accepts phrases that are already in COMMON_JOB_SKILLS or
    ALLOWED_MULTIWORD_SKILLS. Garbage from scraped pages (EEO boilerplate,
    zip codes, city names, recruiter chrome) cannot leak through.
    """
    if not text or not str(text).strip():
        return []

    full = f"{title or ''}\n{text}"
    found = set()
    lower = clean_for_matching(full)

    # Longest-first dictionary scan + PROBLEM 1 FIX acronym pass on raw text
    _scan_dictionary(lower, found, raw_text=full)

    # Alias resolution (handled inside _scan_dictionary already for aliases
    # that appear verbatim; make sure we run the alias resolver too)
    for alias, canon in SKILL_ALIASES.items():
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", lower):
            if canon in COMMON_JOB_SKILLS or canon in ALLOWED_MULTIWORD_SKILLS:
                found.add(canon)

    # Requirement-phrase extractor (already whitelisted)
    _extract_from_requirement_phrases(full, found)

    # PROBLEM 2 FIX — bullet-point / sentence-chunk extractor
    _extract_from_bullet_points(full, found)

    # Final whitelist gate
    clean = set()
    for s in found:
        s = SKILL_ALIASES.get(s, s)
        if (s in COMMON_JOB_SKILLS or s in ALLOWED_MULTIWORD_SKILLS) and looks_like_skill(s):
            clean.add(s)
    return sorted(clean)


# =========================================================
# RESUME READING
# =========================================================
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
        pieces = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages[:10]:
                t = page.extract_text() or ""
                if t.strip():
                    pieces.append(t)
        # Keep newlines so the colon-format parser still works
        raw = "\n".join(pieces)
        raw = raw.replace("\xa0", " ")
        return raw

    return ""


def extract_resume_sections(resume_text: str) -> dict:
    text = normalize_text(resume_text)
    low = text.lower()

    def grab(pattern):
        m = re.search(pattern, low, flags=re.IGNORECASE | re.DOTALL)
        return normalize_text(m.group(2)) if m else ""

    skills_text = grab(
        r"(skills|technical skills|core skills)(.*?)(experience|projects|education|certifications|$)"
    )
    experience_text = grab(
        r"(experience|work experience|projects|project experience)(.*?)(education|certifications|skills|$)"
    )
    summary_text = grab(
        r"(summary|profile|objective)(.*?)(skills|experience|education|$)"
    )

    return {
        "skills_text": skills_text,
        "experience_text": experience_text,
        "summary_text": summary_text,
    }


def build_focus_text(desired_role: str, user_skills: list, resume_sections: dict) -> str:
    skills_part = " ".join(user_skills[:40])
    experience_part = resume_sections.get("experience_text", "")[:1200]
    summary_part = resume_sections.get("summary_text", "")[:500]
    return normalize_text(f"{desired_role} {skills_part} {summary_part} {experience_part}")


# =========================================================
# FULL JOB DESCRIPTION - BEAUTIFULSOUP SCRAPER
# This is the ROOT PROBLEM the project solves:
# Adzuna's free API returns a ~150 char snippet. We fetch the
# actual page and pull the full text so skill extraction is real.
# =========================================================
_NOISE_CLASS_RE = re.compile(
    r"footer|nav|header|cookie|sidebar|similar|related|recommend|"
    r"subscribe|salary|stats|newsletter|alert|social|share|login|"
    r"register|breadcrumb|banner|advert|ad-container|promo|"
    r"eeo|legal|disclaimer",
    re.IGNORECASE,
)

# BUG 1A FIX — expanded to match Adzuna and more generic container class names.
_DESC_CONTAINER_RE = re.compile(
    r"job.?desc|description|job.?detail|posting|job.?content|vacancy|"
    r"job-body|job_body|details-content|"
    r"adp-body|richTextArea|job-adp|listing.?body|job.?text|"
    r"position.?desc|role.?desc|posting.?body",
    re.IGNORECASE,
)

_SECTION_HEADING_RE = re.compile(
    r"responsibilities|requirements|qualifications|skills?\s+required|"
    r"what\s+you|required\s+skills|preferred\s+skills|technical\s+skills|"
    r"key\s+skills|minimum\s+qualifications|preferred\s+qualifications|"
    r"you\s+will\s+need|what\s+we.?re\s+looking\s+for|duties|essential",
    re.IGNORECASE,
)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_full_job_description(url: str) -> str:
    """
    Fetch the real job page and return visible text from job-description
    sections only. Skips nav / footer / EEO disclaimers / cookie banners.
    """
    if not url or not str(url).strip():
        return ""

    # BUG 6 FIX — scraper retry with different UA + terminal diagnostics
    try:
        domain = urlparse(url).netloc.replace("www.", "").lower()
    except Exception:
        domain = ""
    if any(d in domain for d in ("indeed.com", "linkedin.com", "glassdoor.com")):
        print(f"[SCRAPER BLOCKED] {domain} blocks automated scraping")

    resp = None
    last_status = None
    last_error = None
    for attempt, hdrs in enumerate((BROWSER_HEADERS, ALTERNATE_HEADERS), start=1):
        try:
            r = requests.get(url, headers=hdrs, timeout=15, allow_redirects=True)
            last_status = r.status_code
            if r.status_code == 200:
                resp = r
                break
        except Exception as e:
            last_error = repr(e)
    if resp is None:
        print(f"[SCRAPER FAIL] {url} — Status: {last_status}, Error: {last_error}")
        return ""

    try:
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception:
            return ""

    # Step 1 — strip structural noise
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "noscript", "iframe"]):
        tag.decompose()

    # Step 2 — strip by class/id patterns that are noise
    for tag in soup.find_all(attrs={"class": _NOISE_CLASS_RE}):
        tag.decompose()
    for tag in soup.find_all(attrs={"id": _NOISE_CLASS_RE}):
        tag.decompose()

    # Step 3 — prefer a dedicated job-description container
    desc_text = ""
    container = soup.find(attrs={"class": _DESC_CONTAINER_RE}) or \
                soup.find(attrs={"id": _DESC_CONTAINER_RE})
    if container:
        desc_text = container.get_text("\n", strip=True)
    # BUG 5 FIX — per-step diagnostics
    print(f"[SCRAPER] Step 3 container: {'FOUND' if container else 'MISS'} ({len(desc_text)} chars)")

    # BUG 1B FIX — Adzuna-specific container fallback (adp-body, richTextArea,
    # etc.). Adzuna details pages use these class names, which don't match the
    # generic regex above cleanly in every case.
    if not desc_text or len(desc_text) < 100:
        for cls_pattern in ["adp-body", "ui-richTextArea", "adp-template", "job-adp"]:
            az_container = soup.find(attrs={"class": re.compile(cls_pattern, re.IGNORECASE)})
            if az_container:
                candidate = az_container.get_text("\n", strip=True)
                if len(candidate) > 50:
                    desc_text = candidate
                    print(f"[SCRAPER] Step 3b Adzuna container '{cls_pattern}': FOUND ({len(desc_text)} chars)")
                    break
        else:
            print("[SCRAPER] Step 3b Adzuna container: MISS")

    # Step 4 — section-heading walk (responsibilities / requirements / ...)
    chunks = []
    if not desc_text:
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "strong", "b"]):
            htext = heading.get_text(" ", strip=True)
            if not htext or not _SECTION_HEADING_RE.search(htext):
                continue
            # Walk siblings until we hit the next heading
            for sib in heading.find_next_siblings():
                if sib.name in ("h1", "h2", "h3", "h4"):
                    break
                txt = sib.get_text(" ", strip=True)
                if txt:
                    chunks.append(txt)
        if chunks:
            desc_text = "\n".join(chunks)
    # BUG 5 FIX
    print(f"[SCRAPER] Step 4 headings: {len(chunks)} chunks found")

    # Step 5 — final fallback: <p> and <li> only (not <div>/<span>)
    parts = []
    if not desc_text:
        for t in soup.find_all(["p", "li"]):
            txt = t.get_text(" ", strip=True)
            if txt and 15 <= len(txt) <= 2000:
                parts.append(txt)
        seen = set()
        out = []
        for p in parts:
            k = p[:120]
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
        desc_text = "\n".join(out)
    # BUG 5 FIX
    print(f"[SCRAPER] Step 5 p/li: {len(parts)} parts found")

    # BUG 1C FIX — nuclear fallback: if nothing else worked, grab the body text.
    # Trim the first ~200 chars (usually nav/header leftovers) and cap the total.
    if not desc_text or len(desc_text) < 100:
        body = soup.find("body")
        if body:
            full_body_text = body.get_text("\n", strip=True)
            if len(full_body_text) > 300:
                desc_text = full_body_text[200:8000]
                print(f"[SCRAPER] Step 6 body fallback: {len(desc_text)} chars")
            else:
                print(f"[SCRAPER] Step 6 body fallback: SKIP (only {len(full_body_text)} chars)")

    desc_text = re.sub(r"\s+", " ", desc_text).strip()
    # BUG 5 FIX — final summary line
    print(f"[SCRAPER] Final: {len(desc_text)} chars extracted from {url}")
    return desc_text[:8000]


# =========================================================
# JOB POSTING CLEANERS
# =========================================================
def is_direct_company_job(link, description, title) -> bool:
    link = normalize_text(link).lower()
    description = normalize_text(description).lower()
    title = normalize_text(title).lower()

    if not link:
        return False

    domain = urlparse(link).netloc.replace("www.", "").lower()
    if any(bad in domain for bad in BAD_DOMAINS):
        return False

    full_text = f"{title} {description}"
    if any(word in full_text for word in BAD_KEYWORDS):
        return False
    if any(word in title for word in BAD_TITLE_KEYWORDS):
        return False
    return True


def infer_industry(row) -> str:
    text = clean_for_matching(
        f"{row.get('job_title', '')} {row.get('job_description', '')}"
    )
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


def normalize_level(title: str = "", desc: str = "") -> str:
    t = clean_for_matching(f"{title} {desc}")
    if "intern" in t or "internship" in t:
        return "Intern"
    if "entry" in t or "junior" in t or "jr " in t or "new grad" in t or "recent graduate" in t:
        return "Entry"
    if "senior" in t or " lead " in t or "principal" in t or "director" in t or "manager" in t:
        return "Senior"
    return "Mid"


def normalize_job_type(contract_type: str, contract_time: str, title: str, desc: str) -> str:
    t = clean_for_matching(f"{contract_type} {contract_time} {title} {desc}")
    if "intern" in t or "internship" in t:
        return "Internship"
    if "part-time" in t or "part time" in t or contract_time == "part_time":
        return "Part-time"
    if "contract" in t or "temporary" in t or contract_type == "contract":
        return "Contract"
    if "full-time" in t or "full time" in t or contract_time == "full_time":
        return "Full-time"
    return "Other"


def deduplicate_jobs(df):
    if df.empty:
        return df
    temp = df.copy()
    for col, out_col in [
        ("job_title", "job_title_clean"),
        ("company", "company_clean"),
        ("job_location", "location_clean"),
    ]:
        temp[out_col] = temp[col].fillna("").astype(str).str.lower().str.strip()
        temp[out_col] = temp[out_col].str.replace(r"\s+", " ", regex=True)

    if "job_link" in temp.columns:
        temp["job_link_clean"] = temp["job_link"].fillna("").astype(str).str.lower().str.strip()
        temp = temp.drop_duplicates(subset=["job_link_clean"], keep="first")

    temp = temp.drop_duplicates(
        subset=["job_title_clean", "company_clean", "location_clean"], keep="first"
    )

    return temp.drop(
        columns=["job_title_clean", "company_clean", "location_clean", "job_link_clean"],
        errors="ignore",
    ).reset_index(drop=True)


# =========================================================
# SCORING
# =========================================================
def skill_match(user_skills, job_skills):
    us = set(apply_skill_aliases(user_skills))
    js = set(apply_skill_aliases(job_skills))

    matched = sorted(us & js)
    missing = sorted(js - us)
    extra = sorted(us - js)

    raw_score = 0.0 if not js else len(matched) / len(js)

    # HONESTY RULE: if the job only yielded a handful of skills, the
    # description was almost certainly truncated. Don't pretend it's a
    # confident 100%. Cap at 50%.
    capped = False
    if len(js) < MIN_JOB_SKILLS_FOR_FULL_SCORE:
        if raw_score > 0.50:
            raw_score = 0.50
            capped = True

    return {
        "score": raw_score,
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "job_skills_count": len(js),
        "capped": capped,
    }


def role_responsibilities_score(desired_role: str, job_title: str, job_desc: str):
    """
    Score 3 - Target Role Match.
    Compare the job text against the typical responsibilities of the
    desired role rather than naive title matching.
    """
    role_key = (desired_role or "").strip().lower()
    resp = TYPICAL_ROLE_RESPONSIBILITIES.get(role_key, [])

    blob = clean_for_matching(f"{job_title} {job_desc}")
    if not resp:
        # Fallback - count how many role words appear in title/desc
        words = [w for w in role_key.split() if len(w) > 2]
        if not words:
            return 0.0, []
        hits = [w for w in words if w in blob]
        return min(len(hits) / len(words), 1.0), hits

    hits = [r for r in resp if r in blob]
    score = len(hits) / len(resp)

    if role_key and role_key in clean_for_matching(job_title):
        score += 0.30

    return min(score, 1.0), hits


@st.cache_data(show_spinner=False)
def fit_tfidf(job_texts):
    vec = TfidfVectorizer(stop_words="english", max_features=40000, ngram_range=(1, 2))
    X = vec.fit_transform(job_texts)
    return vec, X


def build_job_text(df) -> pd.Series:
    return (
        df["job_title"].fillna("") + " | "
        + df["job_description"].fillna("") + " | "
        + df["job_skills"].fillna("") + " | "
        + df["company"].fillna("") + " | "
        + df["job_location"].fillna("")
    )


def build_match_reason(matched_count, total_job_skills, role_score, nlp_score, used_full_desc):
    parts = []
    if total_job_skills > 0:
        src = "full job description" if used_full_desc else "API snippet"
        parts.append(f"Matched {matched_count} of {total_job_skills} skills from {src}")
    else:
        parts.append("Limited skill extraction")

    if role_score >= 0.70:
        parts.append("Strong role responsibilities match")
    elif role_score >= 0.40:
        parts.append("Moderate role responsibilities match")
    else:
        parts.append("Weak role responsibilities match")

    if nlp_score >= 0.22:
        parts.append("Good profile-to-job similarity")
    elif nlp_score >= 0.10:
        parts.append("Moderate profile-to-job similarity")
    else:
        parts.append("Low profile-to-job similarity")

    return " | ".join(parts)


# =========================================================
# ADZUNA FETCH
# =========================================================
def fetch_live_jobs(query: str, max_pages: int = MAX_ADZUNA_PAGES) -> pd.DataFrame:
    if not APP_ID or not APP_KEY:
        st.error(
            "Missing Adzuna API credentials. Add ADZUNA_APP_ID and "
            "ADZUNA_APP_KEY to .streamlit/secrets.toml"
        )
        return pd.DataFrame()

    jobs = []
    progress = st.progress(0, text="Fetching live jobs...")

    for page in range(1, max_pages + 1):
        progress.progress(page / max_pages, text=f"Fetching page {page} of {max_pages}...")
        url = f"https://api.adzuna.com/v1/api/jobs/us/search/{page}"
        params = {
            "app_id": APP_ID,
            "app_key": APP_KEY,
            "results_per_page": RESULTS_PER_PAGE_API,
            "what": query,
            "content-type": "application/json",
        }

        try:
            r = requests.get(url, params=params, timeout=25)
            r.raise_for_status()
            data = r.json()
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

            category_label = ""
            if isinstance(job.get("category"), dict):
                category_label = job["category"].get("label", "")

            title = normalize_text(job.get("title", ""))
            desc = normalize_text(job.get("description", ""))
            link = normalize_text(job.get("redirect_url", ""))

            extracted = extract_exact_job_skills(title, desc)

            jobs.append({
                "job_title": title,
                "company": normalize_text(company_name),
                "job_location": normalize_text(location_name),
                "job_description": desc,
                "job_link": link,
                "job_skills": ", ".join(extracted),
                "job_skills_list": extracted,
                "category": normalize_text(category_label),
                "contract_type": normalize_text(job.get("contract_type", "")),
                "contract_time": normalize_text(job.get("contract_time", "")),
                "search_country": "United States",
                "salary_min": job.get("salary_min", None),
                "salary_max": job.get("salary_max", None),
                "source_page": page,
            })

    progress.empty()
    df = pd.DataFrame(jobs)
    if df.empty:
        return df
    df = deduplicate_jobs(df)
    df["job_text"] = build_job_text(df)
    return df.reset_index(drop=True)


# =========================================================
# SESSION STATE
# =========================================================
for key, default in [
    ("results_all_ranked", None),
    ("page_no", 1),
    ("jobs_analyzed_count", 0),
    ("selected_job_idx", 0),
    ("saved_jobs", []),
    ("full_desc_cache", {}),       # url -> full description text
    ("full_desc_skills", {}),      # url -> extracted skills list
]:
    if key not in st.session_state:
        st.session_state[key] = default


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 🔎 Search & Filters")

if APP_ID and APP_KEY:
    st.sidebar.success("Live Adzuna API connected")
else:
    st.sidebar.error("Adzuna API credentials missing")

target_industry_ui = st.sidebar.selectbox(
    "🧭 Target Industry",
    [
        "All", "Data & IT", "Healthcare", "Accounting & Finance",
        "Business & Operations", "Marketing", "Human Resources", "Education",
    ],
)

desired_role = st.sidebar.text_input(
    "🎯 Target role or profession",
    value="data analyst",
    placeholder="Example: data analyst, nurse practitioner, accountant",
)

manual_skills = st.sidebar.text_area(
    "✍️ Skills (comma-separated)",
    value="sql, python, excel, power bi",
    height=90,
)

remove_low_quality = st.sidebar.checkbox(
    "🧹 Remove low-quality / recruiter postings", value=True
)
only_usa = st.sidebar.checkbox("🇺🇸 Only USA jobs", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚖️ Scoring Weights")
nlp_weight   = st.sidebar.slider("Profile Similarity",     0.0, 1.0, 0.20, 0.05)
skill_weight = st.sidebar.slider("Required Skills Match",  0.0, 1.0, 0.50, 0.05)
role_weight  = st.sidebar.slider("Target Role Match",      0.0, 1.0, 0.30, 0.05)

total_weight = nlp_weight + skill_weight + role_weight
if total_weight == 0:
    nlp_weight, skill_weight, role_weight = 0.20, 0.50, 0.30
    total_weight = 1.0
nlp_weight   /= total_weight
skill_weight /= total_weight
role_weight  /= total_weight

st.sidebar.caption(
    f"Final = {nlp_weight:.2f}·profile + {skill_weight:.2f}·skills + {role_weight:.2f}·role"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛 Result Settings")

display_page_size = st.sidebar.selectbox("Jobs per page", [10, 20, 25, 50], index=0)

sort_option = st.sidebar.selectbox(
    "Sort results by",
    ["Best Match", "Skills Match", "Role Match", "Job Title (A-Z)", "Company (A-Z)"],
)

work_mode_option       = st.sidebar.selectbox("🏠 Work Mode",      ["All", "Remote", "Hybrid", "Onsite"])
career_level_option    = st.sidebar.selectbox("🎓 Career Level",   ["All", "Intern", "Entry", "Mid", "Senior"])
employment_type_option = st.sidebar.selectbox("💼 Employment Type", ["All", "Full-time", "Part-time", "Internship", "Contract", "Other"])

resume_file = st.sidebar.file_uploader("📄 Upload Resume", type=["pdf", "docx", "txt"])

debug_mode = st.sidebar.checkbox("🐞 Debug Mode (show extraction details)", value=False)

run_search = st.sidebar.button("Find Matching Jobs", type="primary")


# =========================================================
# RESUME / USER PROFILE
# =========================================================
resume_text = read_resume(resume_file)
resume_sections = (
    extract_resume_sections(resume_text)
    if resume_text.strip()
    else {"skills_text": "", "experience_text": "", "summary_text": ""}
)

typed_skills = apply_skill_aliases(
    [s.strip().lower() for s in re.split(r",|;|\|", manual_skills) if s.strip()]
)
resume_skills = extract_skills_from_text(resume_text) if resume_text.strip() else []
manual_detected_skills = (
    extract_skills_from_text(manual_skills) if manual_skills.strip() else []
)

user_skills = apply_skill_aliases(
    sorted(set(typed_skills + manual_detected_skills + resume_skills))
)
focused_user_text = build_focus_text(desired_role, user_skills, resume_sections)


# =========================================================
# TOP: RESUME FEEDBACK
# =========================================================
if resume_file and resume_text.strip():
    st.success(f"Resume loaded ✅ — {len(resume_text):,} characters extracted")
elif resume_file and not resume_text.strip():
    st.warning(
        "Resume uploaded, but text could not be extracted. "
        "DOCX or TXT usually works better than image-based PDFs."
    )

if resume_text.strip():
    with st.expander(
        f"📋 Skills detected from your resume ({len(resume_skills)} found) — click to verify"
    ):
        if resume_skills:
            st.write(", ".join(resume_skills))
        else:
            st.info("No skills detected. Try typing your skills manually in the sidebar.")

        if debug_mode:
            st.markdown("**Raw resume text (first 3000 chars):**")
            st.code(resume_text[:3000])
            st.markdown("**Skills/Experience/Summary sections detected:**")
            st.json({k: (v[:500] + "...") if len(v) > 500 else v
                     for k, v in resume_sections.items()})

if user_skills:
    st.markdown('<div class="section-title">✅ Detected Candidate Skills</div>', unsafe_allow_html=True)
    st.write(", ".join(user_skills[:SHOW_TOP_SKILLS]))
    with st.expander(f"Show all {len(user_skills)} detected skills (resume + typed)"):
        st.write(", ".join(user_skills))
else:
    st.info("Upload a resume or type skills in the sidebar to improve recommendations.")


# =========================================================
# SEARCH EXECUTION
# =========================================================
if run_search:
    if not desired_role.strip() and not user_skills and not resume_text.strip():
        st.warning("Please enter a target role, skills, or upload a resume first.")
        st.stop()

    with st.spinner("Fetching live jobs and ranking..."):
        query = desired_role.strip() or "jobs"
        raw_df = fetch_live_jobs(query=query, max_pages=MAX_ADZUNA_PAGES)

        if raw_df.empty:
            st.warning("No jobs were fetched from the API.")
            st.stop()

        fetched_count = len(raw_df)
        work = raw_df.copy()

        if remove_low_quality:
            work = work[work.apply(
                lambda r: is_direct_company_job(
                    r.get("job_link", ""),
                    r.get("job_description", ""),
                    r.get("job_title", ""),
                ),
                axis=1,
            )].copy()

        cleaned_count = len(work)
        if work.empty:
            st.warning(
                "All fetched jobs were removed by the cleaner. "
                "Uncheck the low-quality filter to see raw results."
            )
            st.stop()

        work["work_mode"]            = work.apply(infer_work_mode, axis=1)
        work["industry"]             = work.apply(infer_industry, axis=1)
        work["job_level_norm"]       = work.apply(
            lambda r: normalize_level(r.get("job_title", ""), r.get("job_description", "")),
            axis=1,
        )
        work["employment_type_norm"] = work.apply(
            lambda r: normalize_job_type(
                r.get("contract_type", ""), r.get("contract_time", ""),
                r.get("job_title", ""), r.get("job_description", ""),
            ),
            axis=1,
        )

        if only_usa:
            work = work[work["search_country"].astype(str).str.lower()
                        .str.contains(USA_PATTERNS, na=False)].copy()
        if work_mode_option != "All":
            work = work[work["work_mode"] == work_mode_option].copy()
        if career_level_option != "All":
            work = work[work["job_level_norm"] == career_level_option].copy()
        if employment_type_option != "All":
            work = work[work["employment_type_norm"] == employment_type_option].copy()
        if target_industry_ui != "All":
            work = work[work["industry"] == target_industry_ui].copy()

        if work.empty:
            st.warning("No jobs match your filters. Try broader filters.")
            st.stop()

        # --- TF-IDF / Profile similarity ---
        vec, X_jobs = fit_tfidf(work["job_text"].fillna("").astype(str).tolist())
        query_text = focused_user_text if focused_user_text.strip() else desired_role.strip()
        X_query = vec.transform([query_text])
        nlp_scores = cosine_similarity(X_query, X_jobs).flatten()

        skill_scores, matched_all, missing_all, extra_all = [], [], [], []
        role_scores, role_hits_all = [], []
        job_skills_all, capped_all = [], []

        for _, row in work.iterrows():
            js = row.get("job_skills_list") or []
            if not js:
                js = extract_exact_job_skills(
                    row.get("job_title", ""), row.get("job_description", "")
                )
            js = apply_skill_aliases(js)

            sm = skill_match(user_skills, js)
            skill_scores.append(sm["score"])
            matched_all.append(sm["matched"])
            missing_all.append(sm["missing"])
            extra_all.append(sm["extra"])
            capped_all.append(sm["capped"])
            job_skills_all.append(js)

            rs, rh = role_responsibilities_score(
                desired_role, row.get("job_title", ""), row.get("job_description", "")
            )
            role_scores.append(rs)
            role_hits_all.append(rh)

        results = work.copy().reset_index(drop=True)
        results["nlp_score"]           = nlp_scores
        results["skill_score"]         = skill_scores
        results["role_score"]          = role_scores
        results["role_hits"]           = role_hits_all
        results["matched_skills"]      = matched_all
        results["missing_skills"]      = missing_all
        results["extra_user_skills"]   = extra_all
        results["expanded_job_skills"] = job_skills_all
        results["skill_score_capped"]  = capped_all
        results["used_full_desc"]      = False

        results["final_score"] = (
            nlp_weight   * results["nlp_score"]
            + skill_weight * results["skill_score"]
            + role_weight  * results["role_score"]
        ).clip(upper=1.0)

        results["match_reason"] = results.apply(
            lambda r: build_match_reason(
                len(r["matched_skills"]) if isinstance(r["matched_skills"], list) else 0,
                len(r["expanded_job_skills"]) if isinstance(r["expanded_job_skills"], list) else 0,
                r["role_score"], r["nlp_score"], r["used_full_desc"],
            ),
            axis=1,
        )

        if sort_option == "Best Match":
            results = results.sort_values("final_score", ascending=False)
        elif sort_option == "Skills Match":
            results = results.sort_values("skill_score", ascending=False)
        elif sort_option == "Role Match":
            results = results.sort_values("role_score", ascending=False)
        elif sort_option == "Job Title (A-Z)":
            results = results.sort_values("job_title", ascending=True)
        elif sort_option == "Company (A-Z)":
            results = results.sort_values("company", ascending=True)
        results = results.reset_index(drop=True)

        results.attrs["fetched_count"] = fetched_count
        results.attrs["cleaned_count"] = cleaned_count

        st.session_state["results_all_ranked"] = results
        st.session_state["jobs_analyzed_count"] = len(work)
        st.session_state["page_no"] = 1
        st.session_state["selected_job_idx"] = 0
        # Reset full-desc caches when a new search runs
        st.session_state["full_desc_cache"] = {}
        st.session_state["full_desc_skills"] = {}


# =========================================================
# RESULTS DISPLAY
# =========================================================
results_all_ranked = st.session_state.get("results_all_ranked")

if results_all_ranked is None or results_all_ranked.empty:
    st.info("Set your filters and click **Find Matching Jobs** in the sidebar to load live jobs.")
    st.stop()

fetched_count = results_all_ranked.attrs.get("fetched_count", len(results_all_ranked))
cleaned_count = results_all_ranked.attrs.get("cleaned_count", len(results_all_ranked))
analyzed_count = st.session_state.get("jobs_analyzed_count", len(results_all_ranked))

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Fetched</div><div class="metric-value">{fetched_count:,}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">After cleaning</div><div class="metric-value">{cleaned_count:,}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Analyzed</div><div class="metric-value">{analyzed_count:,}</div></div>', unsafe_allow_html=True)
with m4:
    best = format_pct(results_all_ranked["final_score"].max())
    st.markdown(f'<div class="metric-card"><div class="metric-title">Top score</div><div class="metric-value">{best}</div></div>', unsafe_allow_html=True)

st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e5e7eb;border-radius:14px;padding:0.9rem 1rem;margin:0.8rem 0;">
<b>How your Overall Match is calculated:</b><br>
Overall = <b>{nlp_weight*100:.0f}%</b> Profile Similarity (resume ↔ job text)
 + <b>{skill_weight*100:.0f}%</b> Skills Match (your skills vs the skills extracted from the job)
 + <b>{role_weight*100:.0f}%</b> Role Match (job text vs typical responsibilities of a <i>{desired_role}</i>).<br>
If fewer than {MIN_JOB_SKILLS_FOR_FULL_SCORE} skills could be extracted from a posting,
the Skills Match is capped at 50% and a warning is shown. Click <b>View Details</b>
on any job to fetch the full page and recompute with complete data.
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
with nav3:
    if st.button("Next ➡", disabled=(current_page == total_pages), key="next_page"):
        st.session_state["page_no"] = min(total_pages, current_page + 1)
        st.session_state["selected_job_idx"] = 0
        st.rerun()
with nav2:
    st.caption(f"Page {current_page} of {total_pages}")


# =========================================================
# LEFT = LIST, RIGHT = DETAIL
# =========================================================
left_col, right_col = st.columns([1.05, 1.35], gap="large")

selected_local_idx = min(st.session_state["selected_job_idx"], max(0, len(page_df) - 1))

with left_col:
    st.markdown("### Job List")

    for idx in range(len(page_df)):
        row        = page_df.iloc[idx]
        global_idx = start_idx + idx
        is_saved   = global_idx in st.session_state["saved_jobs"]
        is_selected = (idx == selected_local_idx)

        with st.container(border=True):
            if is_selected:
                st.markdown(
                    "<span style='background:#2563eb;color:#fff;"
                    "padding:0.12rem 0.6rem;border-radius:999px;"
                    "font-size:0.75rem;font-weight:700;'>"
                    f"👁 VIEWING — Job #{global_idx + 1}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(f"Job #{global_idx + 1}")

            st.markdown(f"**{row['job_title']}**")
            st.caption(f"{row['company']}  ·  {row['job_location']}")
            st.markdown(
                f"`{row['work_mode']}` &nbsp; "
                f"`{row['industry']}` &nbsp; "
                f"`{row['job_level_norm']}` &nbsp; "
                f"`{row['employment_type_norm']}`",
                unsafe_allow_html=True,
            )
            # Use shared percentage helper so left/right panels render identically.
            st.caption(
                f"Overall: **{format_pct(row['final_score'])}**  ·  "
                f"Skills: **{format_pct(row['skill_score'])}**  ·  "
                f"Role: **{format_pct(row['role_score'])}**"
            )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("👁 View Details", key=f"view_{global_idx}"):
                st.session_state["selected_job_idx"] = idx
                st.session_state[f"__trigger_full_fetch_{global_idx}"] = True
                st.rerun()
        with c2:
            if is_saved:
                if st.button("★ Unsave", key=f"unsave_{global_idx}"):
                    st.session_state["saved_jobs"].remove(global_idx)
                    st.rerun()
            else:
                if st.button("☆ Save", key=f"save_{global_idx}"):
                    st.session_state["saved_jobs"].append(global_idx)
                    st.rerun()

    if st.session_state["saved_jobs"]:
        st.markdown("### ⭐ Saved Jobs")
        st.write(f"Saved count: {len(st.session_state['saved_jobs'])}")


# =========================================================
# RIGHT: JOB DETAIL
# =========================================================
with right_col:
    if len(page_df) == 0:
        st.info("No jobs to display on this page.")
    else:
        selected_row = page_df.iloc[selected_local_idx]
        selected_global_idx = start_idx + selected_local_idx
        job_link = str(selected_row.get("job_link", "") or "")
        trigger_key = f"__trigger_full_fetch_{selected_global_idx}"

        st.markdown(
            f'<div class="viewing-tag">👁 Currently Viewing: Job #{selected_global_idx + 1}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="job-detail-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="job-detail-title">{safe(selected_row["job_title"])}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="job-meta"><strong>Company:</strong> {safe(selected_row["company"])}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="job-meta"><strong>Location:</strong> {safe(selected_row["job_location"])}</div>', unsafe_allow_html=True)

        if selected_global_idx in st.session_state["saved_jobs"]:
            st.markdown('<div class="saved-pill">★ Saved</div>', unsafe_allow_html=True)

        # BUG 7 FIX — single-line markdown so Streamlit doesn't render as code block
        badges_html = f'<div class="job-meta" style="margin-top:0.5rem;"><span class="badge badge-blue">{safe(selected_row["work_mode"])}</span><span class="badge badge-purple">{safe(selected_row["industry"])}</span><span class="badge badge-orange">{safe(selected_row["job_level_norm"])}</span><span class="badge badge-green">{safe(selected_row["employment_type_norm"])}</span></div>'
        st.markdown(badges_html, unsafe_allow_html=True)

        # ----- snapshot initial (API-snippet-based) values -----
        api_snippet = str(selected_row.get("job_description", "") or "")
        initial_matched = selected_row["matched_skills"] if isinstance(selected_row["matched_skills"], list) else []
        initial_missing = selected_row["missing_skills"] if isinstance(selected_row["missing_skills"], list) else []
        initial_extra   = selected_row["extra_user_skills"] if isinstance(selected_row["extra_user_skills"], list) else []
        initial_job_skills = selected_row["expanded_job_skills"] if isinstance(selected_row["expanded_job_skills"], list) else []
        initial_skill_score = float(selected_row["skill_score"])
        initial_role_score  = float(selected_row["role_score"])
        initial_nlp_score   = float(selected_row["nlp_score"])
        initial_final_score = float(selected_row["final_score"])
        initial_capped      = bool(selected_row.get("skill_score_capped", False))

        # ----- Full description fetch via BeautifulSoup -----
        full_desc = st.session_state["full_desc_cache"].get(job_link, "")
        used_full_desc = bool(full_desc)
        fetched_full_desc_this_run = False

        if st.session_state.pop(trigger_key, False) and job_link and not full_desc:
            with st.spinner("Loading full job description from the source website..."):
                fetched = fetch_full_job_description(job_link)
                if fetched and len(fetched) > len(api_snippet) + 50:
                    st.session_state["full_desc_cache"][job_link] = fetched
                    full_desc = fetched
                    used_full_desc = True
                    fetched_full_desc_this_run = True
                else:
                    st.session_state["full_desc_cache"][job_link] = ""
                    used_full_desc = False

        # ----- Recalculate with full text when available -----
        if used_full_desc:
            full_job_skills = st.session_state["full_desc_skills"].get(job_link)
            if full_job_skills is None:
                full_job_skills = extract_exact_job_skills(
                    selected_row["job_title"], full_desc
                )
                st.session_state["full_desc_skills"][job_link] = full_job_skills

            sm = skill_match(user_skills, full_job_skills)
            display_job_skills = full_job_skills
            display_matched    = sm["matched"]
            display_missing    = sm["missing"]
            display_extra      = sm["extra"]
            display_skill_score = sm["score"]
            display_capped      = sm["capped"]

            role_sc, role_hits = role_responsibilities_score(
                desired_role, selected_row["job_title"], full_desc
            )
            display_role_score = role_sc

            display_nlp_score = initial_nlp_score  # NLP stays (needs corpus refit)
            display_final_score = min(
                nlp_weight * display_nlp_score
                + skill_weight * display_skill_score
                + role_weight * display_role_score,
                1.0,
            )

            st.markdown(
                '<div class="success-banner">✅ Full job description loaded — skill data updated from the complete page.</div>',
                unsafe_allow_html=True,
            )

            # Sync recalculated scores back to the master DataFrame so the left
            # Job List panel shows the same numbers as the right Detail panel
            # on the next rerun (otherwise the left shows stale API-snippet scores).
            master_df = st.session_state["results_all_ranked"]
            master_idx = selected_global_idx
            if master_idx < len(master_df):
                master_df.at[master_idx, "skill_score"]         = display_skill_score
                master_df.at[master_idx, "role_score"]          = display_role_score
                master_df.at[master_idx, "final_score"]         = display_final_score
                master_df.at[master_idx, "matched_skills"]      = display_matched
                master_df.at[master_idx, "missing_skills"]      = display_missing
                master_df.at[master_idx, "extra_user_skills"]   = display_extra
                master_df.at[master_idx, "expanded_job_skills"] = display_job_skills
                master_df.at[master_idx, "skill_score_capped"]  = display_capped
                master_df.at[master_idx, "used_full_desc"]      = True
                master_df.at[master_idx, "match_reason"]        = build_match_reason(
                    len(display_matched), len(display_job_skills),
                    display_role_score, display_nlp_score, True,
                )

                # The left list has already rendered in this run. After a full
                # description changes the authoritative scores, rerun once so
                # both panels read the same stored display values immediately.
                if fetched_full_desc_this_run:
                    st.rerun()
        else:
            display_job_skills  = initial_job_skills
            display_matched     = initial_matched
            display_missing     = initial_missing
            display_extra       = initial_extra
            display_skill_score = initial_skill_score
            display_role_score  = initial_role_score
            display_nlp_score   = initial_nlp_score
            display_final_score = initial_final_score
            display_capped      = initial_capped
            role_hits           = selected_row.get("role_hits", []) if isinstance(selected_row.get("role_hits", []), list) else []

            if job_link:
                if job_link in st.session_state["full_desc_cache"] and not st.session_state["full_desc_cache"][job_link]:
                    # BUG 1D FIX — show a Retry button next to the warning.
                    # @st.cache_data caches empty results for 1 hour; the retry
                    # clears that so a fresh fetch attempt can run.
                    col_warn, col_retry = st.columns([3, 1])
                    with col_warn:
                        st.markdown(
                            '<div class="warning-banner">⚠️ Could not load the full description from the source site — showing results based on the API snippet only.</div>',
                            unsafe_allow_html=True,
                        )
                    with col_retry:
                        if st.button("🔄 Retry", key=f"retry_{selected_global_idx}"):
                            st.cache_data.clear()
                            st.session_state["full_desc_cache"].pop(job_link, None)
                            st.session_state["full_desc_skills"].pop(job_link, None)
                            st.session_state[f"__trigger_full_fetch_{selected_global_idx}"] = True
                            st.rerun()

        # ----- Honesty warning -----
        if display_capped or len(display_job_skills) < MIN_JOB_SKILLS_FOR_FULL_SCORE:
            st.markdown(
                f'<div class="warning-banner">⚠️ Only {len(display_job_skills)} skill(s) could be extracted — the API description is likely incomplete. Skills Match is capped at 50%. Click <b>Open Job Posting</b> or wait for the full description to load.</div>',
                unsafe_allow_html=True,
            )

        # ----- Metric boxes -----
        # BUG 2 FIX — format_pct is defined at module level so both the left
        # Job List panel and the right Detail panel share identical formatting.
        s1, s2, s3, s4 = st.columns(4)
        with s1: st.metric("Overall Match",     format_pct(display_final_score))
        with s2: st.metric("Profile Similarity", format_pct(display_nlp_score))
        with s3: st.metric("Skills Match",       format_pct(display_skill_score))
        with s4: st.metric("Role Match",         format_pct(display_role_score))

        # BUG 3 FIX — Prominent warning when scoring is based on the API snippet
        # and the skill count is suspiciously low (< 8). Without the full page,
        # an apparent 100% can be misleading.
        if (not used_full_desc) and len(display_job_skills) < 8:
            st.markdown(
                f'<div class="warning-banner">⚠️ <b>Match based on incomplete data.</b> '
                f'Only {len(display_job_skills)} skill(s) were extracted from the API snippet — '
                f'real postings typically list 15–25+. The match percentage above is '
                f'<u>not reliable</u> until the full description is loaded. '
                f'Click <b>Retry</b> below or <b>Open Job Posting</b> to verify.</div>',
                unsafe_allow_html=True,
            )

        source_label = "full job description" if used_full_desc else "API snippet only"
        reason = build_match_reason(
            len(display_matched), len(display_job_skills),
            display_role_score, display_nlp_score, used_full_desc,
        )
        st.info(f"**Why ranked here:** {reason}  \n*(Based on {source_label})*")

        # PROBLEM 3 FIX — Collapsible score-math expander (always present)
        with st.expander("📊 How these scores were calculated", expanded=False):
            _denom = max(len(display_job_skills), 1)
            _cap_line = "  (capped at 50% because fewer than 4 skills were extracted)" if display_capped else ""
            st.markdown("**Skills Match**")
            st.markdown(
                f"Skills extracted from this job: "
                f"`{display_skills_str(display_job_skills) if display_job_skills else '(none)'}` "
                f"({len(display_job_skills)} total)"
            )
            st.markdown(
                f"Your skills that matched: "
                f"`{display_skills_str(display_matched) if display_matched else '(none)'}` "
                f"({len(display_matched)})"
            )
            st.markdown(
                f"Your skills that are missing: "
                f"`{display_skills_str(display_missing) if display_missing else '(none)'}` "
                f"({len(display_missing)})"
            )
            st.code(
                f"Calculation: {len(display_matched)} matched / {_denom} total "
                f"= {(len(display_matched)/_denom)*100:.1f}%{_cap_line}\n"
                f"→ Skills Match = {format_pct(display_skill_score)}"
            )

            _resp_list = TYPICAL_ROLE_RESPONSIBILITIES.get(desired_role.strip().lower(), [])
            _rhits = role_hits if isinstance(role_hits, list) else []
            _title_has_role = desired_role.strip().lower() in (selected_row.get("job_title", "") or "").lower()
            _title_bonus = 0.3 if _title_has_role else 0.0
            st.markdown("**Role Match**")
            st.markdown(
                f"Checking if this job matches what a *“{safe(desired_role)}”* typically does."
            )
            if _resp_list:
                st.markdown(
                    f"Typical `{safe(desired_role)}` responsibilities checked ({len(_resp_list)} total):"
                )
                hit_set = set([r.lower() for r in _rhits])
                marks = "  ".join(
                    f"{safe(r)} {'✓' if r.lower() in hit_set else '✗'}"
                    for r in _resp_list
                )
                st.markdown(marks)
                _base_pct = (len(_rhits) / max(len(_resp_list), 1)) * 100
                st.code(
                    f"Responsibilities found in job: {len(_rhits)} of {len(_resp_list)}\n"
                    f"Base: {len(_rhits)} / {len(_resp_list)} = {_base_pct:.1f}%\n"
                    f"Title bonus (title contains “{desired_role}”): "
                    f"{'+30%' if _title_has_role else 'none'}\n"
                    f"→ Role Match = min({_base_pct:.1f}% + {int(_title_bonus*100)}%, 100%) "
                    f"= {format_pct(display_role_score)}"
                )
            else:
                st.markdown("(no predefined responsibility list — using role-word fallback)")
                st.code(f"→ Role Match = {format_pct(display_role_score)}")

            st.markdown("**Profile Similarity**")
            st.markdown(
                "Method: TF-IDF cosine similarity between your resume/profile text "
                "and the full job text (title + description + skills)."
            )
            st.code(f"→ Profile Similarity = {format_pct(display_nlp_score)}")

            st.markdown("**Overall Match**")
            st.code(
                f"Formula: ({nlp_weight:.0%} × Profile) + ({skill_weight:.0%} × Skills) + ({role_weight:.0%} × Role)\n"
                f"       = ({nlp_weight:.2f} × {format_pct(display_nlp_score)}) "
                f"+ ({skill_weight:.2f} × {format_pct(display_skill_score)}) "
                f"+ ({role_weight:.2f} × {format_pct(display_role_score)})\n"
                f"       = {nlp_weight*display_nlp_score*100:.2f}% "
                f"+ {skill_weight*display_skill_score*100:.2f}% "
                f"+ {role_weight*display_role_score*100:.2f}%\n"
                f"       = {display_final_score*100:.2f}% → shown as {format_pct(display_final_score)}\n"
                f"Source: {source_label}"
            )

            # BUG 9 FIX — worked example with actual numbers from this job
            st.markdown("---")
            st.markdown("**Worked Example (this job):**")
            st.code(
                f"SKILLS MATCH:\n"
                f"  This job requires {len(display_job_skills)} skills.\n"
                f"  You have {len(display_matched)} of them.\n"
                f"  Skills Match = {len(display_matched)}/{_denom} = {format_pct(display_skill_score)}\n\n"
                f"ROLE MATCH:\n"
                f"  {len(_rhits)} of {len(_resp_list)} '{desired_role}' responsibility keywords found.\n"
                f"  Title bonus: {'+30%' if _title_has_role else 'none'}\n"
                f"  Role Match = {format_pct(display_role_score)}\n\n"
                f"PROFILE SIMILARITY:\n"
                f"  TF-IDF cosine similarity = {format_pct(display_nlp_score)}\n\n"
                f"OVERALL MATCH:\n"
                f"  = ({nlp_weight:.0%} × {format_pct(display_nlp_score)}) + ({skill_weight:.0%} × {format_pct(display_skill_score)}) + ({role_weight:.0%} × {format_pct(display_role_score)})\n"
                f"  = {nlp_weight*display_nlp_score*100:.2f} + {skill_weight*display_skill_score*100:.2f} + {role_weight*display_role_score*100:.2f}\n"
                f"  = {format_pct(display_final_score)}"
            )

        # BUG 7 FIX — single-line markdown for each skill box (no leading whitespace)
        matched_text = safe(display_skills_str(display_matched)) if display_matched else '—'
        st.markdown(f'<div class="matched-box"><b>✅ Matched Skills ({len(display_matched)} matched)</b><br>{matched_text}</div>', unsafe_allow_html=True)

        missing_text = safe(display_skills_str(display_missing)) if display_missing else '—'
        st.markdown(f'<div class="missing-box"><b>❌ Missing Skills ({len(display_missing)} skills you don\'t have)</b><br>{missing_text}</div>', unsafe_allow_html=True)

        required_text = safe(display_skills_str(display_job_skills)) if display_job_skills else '—'
        st.markdown(f'<div class="required-box"><b>🛠 Skills Required by This Job ({len(display_job_skills)} total)</b><br>{required_text}</div>', unsafe_allow_html=True)

        if display_extra:
            extra_text = safe(display_skills_str(display_extra[:30]))
            st.markdown(f'<div class="extra-box"><b>➕ Extra Skills You Have ({len(display_extra)})</b><br>{extra_text}</div>', unsafe_allow_html=True)

        # BUG 8 FIX — cleaner terminal format with proper display case + job number
        try:
            print("\n" + "═" * 60)
            print(f"  CURRENTLY VIEWING: Job #{selected_global_idx + 1}")
            print(f"  {selected_row['job_title']} @ {selected_row['company']}")
            print(f"  {selected_row['job_location']}")
            print("─" * 60)
            print(f"  Skills Extracted ({len(display_job_skills)}): {display_skills_str(display_job_skills)}")
            print(f"  Your Matches ({len(display_matched)}): {display_skills_str(display_matched)}")
            print(f"  Missing ({len(display_missing)}): {display_skills_str(display_missing)}")
            print("─" * 60)
            print(f"  Skills Match : {format_pct(display_skill_score)}")
            print(f"  Profile Sim. : {format_pct(display_nlp_score)}")
            print(f"  Role Match   : {format_pct(display_role_score)}")
            print(f"  Overall      : {format_pct(display_final_score)}")
            print(f"  Source: {'Full page scrape' if used_full_desc else 'API snippet only'}")
            print("═" * 60)
        except Exception:
            pass

        # ----- Debug panel (UI) -----
        if debug_mode:
            with st.expander("🐞 Debug — full extraction pipeline"):
                st.markdown("**Step 1 — API snippet (what Adzuna returned)**")
                st.code(api_snippet[:3000] or "(empty)")

                st.markdown(f"**Step 2 — Full description (scraped = {used_full_desc})**")
                st.code((full_desc or "(none)")[:3000])

                st.markdown(f"**Step 3 — Extracted job skills ({len(display_job_skills)})**")
                st.write(display_job_skills)

                st.markdown(f"**Step 4 — Your resume + typed skills ({len(user_skills)})**")
                st.write(user_skills)

                st.markdown(f"**Step 5 — Matched ({len(display_matched)})**")
                st.write(display_matched)

                st.markdown(f"**Step 6 — Missing ({len(display_missing)})**")
                st.write(display_missing)

                st.markdown("**Step 7 — Score arithmetic**")
                denom = max(len(display_job_skills), 1)
                st.code(
                    f"skill_score = len(matched) / len(job_skills) "
                    f"= {len(display_matched)} / {denom} = {len(display_matched)/denom:.3f}\n"
                    f"(cap applied: {display_capped})\n\n"
                    f"final = {nlp_weight:.2f}·nlp + {skill_weight:.2f}·skill + {role_weight:.2f}·role\n"
                    f"      = {nlp_weight:.2f}·{display_nlp_score:.3f} "
                    f"+ {skill_weight:.2f}·{display_skill_score:.3f} "
                    f"+ {role_weight:.2f}·{display_role_score:.3f}\n"
                    f"      = {display_final_score:.3f}"
                )

                st.markdown(
                    f"**Step 8 — Typical responsibilities used for '{desired_role}'**"
                )
                resp_list = TYPICAL_ROLE_RESPONSIBILITIES.get(
                    desired_role.strip().lower(), []
                )
                st.write(resp_list if resp_list else "(no predefined list — using role-word fallback)")
                st.markdown(f"**Matched responsibilities:** {role_hits}")

        # ----- Description preview -----
        st.markdown("### 📄 Job Description")
        shown = full_desc if used_full_desc else api_snippet
        if shown:
            st.write(shown[:4000] + ("..." if len(shown) > 4000 else ""))
        else:
            st.write("No description available.")

        # ----- Salary -----
        sal_min = selected_row.get("salary_min")
        sal_max = selected_row.get("salary_max")
        if pd.notna(sal_min) or pd.notna(sal_max):
            if pd.notna(sal_min) and pd.notna(sal_max):
                st.write(f"### 💰 Salary Range\n${float(sal_min):,.0f} – ${float(sal_max):,.0f}")
            elif pd.notna(sal_min):
                st.write(f"### 💰 Salary Minimum\n${float(sal_min):,.0f}")
            else:
                st.write(f"### 💰 Salary Maximum\n${float(sal_max):,.0f}")

        if job_link:
            st.link_button("🔗 Open Job Posting", job_link)

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# MARKET SUMMARY / DOWNLOAD (below results)
# =========================================================
st.divider()
st.markdown('<div class="section-title">📊 Job Market Summary</div>', unsafe_allow_html=True)

sm1, sm2 = st.columns(2)
with sm1:
    st.write("**Top Job Titles**")
    top_titles = results_all_ranked["job_title"].value_counts().head(5)
    if not top_titles.empty:
        st.dataframe(
            top_titles.rename_axis("Job Title").reset_index(name="Count"),
            width='stretch',  # BUG 4 FIX — use_container_width deprecated
        )

    st.write("**Top Companies**")
    top_companies = results_all_ranked["company"].value_counts().head(5)
    if not top_companies.empty:
        st.dataframe(
            top_companies.rename_axis("Company").reset_index(name="Count"),
            width='stretch',  # BUG 4 FIX — use_container_width deprecated
        )

with sm2:
    st.write("**Industry Distribution**")
    ind_counts = results_all_ranked["industry"].value_counts()
    if not ind_counts.empty:
        st.bar_chart(ind_counts)

    st.write("**Career Level Distribution**")
    lvl_counts = results_all_ranked["job_level_norm"].value_counts()
    if not lvl_counts.empty:
        st.bar_chart(lvl_counts)

st.divider()
st.markdown('<div class="section-title">🧠 Resume Improvement Suggestions</div>', unsafe_allow_html=True)

top_missing = []
for ms in results_all_ranked.head(25)["missing_skills"].tolist():
    if isinstance(ms, list):
        top_missing.extend(ms)
missing_counts = (
    pd.Series(top_missing).value_counts().head(10)
    if top_missing
    else pd.Series(dtype=int)
)

if len(missing_counts):
    st.write("Across your top-ranked jobs, these skills came up most often as missing from your resume:")
    st.dataframe(
        missing_counts.rename_axis("Skill").reset_index(name="Appears in N jobs"),
        width='stretch',  # BUG 4 FIX — use_container_width deprecated
    )
else:
    st.success("No major skill gaps detected in your top ranked jobs.")

st.markdown("### 💰 Salary Insight")
salary_df = results_all_ranked[["salary_min", "salary_max"]].dropna(how="all")
if not salary_df.empty:
    v_min = pd.to_numeric(salary_df["salary_min"], errors="coerce").dropna()
    v_max = pd.to_numeric(salary_df["salary_max"], errors="coerce").dropna()
    if not v_min.empty or not v_max.empty:
        a_min = v_min.mean() if not v_min.empty else None
        a_max = v_max.mean() if not v_max.empty else None
        if a_min is not None and a_max is not None:
            st.write(f"Average salary range across ranked jobs: **${a_min:,.0f} – ${a_max:,.0f}**")
        elif a_min is not None:
            st.write(f"Average minimum salary: **${a_min:,.0f}**")
        else:
            st.write(f"Average maximum salary: **${a_max:,.0f}**")
    else:
        st.write("Salary data is not available for the current ranked jobs.")
else:
    st.write("Salary data is not available for the current ranked jobs.")

st.divider()
st.markdown('<div class="section-title">⬇️ Download Results</div>', unsafe_allow_html=True)

download_cols = [
    "job_title", "company", "job_location", "work_mode", "industry",
    "employment_type_norm", "job_level_norm", "job_skills",
    "final_score", "nlp_score", "skill_score", "role_score",
    "salary_min", "salary_max", "job_link",
]
download_cols = [c for c in download_cols if c in results_all_ranked.columns]

csv_all = results_all_ranked[download_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download All Ranked Results (CSV)",
    data=csv_all,
    file_name="all_ranked_job_recommendations.csv",
    mime="text/csv",
)
