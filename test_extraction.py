"""
Multi-industry skill extraction tests.

Run:
    python test_extraction.py
"""

import re


SKILLS = {
    # Data & IT
    "sql", "python", "r", "excel", "tableau", "power bi", "looker", "sigma",
    "metabase", "snowflake", "databricks", "aws", "azure", "gcp", "etl",
    "elt", "api", "database", "databases", "data modeling", "dashboards",
    "reporting", "statistics", "machine learning", "data visualization",
    "data cleaning", "data validation", "data governance", "data pipelines",
    "data warehouse", "data quality", "kpi", "business intelligence", "spark", "kafka",
    "pandas", "numpy", "git", "jira", "agile", "microsoft excel",
    # Healthcare
    "ehr", "emr", "epic", "cerner", "hipaa", "patient care",
    "care coordination", "medical terminology", "clinical documentation",
    "insurance verification", "prior authorization", "icd-10", "cpt",
    "triage", "case management", "telehealth", "claims processing",
    # Accounting & Finance
    "gaap", "accounts payable", "accounts receivable", "reconciliation",
    "general ledger", "payroll", "bookkeeping", "quickbooks",
    "financial reporting", "budgeting", "forecasting", "audit",
    "tax preparation", "variance analysis", "cash flow", "financial analysis",
    "financial modeling", "month-end close",
    # Business & Operations
    "business analysis", "requirements gathering", "process improvement",
    "workflow optimization", "stakeholder management", "vendor management",
    "supply chain", "operations management", "project management",
    "documentation", "kpi tracking", "procurement", "risk management",
    # Marketing
    "seo", "sem", "google analytics", "campaign management",
    "email marketing", "social media marketing", "content marketing", "crm",
    "salesforce", "market research", "advertising", "lead generation",
    "brand strategy", "marketing analytics", "paid search", "paid social",
    "hubspot", "mailchimp",
    # HR / Education
    "recruitment", "onboarding", "hris", "employee relations",
    "benefits administration", "performance management", "policy compliance",
    "candidate screening", "talent acquisition", "workday", "teaching",
    "lesson planning", "curriculum development", "classroom management",
    "grading", "assessment", "academic advising", "instruction",
    "student engagement", "lms", "canvas", "blackboard",
    # Customer/Admin/general
    "customer service", "data entry", "scheduling", "calendar management",
    "microsoft office", "email communication", "records management",
    "phone support", "order processing", "administrative support",
    "communication", "leadership", "problem solving", "collaboration",
}

ALIASES = {
    "powerbi": "power bi",
    "microsoft power bi": "power bi",
    "structured query language": "sql",
    "extract transform load": "etl",
    "extract, transform, and load": "etl",
    "extract transform and load": "etl",
    "application programming interface": "api",
    "apis": "api",
    "business intelligence tools": "business intelligence",
    "bi tools": "business intelligence",
    "dashboarding": "dashboards",
    "reporting tools": "reporting",
    "key performance indicators": "kpi",
    "key performance indicator": "kpi",
    "electronic health record": "ehr",
    "electronic health records": "ehr",
    "electronic medical record": "emr",
    "electronic medical records": "emr",
    "customer relationship management": "crm",
    "human resource information system": "hris",
    "human resources information system": "hris",
    "search engine optimization": "seo",
    "search engine marketing": "sem",
    "accounts payable": "accounts payable",
    "accounts receivable": "accounts receivable",
    "ap processing": "accounts payable",
    "ar processing": "accounts receivable",
    "microsoft excel": "excel",
    "ms excel": "excel",
    "microsoft office": "microsoft office",
    "learning management system": "lms",
    "learning management systems": "lms",
    "month end close": "month-end close",
    "reconciliations": "reconciliation",
    "social media management": "social media marketing",
}

DISPLAY = {
    "sql": "SQL", "r": "R", "api": "API", "etl": "ETL", "elt": "ELT",
    "kpi": "KPI", "ehr": "EHR", "emr": "EMR", "hipaa": "HIPAA",
    "gaap": "GAAP", "crm": "CRM", "hris": "HRIS", "seo": "SEO",
    "sem": "SEM", "icd-10": "ICD-10", "cpt": "CPT", "aws": "AWS",
    "gcp": "GCP", "lms": "LMS",
}

GENERIC_FALSE_POSITIVES = {
    "equal opportunity", "race", "color", "religion", "sex", "age",
    "disability", "veteran", "national origin", "remote", "required",
    "reporting to", "new york", "san francisco", "atlanta", "chicago",
    "boston", "dallas", "phoenix", "benefits", "salary", "degree",
}


def normalize(text):
    text = "" if text is None else str(text).lower()
    text = text.replace("&", " and ")
    return re.sub(r"\s+", " ", text).strip()


def pattern(term):
    escaped = re.escape(normalize(term)).replace(r"\ ", r"\s+")
    escaped = escaped.replace("/", r"\s*/\s*")
    return r"(?<!\w)" + escaped + r"(?!\w)"


def split_skill_list(text):
    parts = re.split(r",|;|\||\band\b|\bor\b", text, flags=re.I)
    out = []
    for part in parts:
        item = normalize(part).strip(" .:-")
        if not item:
            continue
        if "/" in item and item not in SKILLS and item not in ALIASES:
            out.extend([normalize(p).strip(" .:-") for p in item.split("/") if p.strip()])
        else:
            out.append(item)
    return out


def extract_exact_job_skills(title, text):
    blob_raw = f"{title or ''}\n{text or ''}"
    blob = normalize(blob_raw)
    found = set()

    for sk in sorted(SKILLS, key=len, reverse=True):
        if sk in {"r"}:
            if re.search(r"(?:^|[,;:\s/(\[])r(?:$|[,;:\s/)\]])", blob):
                found.add(sk)
        elif re.search(pattern(sk), blob):
            found.add(sk)

    for alias, canon in ALIASES.items():
        if re.search(pattern(alias), blob):
            found.add(canon)

    phrase_patterns = [
        r"(?:experience with|skilled in|knowledge of|familiarity with|proficient in|working knowledge of)\s+([a-z0-9#\+/\- ,;]{3,260})",
        r"(?:required skills?|preferred qualifications?|qualifications?|requirements?|tools?|technologies?)[:\s]+([a-z0-9#\+/\- ,;]{3,320})",
    ]
    for pat in phrase_patterns:
        for chunk in re.findall(pat, blob, flags=re.I):
            for item in split_skill_list(chunk):
                canon = ALIASES.get(item, item)
                if canon in SKILLS:
                    found.add(canon)

    for chunk in re.findall(r"\(([^)]{2,220})\)", blob):
        for item in split_skill_list(chunk):
            canon = ALIASES.get(item, item)
            if canon in SKILLS:
                found.add(canon)

    return sorted(s for s in found if s not in GENERIC_FALSE_POSITIVES)


def missing_skills(candidate_skills, job_skills):
    candidate = {ALIASES.get(normalize(s), normalize(s)) for s in candidate_skills}
    return sorted(set(job_skills) - candidate)


CASES = [
    {
        "name": "Data/IT - BI Analyst",
        "title": "Business Intelligence Analyst",
        "description": """
        Responsibilities include dashboarding, KPI reporting, and stakeholder presentations.
        Required Skills: Structured Query Language, PowerBI, Tableau, Excel, Snowflake,
        data modeling, data validation, reporting tools, and business intelligence tools.
        Equal opportunity employer. Location: New York, NY.
        """,
        "expected": {"sql", "power bi", "tableau", "excel", "snowflake", "data modeling", "data validation", "reporting", "business intelligence", "kpi", "dashboards"},
        "candidate": {"sql", "excel", "tableau"},
        "must_miss": {"power bi", "snowflake", "data modeling"},
    },
    {
        "name": "Data/IT - Data Engineer",
        "title": "Data Engineer",
        "description": """
        Build ETL/ELT pipelines and APIs on AWS and Databricks. Experience with Python,
        SQL, Spark, Kafka, data warehouse design, data governance, Git, and Agile delivery.
        Do not extract remote, required, or reporting to the manager as skills.
        """,
        "expected": {"etl", "elt", "api", "aws", "databricks", "python", "sql", "spark", "kafka", "data warehouse", "data governance", "git", "agile"},
        "candidate": {"python", "sql", "git"},
        "must_miss": {"etl", "aws", "databricks", "spark"},
    },
    {
        "name": "Healthcare - Care Coordinator",
        "title": "Clinical Care Coordinator",
        "description": """
        Qualifications: patient care, care coordination, electronic health records (Epic,
        Cerner), HIPAA, medical terminology, clinical documentation, insurance verification,
        prior authorization, ICD-10, CPT, telehealth, and case management.
        EEO statement: race, color, religion, sex, national origin, disability.
        """,
        "expected": {"patient care", "care coordination", "ehr", "epic", "cerner", "hipaa", "medical terminology", "clinical documentation", "insurance verification", "prior authorization", "icd-10", "cpt", "telehealth", "case management"},
        "candidate": {"patient care", "hipaa", "epic"},
        "must_miss": {"prior authorization", "icd-10", "cpt"},
    },
    {
        "name": "Accounting/Finance - Staff Accountant",
        "title": "Staff Accountant",
        "description": """
        Requirements: GAAP, accounts payable, accounts receivable, reconciliations,
        general ledger, payroll, QuickBooks, financial reporting, budgeting, forecasting,
        audit support, tax preparation, variance analysis, cash flow, and month end close.
        Location: Dallas, TX. Benefits and salary listed separately.
        """,
        "expected": {"gaap", "accounts payable", "accounts receivable", "reconciliation", "general ledger", "payroll", "quickbooks", "financial reporting", "budgeting", "forecasting", "audit", "tax preparation", "variance analysis", "cash flow", "month-end close"},
        "candidate": {"excel", "quickbooks", "payroll"},
        "must_miss": {"gaap", "general ledger", "variance analysis"},
    },
    {
        "name": "Business/Ops - Operations Analyst",
        "title": "Operations Analyst",
        "description": """
        Responsibilities: business analysis, requirements gathering, process improvement,
        workflow optimization, stakeholder management, vendor management, supply chain,
        operations management, project management, documentation, KPI tracking,
        procurement, and risk management.
        """,
        "expected": {"business analysis", "requirements gathering", "process improvement", "workflow optimization", "stakeholder management", "vendor management", "supply chain", "operations management", "project management", "documentation", "kpi tracking", "procurement", "risk management"},
        "candidate": {"project management", "documentation"},
        "must_miss": {"requirements gathering", "vendor management", "supply chain"},
    },
    {
        "name": "Marketing - Digital Marketing Specialist",
        "title": "Digital Marketing Specialist",
        "description": """
        Skilled in search engine optimization, search engine marketing, Google Analytics,
        campaign management, email marketing, social media management, content marketing,
        CRM, Salesforce, market research, advertising, lead generation, brand strategy,
        marketing analytics, paid search, paid social, HubSpot, and Mailchimp.
        """,
        "expected": {"seo", "sem", "google analytics", "campaign management", "email marketing", "social media marketing", "content marketing", "crm", "salesforce", "market research", "advertising", "lead generation", "brand strategy", "marketing analytics", "paid search", "paid social", "hubspot", "mailchimp"},
        "candidate": {"seo", "google analytics", "crm"},
        "must_miss": {"sem", "campaign management", "mailchimp"},
    },
    {
        "name": "HR/Education - HR Training Coordinator",
        "title": "HR Training Coordinator",
        "description": """
        Preferred qualifications: recruitment, onboarding, HRIS, employee relations,
        benefits administration, performance management, policy compliance, candidate
        screening, talent acquisition, Workday, instruction, curriculum development,
        LMS (Canvas, Blackboard), assessment, grading, and student engagement.
        """,
        "expected": {"recruitment", "onboarding", "hris", "employee relations", "benefits administration", "performance management", "policy compliance", "candidate screening", "talent acquisition", "workday", "instruction", "curriculum development", "lms", "canvas", "blackboard", "assessment", "grading", "student engagement"},
        "candidate": {"recruitment", "onboarding", "workday"},
        "must_miss": {"hris", "employee relations", "lms"},
    },
]


def display(skill):
    return DISPLAY.get(skill, skill.title())


def run_case(case):
    extracted = set(extract_exact_job_skills(case["title"], case["description"]))
    expected = case["expected"]
    missed_expected = expected - extracted
    missing = set(missing_skills(case["candidate"], extracted))
    false_hits = extracted & GENERIC_FALSE_POSITIVES
    location_hits = extracted & {"new york", "san francisco", "atlanta", "chicago", "boston", "dallas", "phoenix"}
    generic_hits = extracted & {"remote", "required", "benefits", "salary", "degree"}

    assert not missed_expected, f"{case['name']} missed expected skills: {sorted(missed_expected)}"
    assert case["must_miss"] <= missing, f"{case['name']} missing-skills logic failed: expected {sorted(case['must_miss'])}, got {sorted(missing)}"
    assert not false_hits, f"{case['name']} extracted EEO boilerplate: {sorted(false_hits)}"
    assert not location_hits, f"{case['name']} extracted locations: {sorted(location_hits)}"
    assert not generic_hits, f"{case['name']} extracted generic words: {sorted(generic_hits)}"
    return extracted, missing


def main():
    print("=" * 72)
    print("MULTI-INDUSTRY SKILL EXTRACTION TESTS")
    print("=" * 72)
    for case in CASES:
        extracted, missing = run_case(case)
        print(f"[PASS] {case['name']}")
        print(f"       extracted {len(extracted)}: {', '.join(display(s) for s in sorted(extracted))}")
        print(f"       missing   {len(missing)}: {', '.join(display(s) for s in sorted(missing))}")
    print("=" * 72)
    print(f"PASS: {len(CASES)} cases")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

