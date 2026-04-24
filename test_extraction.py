"""
TASK 5: Test Skill Extraction Against Multiple Job Descriptions
================================================================
STANDALONE version - does NOT import from app.py (avoids streamlit dependency).
Instead, it copies the core extraction functions directly.

Run: python test_extraction.py
"""

import re

# === Core extraction logic copied from app.py ===

COMMON_JOB_SKILLS = {
    "sql","python","excel","power bi","tableau","pandas","numpy",
    "matplotlib","seaborn","scikit-learn","machine learning","nlp",
    "statistics","data analysis","data analytics","data visualization",
    "data science","data management","data cleaning","data validation",
    "dashboards","business intelligence","eda","predictive modeling",
    "tf-idf","streamlit","reporting","deep learning",
    "feature engineering","data modeling","data warehousing",
    "data warehouse","etl","elt","kpi","metrics",
    "ad hoc reporting","data governance","forecasting",
    "a/b testing","regression analysis","predictive analytics",
    "natural language processing","computer vision",
    "structured data","unstructured data","data profiling",
    "data quality","data integration","data pipelines","data mining",
    "data storytelling","storytelling","data reliability",
    "data accuracy","schema design","scope management",
    "requirements negotiation","data quality testing",
    "data presentation","insight generation",
    "mysql","postgresql","sqlite","sql server","mongodb",
    "oracle","nosql","snowflake","bigquery","redshift",
    "stored procedures","transact-sql","t-sql",
    "aws","azure","gcp","cloud computing","docker","kubernetes",
    "terraform","ansible","git","ci/cd","devops","linux",
    "jenkins","airflow","databricks",
    "flask","django","html","css","javascript","typescript",
    "java","c++","c#","ruby","go","rust","kotlin","swift",
    "node.js","react","angular","vue","rest api","api",
    "graphql","spring boot",
    "ssis","ssrs","ssms","ssas","looker","qlik","qlik sense",
    "microstrategy","alteryx","knime","informatica","sas","r",
    "spss","google sheets","google analytics","power query",
    "pivot tables","vlookup","xlookup",
    "gis","gis tools","arcgis","qgis",
    "data integrity","data accuracy",
    "ad hoc","performance metrics","kpi tracking",
    "stakeholder communication","cross-functional","project tracking",
    "powerpoint","word","outlook","ms office","ms access",
    "sharepoint","ms project","ms visio","jira","confluence",
    "agile","scrum","kanban",
    "communication","teamwork","leadership","documentation",
    "problem solving","project management","stakeholder management",
    "critical thinking","time management","analytical thinking",
    "collaboration",
    "patient care","clinical assessment","triage","diagnosis",
    "treatment planning","care coordination","patient education",
    "medical records","ehr","emr","epic","cerner","hipaa",
    "infection control","telehealth","case management","medical terminology",
    "accounting","bookkeeping","financial reporting","budgeting",
    "audit","auditing","gaap","accounts payable",
    "accounts receivable","bank reconciliation","payroll",
    "financial analysis","tax preparation","quickbooks",
    "general ledger","cash flow","reconciliation",
    "business analysis","requirements gathering","process improvement",
    "operations management","workflow optimization","crm",
    "salesforce","vendor management","supply chain",
    "customer service","sap","erp","workday",
    "seo","sem","social media","campaign management",
    "content marketing","content creation","email marketing",
    "brand strategy","market research","advertising",
    "copywriting","digital marketing","lead generation",
    "recruitment","talent acquisition","employee relations",
    "onboarding","benefits administration","performance management",
    "hris","policy compliance","interviewing","candidate screening",
    "teaching","curriculum development","lesson planning",
    "classroom management","assessment","academic advising","instruction",
    "unit testing","integration testing","qa","quality assurance",
    "test automation","selenium",
}

SKILL_ALIASES = {
    "ms excel":"excel","microsoft excel":"excel","advanced excel":"excel",
    "powerbi":"power bi","microsoft power bi":"power bi",
    "scikit learn":"scikit-learn","sklearn":"scikit-learn",
    "natural language processing":"nlp",
    "sqlite":"sql","mysql":"sql","postgresql":"sql","postgres":"sql",
    "sql server":"sql","ms sql":"sql","t-sql":"sql","tsql":"sql",
    "transact-sql":"sql","structured query language":"sql",
    "amazon web services":"aws","tf idf":"tf-idf","tfidf":"tf-idf",
    "negotiate requirements":"requirements negotiation",
    "negotiating requirements":"requirements negotiation",
    "visualization":"data visualization",
    "data visualisation":"data visualization","data viz":"data visualization",
    "storytelling with data":"data storytelling",
    "present insights":"data storytelling",
    "presenting insights":"data storytelling",
    "ensure data accuracy":"data accuracy","ensure accuracy":"data accuracy",
    "ensure data reliability":"data reliability",
    "reliability":"data reliability",
    "data quality test":"data quality testing",
    "data quality tests":"data quality testing",
    "quality testing":"data quality testing",
    "a/b testing analysis":"a/b testing",
    "schema development":"schema design","database schema":"schema design",
    "py":"python","js":"javascript","restful api":"rest api",
    "bi tools":"business intelligence",
    "dashboarding":"dashboards","dashboard":"dashboards",
    "dashboard development":"dashboards",
    "reporting tools":"reporting","ad hoc reporting":"reporting",
    "pivot table":"pivot tables",
    "a/b tests":"a/b testing","ab testing":"a/b testing",
    "qlik sense":"qlik",
}

GENERIC_NOT_SKILLS = {
    "experience","knowledge","skills","ability","responsible",
    "responsibilities","tools","tool","management","work",
    "working","support","operations","business","computer",
    "university","education","training","certification",
    "role","roles","candidate","job","position","strong",
    "excellent","required","ability to","must have","plus",
    "etc","using","including","basic","advanced","benefits",
    "salary","responsibility","duties","programs","program",
    "location","citizen","applicant","minimum","maximum",
    "year","years","degree","bachelor","master","phd",
    "related","equivalent","full time","part time","company",
    "employer","department","description","overview","summary",
    "environment","fast paced","detail oriented","independently",
    "ensure","maintain","provide","assist","perform",
    "develop","create","implement","coordinate",
    "review","prepare","analyze","evaluate","monitor",
    "communicate","formats","format","methods","processes",
    "process","networks","system","systems",
    "color","religion","sex","age","race","disability",
    "veteran","gender","orientation","national origin",
    "attention to detail","written","verbal","available",
    "proficiency","familiarity","insights","trends",
    "accuracy","consistency","integrity",
}

INSTITUTION_WORDS = {"college","university","academy","institute","school","inc","llc","corp","ltd"}
US_LOCATION_WORDS = {"atlanta","georgia","marietta","san","ramon","ga","tn","ca","ny","tx","fl"}

ALLOWED_MULTIWORD_SKILLS = {
    "data analysis","data analytics","data visualization",
    "data science","data cleaning","data validation",
    "data management","data mining","data integrity",
    "data accuracy","data governance","data modeling",
    "data warehousing","data warehouse","data profiling",
    "data quality","data integration","data pipelines",
    "machine learning","deep learning","natural language processing",
    "computer vision","power bi","power query","pivot tables",
    "business intelligence","predictive modeling","predictive analytics",
    "feature engineering","a/b testing","regression analysis",
    "scikit-learn","sql server","ms sql","google analytics",
    "google sheets","rest api","cloud computing",
    "project management","stakeholder management",
    "customer service","problem solving","critical thinking",
    "time management","analytical thinking","ad hoc reporting",
    "ad hoc","gis tools","social media","content marketing",
    "email marketing","digital marketing","lead generation",
    "market research","campaign management","brand strategy",
    "financial analysis","financial reporting","general ledger",
    "accounts payable","accounts receivable","bank reconciliation",
    "tax preparation","cash flow","operations management",
    "workflow optimization","vendor management","supply chain",
    "patient care","clinical assessment","treatment planning",
    "care coordination","patient education","medical records",
    "medical terminology","infection control","case management",
    "telehealth","talent acquisition","employee relations",
    "benefits administration","performance management",
    "policy compliance","candidate screening",
    "curriculum development","lesson planning","classroom management",
    "academic advising","unit testing","integration testing",
    "quality assurance","test automation","ms office","ms access",
    "ms project","ms visio","spring boot","node.js",
    "qlik sense","structured data","unstructured data",
    "kpi tracking","performance metrics","stakeholder communication",
    "cross-functional","project tracking",
    "requirements gathering","process improvement","business analysis",
    "data storytelling","data reliability","schema design",
    "scope management","requirements negotiation",
    "data quality testing","data presentation","insight generation",
    "stored procedures",
}

_SENTENCE_WORDS = {
    "shall","will","would","could","should","must","may","might",
    "the","this","that","these","those","their","our","your",
    "who","whom","which","where","when","what","how","why",
    "been","being","have","has","had","does","did",
    "not","also","very","just","only","than","then",
    "from","into","with","within","without","between",
    "about","after","before","during","through",
}

_NON_SKILL_PATTERNS = [
    r"\b(?:us citizen|citizen|citizenship|clearance)\b",
    r"\b(?:at least|one or more|years? of|degree in|must be|ability to)\b",
    r"\b(?:shall|will|would|could|should)\s+\w+",
    r"^the\s+",
]

DISPLAY_CASE = {
    "etl":"ETL","sql":"SQL","kpi":"KPI","nlp":"NLP","aws":"AWS",
    "gcp":"GCP","api":"API","power bi":"Power BI","python":"Python",
    "excel":"Excel","tableau":"Tableau","r":"R","git":"Git",
    "jira":"JIRA","confluence":"Confluence","snowflake":"Snowflake",
    "azure":"Azure","airflow":"Airflow","agile":"Agile",
    "pandas":"Pandas","numpy":"NumPy","scikit-learn":"Scikit-learn",
    "machine learning":"Machine Learning",
    "data visualization":"Data Visualization",
    "data analysis":"Data Analysis",
    "data storytelling":"Data Storytelling",
    "data accuracy":"Data Accuracy",
    "data reliability":"Data Reliability",
    "data quality":"Data Quality",
    "data quality testing":"Data Quality Testing",
    "schema design":"Schema Design",
    "scope management":"Scope Management",
    "requirements negotiation":"Requirements Negotiation",
    "requirements gathering":"Requirements Gathering",
    "data modeling":"Data Modeling",
    "data warehousing":"Data Warehousing",
    "data warehouse":"Data Warehouse",
    "data governance":"Data Governance",
    "data management":"Data Management",
    "data integration":"Data Integration",
    "data pipelines":"Data Pipelines",
    "data integrity":"Data Integrity",
    "predictive modeling":"Predictive Modeling",
    "predictive analytics":"Predictive Analytics",
    "forecasting":"Forecasting","dashboards":"Dashboards",
    "reporting":"Reporting","documentation":"Documentation",
    "communication":"Communication",
    "stakeholder management":"Stakeholder Management",
    "project management":"Project Management",
    "business intelligence":"Business Intelligence",
    "problem solving":"Problem Solving",
    "analytical thinking":"Analytical Thinking",
    "teamwork":"Teamwork","a/b testing":"A/B Testing",
    "kpi tracking":"KPI Tracking","pivot tables":"Pivot Tables",
    "statistics":"Statistics","storytelling":"Storytelling",
}

def display_skill(skill):
    return DISPLAY_CASE.get(skill.lower(), skill.title() if " " in skill else skill)

def normalize_text(x):
    x = "" if x is None else str(x)
    x = x.replace("\xa0"," ")
    return re.sub(r"\s+"," ",x).strip()

def clean_for_matching(text):
    text = normalize_text(text).lower()
    text = text.replace("&"," and ").replace("/"," / ")
    text = re.sub(r"[\|\(\)\[\]\{\}:]"," ",text)
    return re.sub(r"\s+"," ",text).strip()

def looks_like_skill(s):
    s = (s or "").strip().lower()
    if not (2 <= len(s) <= 40): return False
    if s in GENERIC_NOT_SKILLS: return False
    is_approved = s in ALLOWED_MULTIWORD_SKILLS or s in COMMON_JOB_SKILLS
    if re.match(r"^\d{5}",s): return False
    if "&" in s or "http" in s or "www." in s: return False
    words = s.split()
    if len(words)==1 and words[0] in US_LOCATION_WORDS: return False
    if any(w in INSTITUTION_WORDS for w in words): return False
    if not re.search(r"[a-z0-9\+#]",s): return False
    if s.isdigit(): return False
    if is_approved: return True
    if len(words)>3: return False
    if len(words)>=2 and any(w in _SENTENCE_WORDS for w in words): return False
    if len(words)>=2 and any(w in GENERIC_NOT_SKILLS for w in words): return False
    if len(words)>=2: return False
    for pat in _NON_SKILL_PATTERNS:
        if re.search(pat,s): return False
    return True

def _split_slash_variants(tokens):
    out = []
    for t in tokens:
        t = str(t).strip().lower()
        if not t: continue
        if t in ALLOWED_MULTIWORD_SKILLS or t in SKILL_ALIASES or t in COMMON_JOB_SKILLS:
            out.append(t); continue
        if "/" in t:
            out.extend([p.strip() for p in t.split("/") if p.strip()])
        else: out.append(t)
    return out

def apply_skill_aliases(skills):
    out = set()
    for s in _split_slash_variants(skills):
        s = SKILL_ALIASES.get(str(s).strip().lower(), str(s).strip().lower())
        out.add(s)
    return sorted([s for s in out if looks_like_skill(s)])

def _scan_dictionary(text_lower, found):
    for sk in sorted(COMMON_JOB_SKILLS, key=len, reverse=True):
        if re.search(r"(?<!\w)"+re.escape(sk)+r"(?!\w)", text_lower):
            found.add(sk)
    for alias, canon in SKILL_ALIASES.items():
        if re.search(r"(?<!\w)"+re.escape(alias)+r"(?!\w)", text_lower):
            found.add(canon)

def _extract_from_requirement_phrases(raw_text, found):
    stripped = re.sub(r"\(.*?\)"," ",str(raw_text))
    text = clean_for_matching(stripped)
    patterns = [
        r"(?:experience with|proficiency in|proficient in|knowledge of|expertise in|skilled in|hands[- ]on experience with|experience using|familiarity with|familiar with)\s+([a-zA-Z0-9\+\#\/\-\s,]{3,300})",
        r"(?:required skills?|preferred skills?|technical skills?|core skills?|key skills?)[:\s]+([a-zA-Z0-9\+\#\/\-\s,]{3,350})",
        r"(?:must have|should have|nice to have)\s+([a-zA-Z0-9\+\#\/\-\s,]{3,250})",
        r"(?:tools?|technologies?|stack|platforms?)[:\s]+([a-zA-Z0-9\+\#\/\-\s,]{3,250})",
        r"(?:using|use of|build using|reports using)\s+([a-zA-Z0-9\+\#\/\-\s,]{3,250})",
    ]
    for pat in patterns:
        for m in re.findall(pat, text, flags=re.IGNORECASE):
            for p in re.split(r",|;|/|\band\b|\bor\b", m):
                p = SKILL_ALIASES.get(p.strip(" .:-").lower(), p.strip(" .:-").lower())
                if (p in COMMON_JOB_SKILLS or p in ALLOWED_MULTIWORD_SKILLS) and looks_like_skill(p):
                    found.add(p)

def _extract_from_colon_lines(raw_text, found):
    for line in str(raw_text).splitlines():
        if ":" not in line: continue
        _, _, right = line.partition(":")
        right = re.sub(r"\(.*?\)","",right)
        tokens = []
        for p in re.split(r",|;|\|",right):
            p = p.strip(" .:-").lower()
            if not p: continue
            if p in ALLOWED_MULTIWORD_SKILLS or p in COMMON_JOB_SKILLS or p in SKILL_ALIASES:
                tokens.append(p); continue
            if "/" in p:
                tokens.extend([x.strip() for x in p.split("/") if x.strip()])
            else: tokens.append(p)
        for p in tokens:
            if re.match(r"^https?://",p) or re.match(r"^www\.",p): continue
            if any(w in INSTITUTION_WORDS for w in p.split()): continue
            p = SKILL_ALIASES.get(p,p)
            if (p in COMMON_JOB_SKILLS or p in ALLOWED_MULTIWORD_SKILLS) and looks_like_skill(p):
                found.add(p)

def _extract_from_bullet_points(raw_text, found):
    text = clean_for_matching(raw_text)
    for chunk in re.split(r"[.;\u2022\u25cf\u25cb\u2013\u2014\n\r]", text):
        chunk = chunk.strip()
        if len(chunk) < 5: continue
        _scan_dictionary(chunk, found)

def extract_exact_job_skills(title, text):
    if not text or not str(text).strip(): return []
    full = f"{title or ''}\n{text}"
    found = set()
    lower = clean_for_matching(full)
    _scan_dictionary(lower, found)
    for alias, canon in SKILL_ALIASES.items():
        if re.search(r"(?<!\w)"+re.escape(alias)+r"(?!\w)", lower):
            if canon in COMMON_JOB_SKILLS or canon in ALLOWED_MULTIWORD_SKILLS:
                found.add(canon)
    _extract_from_requirement_phrases(full, found)
    _extract_from_colon_lines(full, found)
    _extract_from_bullet_points(full, found)
    clean = set()
    for s in found:
        s = SKILL_ALIASES.get(s,s)
        if (s in COMMON_JOB_SKILLS or s in ALLOWED_MULTIWORD_SKILLS) and looks_like_skill(s):
            clean.add(s)
    return sorted(clean)

# =========================================================
# SAMPLE JOB DESCRIPTIONS
# =========================================================
SAMPLE_JOBS = [
    {
        "name": "Sample Job 1 - Data Analyst at HealthCorp",
        "title": "Data Analyst",
        "description": """
        We are looking for a Data Analyst. You will work with stakeholders
        to negotiate requirements, define project scope, and deliver insights.

        Responsibilities:
        - Build and maintain ETL pipelines to extract, transform, and load data
        - Create dashboards and data visualization reports using Tableau and Power BI
        - Ensure data accuracy, reliability, and quality across all reporting
        - Present insights using storytelling and visualization techniques
        - Write complex SQL queries to analyze large datasets
        - Collaborate with cross-functional teams on schema design and data modeling
        - Maintain documentation for all data processes and pipelines
        - Perform data quality testing and validation

        Requirements:
        - Proficiency in SQL, Python, and Excel
        - Experience with ETL processes and data warehousing
        - Strong data visualization skills (Tableau, Power BI)
        - Knowledge of statistics and predictive modeling
        - Experience with data governance and data management
        - Excellent communication and stakeholder management skills
        """,
        "expected_skills": [
            "etl","sql","python","excel","tableau","power bi",
            "data visualization","data accuracy","data quality",
            "dashboards","data modeling","documentation",
            "statistics","predictive modeling","data governance",
            "data management","communication","stakeholder management",
            "data storytelling","schema design","data quality testing",
            "data warehousing","requirements negotiation",
            "storytelling","data reliability",
        ],
    },
    {
        "name": "Sample Job 2 - Junior Data Analyst at TechStartup",
        "title": "Junior Data Analyst",
        "description": """
        TechStartup is hiring a Junior Data Analyst!

        What you'll do:
        - Analyze data using SQL and Python to support business decisions
        - Build interactive dashboards in Power BI for KPI tracking
        - Clean and validate datasets to ensure data integrity
        - Support ad hoc reporting requests from various departments
        - Use Excel for data analysis and pivot tables
        - Assist with A/B testing analysis
        - Help maintain data pipelines using Airflow

        What we're looking for:
        - Familiarity with SQL, Python, and Excel
        - Experience with Power BI or Tableau
        - Understanding of statistics and data analysis
        - Strong problem solving and analytical thinking
        - Good communication skills and teamwork
        - Knowledge of pandas and numpy is a plus
        - Experience with Git version control
        """,
        "expected_skills": [
            "sql","python","excel","power bi","tableau",
            "dashboards","kpi tracking","data pipelines",
            "statistics","data analysis","problem solving",
            "analytical thinking","communication","teamwork",
            "pandas","numpy","git","airflow","pivot tables",
            "a/b testing","data integrity","reporting",
        ],
    },
    {
        "name": "Sample Job 3 - Senior Data Analyst at FinanceBank",
        "title": "Senior Data Analyst",
        "description": """
        The Senior Data Analyst will lead analytics initiatives.

        Key Responsibilities:
        - Design and implement ETL processes for data integration
        - Develop forecasting models and predictive analytics solutions
        - Create comprehensive data visualization and reporting using Tableau
        - Manage data warehouse architecture and schema design
        - Ensure data reliability and accuracy across all systems
        - Lead requirements gathering and scope management for analytics projects
        - Present findings to stakeholders using data storytelling
        - Mentor junior analysts on SQL, Python, and best practices

        Required Skills:
        Programming: Python, SQL, R
        Data Analysis & Visualization: Tableau, Power BI, Excel
        Database: Snowflake, SQL Server, PostgreSQL
        Tools: Git, JIRA, Confluence
        Cloud: AWS, Azure

        Preferred:
        - Strong knowledge of machine learning and scikit-learn
        - Experience with Agile methodology
        - Excellent project management skills
        """,
        "expected_skills": [
            "etl","sql","python","r","excel","tableau","power bi",
            "data visualization","data warehouse","schema design",
            "forecasting","predictive analytics","data storytelling",
            "data reliability","data accuracy","requirements gathering",
            "scope management","snowflake","git","jira","confluence",
            "aws","azure","machine learning","scikit-learn","agile",
            "project management","data integration","reporting",
            "business intelligence",
        ],
    },
]


def run_test(job):
    expected = set(apply_skill_aliases(job["expected_skills"]))
    extracted = set(extract_exact_job_skills(job["title"], job["description"]))
    found = extracted & expected
    missed = expected - extracted
    unexpected = extracted - expected
    return {
        "name": job["name"],
        "extracted": sorted(extracted),
        "expected": sorted(expected),
        "found": sorted(found),
        "missed": sorted(missed),
        "unexpected": sorted(unexpected),
        "coverage": len(found)/max(len(expected),1)*100,
    }


def main():
    print("="*70)
    print("  SKILL EXTRACTION TEST SUITE")
    print("  Testing against 3 sample data analyst job descriptions")
    print("="*70)

    total_expected = 0
    total_found = 0
    all_missed = set()

    for job in SAMPLE_JOBS:
        result = run_test(job)
        total_expected += len(result["expected"])
        total_found += len(result["found"])
        all_missed.update(result["missed"])

        print(f"\n{'-'*70}")
        print(f"  TEST: {result['name']}")
        print(f"{'-'*70}")
        print(f"  Expected: {len(result['expected'])}  |  Extracted: {len(result['extracted'])}  |  Coverage: {result['coverage']:.1f}%")

        print(f"\n  FOUND ({len(result['found'])}):")
        for s in result["found"]:
            print(f"      [OK] {display_skill(s)}")

        if result["missed"]:
            print(f"\n  MISSED ({len(result['missed'])}):")
            for s in result["missed"]:
                print(f"      [!!] {display_skill(s)}")
        else:
            print(f"\n  No skills missed!")

        if result["unexpected"]:
            print(f"\n  EXTRA ({len(result['unexpected'])}):")
            for s in result["unexpected"]:
                print(f"      [++] {display_skill(s)}")

    coverage = total_found/max(total_expected,1)*100
    print(f"\n{'='*70}")
    print(f"  OVERALL: {total_found}/{total_expected} = {coverage:.1f}%")
    if coverage >= 90: print(f"  PASS")
    elif coverage >= 75: print(f"  PARTIAL")
    else: print(f"  FAIL")

    if all_missed:
        print(f"\n  All missed skills: {', '.join(display_skill(s) for s in sorted(all_missed))}")
    print(f"{'='*70}\n")
    return coverage >= 75

if __name__ == "__main__":
    exit(0 if main() else 1)
