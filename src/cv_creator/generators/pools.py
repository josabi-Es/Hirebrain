"""Minimal English pools for base CV generation."""

# Explicit levels for titles where keyword heuristics are ambiguous (0=junior … 4=exec).
TITLE_LEVEL_OVERRIDES: dict[str, int] = {
    "Junior Software Engineer": 0,
    "Junior Data Analyst": 0,
    "Junior Financial Analyst": 0,
    "HR Assistant": 0,
    "Sales Development Representative": 0,
    "Software Engineer": 1,
    "Data Analyst": 1,
    "Financial Analyst": 1,
    "Backend Developer": 1,
    "Frontend Developer": 1,
    "Full Stack Developer": 1,
    "DevOps Engineer": 1,
    "Site Reliability Engineer": 1,
    "Data Engineer": 1,
    "Machine Learning Engineer": 1,
    "HR Generalist": 1,
    "Account Executive": 1,
    "Senior Software Engineer": 2,
    "Senior Data Analyst": 2,
    "Senior Financial Analyst": 2,
    "Senior Account Executive": 2,
    "Software Engineering Manager": 3,
    "Data Science Manager": 3,
    "HR Manager": 3,
    "Finance Manager": 3,
    "Sales Manager": 3,
    "Solutions Architect": 3,
    "Analytics Lead": 3,
    "Analytics Manager": 3,
    "Senior Business Intelligence Analyst": 2,
    "Business Intelligence Analyst": 1,
    "Director of Human Resources": 4,
    "Director of Finance": 4,
    "Regional Sales Director": 4,
}


def role_level(job_title: str) -> int:
    """
    Map a job title to a seniority band (0=junior … 4=executive).

    Uses explicit overrides first, then keyword heuristics.
    """
    if job_title in TITLE_LEVEL_OVERRIDES:
        return TITLE_LEVEL_OVERRIDES[job_title]

    lowered = job_title.lower()
    if any(k in lowered for k in ("director", "head", "vice president", "vp", "chief")):
        return 4
    if any(k in lowered for k in ("manager", "lead", "principal", "architect")):
        return 3
    if "senior" in lowered:
        return 2
    if any(k in lowered for k in ("junior", "intern", "entry", "assistant")):
        return 0
    if "representative" in lowered and "senior" not in lowered:
        return 0
    return 1


DOMAIN_POOLS: dict[str, dict[str, list[str]]] = {
    "Software Engineering": {
        "job_titles": [
            "Junior Software Engineer",
            "Software Engineer",
            "Senior Software Engineer",
            "Backend Developer",
            "Frontend Developer",
            "Full Stack Developer",
            "DevOps Engineer",
            "Site Reliability Engineer",
            "Software Engineering Manager",
            "Solutions Architect",
        ],
        "skills": [
            "Python",
            "Java",
            "JavaScript",
            "TypeScript",
            "Go",
            "C#",
            "SQL",
            "PostgreSQL",
            "React",
            "Node.js",
            "Django",
            "FastAPI",
            "REST APIs",
            "Microservices",
            "Docker",
            "Kubernetes",
            "AWS",
            "CI/CD",
            "Git",
            "System Design",
        ],
    },
    "Data Science": {
        "job_titles": [
            "Junior Data Analyst",
            "Data Analyst",
            "Senior Data Analyst",
            "Business Intelligence Analyst",
            "Data Engineer",
            "Machine Learning Engineer",
            "Data Science Manager",
            "Analytics Manager",
            "Analytics Lead",
            "Senior Business Intelligence Analyst",
        ],
        "skills": [
            "Python",
            "SQL",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Apache Spark",
            "ETL",
            "Power BI",
            "Tableau",
            "A/B Testing",
            "Statistics",
            "Feature Engineering",
            "Model Monitoring",
            "Data Visualization",
            "Forecasting",
            "Experiment Design",
        ],
    },
    "Human Resources": {
        "job_titles": [
            "HR Assistant",
            "HR Generalist",
            "Talent Acquisition Specialist",
            "Technical Recruiter",
            "People Operations Specialist",
            "Compensation and Benefits Analyst",
            "HR Business Partner",
            "HR Manager",
            "People Operations Manager",
            "Director of Human Resources",
        ],
        "skills": [
            "Applicant Tracking Systems",
            "Talent Sourcing",
            "Interviewing",
            "Employee Onboarding",
            "Performance Management",
            "Compensation Benchmarking",
            "Labor Law Compliance",
            "Conflict Resolution",
            "HRIS",
            "Workforce Planning",
            "Employee Engagement",
            "Policy Development",
            "Stakeholder Management",
            "Communication",
            "Coaching",
        ],
    },
    "Finance": {
        "job_titles": [
            "Junior Financial Analyst",
            "Financial Analyst",
            "Senior Financial Analyst",
            "FP&A Analyst",
            "Accounting Specialist",
            "Finance Manager",
            "Controller",
            "Director of Finance",
        ],
        "skills": [
            "Financial Modeling",
            "Budgeting",
            "Forecasting",
            "Variance Analysis",
            "Excel",
            "Power BI",
            "SQL",
            "ERP Systems",
            "Accounting Reconciliation",
            "Cash Flow Analysis",
            "Cost Optimization",
            "Financial Reporting",
            "Risk Assessment",
            "Internal Controls",
            "Business Partnering",
        ],
    },
    "Sales": {
        "job_titles": [
            "Sales Development Representative",
            "Account Executive",
            "Senior Account Executive",
            "Key Account Manager",
            "Business Development Manager",
            "Sales Operations Analyst",
            "Sales Manager",
            "Regional Sales Director",
        ],
        "skills": [
            "Prospecting",
            "Pipeline Management",
            "CRM",
            "Lead Qualification",
            "Negotiation",
            "Territory Planning",
            "Account Management",
            "Sales Forecasting",
            "Revenue Analysis",
            "Cold Outreach",
            "Solution Selling",
            "Objection Handling",
            "Presentation Skills",
            "Stakeholder Management",
            "Contract Management",
        ],
    },
}

DOMAIN_WEIGHTS = {
    # Favor technical profiles during random domain selection.
    "Software Engineering": 6,
    "Data Science": 4,
    "Human Resources": 1,
    "Finance": 1,
    "Sales": 1,
}

DOMAINS = [
    domain
    for domain, weight in DOMAIN_WEIGHTS.items()
    if domain in DOMAIN_POOLS
    for _ in range(weight)
]

COMPANIES = [
    "TechCorp Solutions",
    "DataFlow Inc",
    "CloudNine Systems",
    "InnovateLabs",
    "Nexus Technologies",
    "Vertex Analytics",
    "Apex Digital",
    "Horizon Software",
    "BluePeak Consulting",
    "Summit IT Services",
    "Quantum Leaf",
    "Sterling Partners",
    "Meridian Systems",
    "Northbridge Analytics",
    "BrightPath Health Tech",
    "Fusion Retail Group",
    "GreenEarth Logistics",
    "Atlas Fintech",
    "Pioneer Automation",
    "Cobalt Security",
    "Lumen Networks",
    "Orbit Media Labs",
    "Riverstone Energy IT",
    "Silverline Insurance Tech",
    "CoreStack Platforms",
    "Helix Biotech",
    "UrbanGrid Mobility",
    "Pacific Trade Systems",
    "Echo Communications",
    "Vantage ERP Solutions",
]

DEGREES = [
    "B.S. in Computer Science",
    "B.S. in Software Engineering",
    "B.S. in Information Systems",
    "B.S. in Information Technology",
    "B.S. in Data Science",
    "B.S. in Statistics",
    "B.S. in Mathematics",
    "B.S. in Electrical Engineering",
    "B.S. in Cybersecurity",
    "B.S. in Business Administration",
    "B.B.A. in Business Administration",
    "B.A. in Economics",
    "B.S. in Finance",
    "B.S. in Industrial Engineering",
    "B.S. in Computer Engineering",
    "M.S. in Computer Science",
    "M.S. in Data Science",
    "M.S. in Information Systems",
    "M.B.A.",
    "Associate Degree in Computer Programming",
]

SKILLS = [
    # Backward-compat alias for callers still expecting a flat skills list.
    skill
    for domain in DOMAIN_POOLS.values()
    for skill in domain["skills"]
]

JOB_TITLES = [
    # Backward-compat alias for callers still expecting a flat title list.
    title
    for domain in DOMAIN_POOLS.values()
    for title in domain["job_titles"]
]
