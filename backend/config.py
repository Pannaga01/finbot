# RBAC collection mapping by role
ROLE_COLLECTIONS = {
    "employee": ["general"],
    "hr": ["general", "hr"],
    "finance": ["general", "finance"],
    "engineering": ["general", "engineering"],
    "marketing": ["general", "marketing"],
    "c_level": ["general", "finance", "engineering", "marketing"],
}

SENTENCE_TRANSFORMER_LLM = "all-MiniLM-L6-v2"

ROLE_ACCESS = {
    "general" : ["employee", "hr", "finance", "engineering", "marketing", "c_level"],
    "hr" : ["hr", "c_level"],
    "finance" : ["finance", "c_level"],
    "engineering" : ["engineering", "c_level"],
    "marketing" : ["marketing", "c_level"]
}

PII_PATTERNS = {
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "email": r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
    "phone": r"\b[6-9]\d{9}\b",
    "bank_account": r"\b\d{9,18}\b",
}

INJECTION_PATTERNS = [
    r"ignore .* instructions",
    r"bypass .* security",
    r"act as .* admin",
    r"show me all documents",
    r"no restrictions",
    r"override .* system",
]

MAX_QUERIES = 20

COLLECTION_NAME = "finbot"

GROQ_MODEL  = "openai/gpt-oss-safeguard-20b"

SYSTEM_PROMPT = """You are a helpful assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so — do not make things up.
Always cite the section name when referencing specific information."""

ROLE_SENSITIVE_KEYWORDS = {
    "finance": ["budget", "revenue", "profit", "balance sheet"],
    "engineering": ["architecture", "deployment", "pipeline"],
    "hr": ["salary", "employee", "leave", "policy"],
    "marketing": ["campaign", "leads", "engagement"],
}

UNGROUNDED_WARNING = "\n\n⚠️ This response may contain information not found in retrieved documents."
CITATION_WARNING = "\n\n⚠️ No source citations found."
LEAKAGE_WARNING = "\n\n⚠️ Potential cross-role data leakage detected."

QUESTIONS = [
    # ===== DEPARTMENT BUDGET & VARIANCE ANALYSIS (Finance/Operations) =====
    "What was the total operating expense variance for FinSolve Technologies in 2024?",
    "Which department achieved perfect budget alignment with zero variance in 2024?",
    "What was the primary reason for the Technology department’s overspend in 2024?",
    "How many engineers were actually employed in the Technology department versus budgeted?",
    "What was the Finance department’s total variance and overall status?",
    "Which department had the highest controlled overspend percentage?",
    "What was the total headcount of FinSolve Technologies at the end of 2024?",
    "Which department achieved 104% of its revenue target, and what caused its overspend?",
    "What was the ratio of CapEx to OpEx in 2024?",
    "What savings opportunities were identified under 'Immediate Actions (Q1 2025)'?",
    "Which department’s overspend was primarily due to external legal counsel fees?",
    "What was the average compensation cost per employee companywide?",
    "Which department achieved the lowest cost per employee?",
    "What are the top two unfavorable variance drivers identified across departments?",
    "How much was proposed as the total company budget for FY2025?",
    "What was the percentage increase proposed for the HR department in FY2025?",
    "What was the Technology department’s percentage of total budget share in FY2024?",
    "Which department had the highest average salary including benefits?",
    "What was the main favorable factor leading to Finance department’s underspend?",
    "Which specific measure was recommended to reduce Technology cloud costs for 2025?",

    # ===== ENGINEERING MASTER DOCUMENT =====
    "What type of architecture does FinSolve’s system follow?",
    "Which databases are used for transactional data and user profiles?",
    "What is the primary purpose of the API Gateway in FinSolve’s architecture?",
    "Which authentication standard is implemented for user login?",
    "What programming frameworks are used for mobile app development?",
    "What is the targeted maximum P95 latency for the Payment Service?",
    "What is the defined Recovery Time Objective (RTO) for disaster recovery?",
    "Which AWS services are explicitly part of FinSolve’s infrastructure stack?",
    "What is the engineering headcount as of the 2025 Engineering Document?",
    "How many direct reports does the VP Engineering - Backend Systems manage?",

    # ===== EMPLOYEE HANDBOOK =====
    "What is FinSolve’s standard hybrid work model arrangement per week?",
    "During which hours must employees be available under the Core Hours policy?",
    "How many days of annual privilege leave are employees entitled to?",
    "What is the maternity leave duration for the first two children?",
    "How many volunteering days per year are employees eligible for?",
    "What is the maximum reimbursement limit per day for meals during domestic travel for mid-level employees?",
    "How long is the mandatory notice period for managers when resigning?",
    "What is the period employees can work remotely from a different city or country per year?",
    "What is the tuition reimbursement cap per employee annually?",
    "What is FinSolve’s penalty or action for repeated absenteeism without approval?",

    # ===== CAMPAIGN PERFORMANCE DATA =====
    "How many total campaigns did FinSolve execute in 2024?",
    "Which campaign achieved the highest ROI during 2024?",
    "What was the overall average campaign ROI across all campaigns?",
    "Which campaign type achieved the strongest ROI category-wise?",
    "What was the ROI of the underperforming Event Sponsorships - LATAM campaign?",
    "Which A/B test confirmed that longer video duration improved engagement?",
    "At what time of day did email campaigns achieve the highest open rate during holiday campaigns?",
    "What overall ROI improvement did localized content bring compared to translated campaigns?",
    "Which two channels delivered ROI above 3x on average?",
    "What was the main finding regarding B2B messaging strategy effectiveness?"
]

GROUND_TRUTHS = [
    "₹241.6 Crore actual vs ₹241.0 Crore budget, total unfavorable variance ₹0.6 Crore (0.2%).",
    "Administration department achieved perfect budget alignment with zero variance (₹1000L budget and actual).",
    "Higher cloud infrastructure costs for Southeast Asia expansion and enhanced collaboration tools.",
    "Actual 188 engineers vs 185 budgeted.",
    "Total variance ₹13 Lakh underspend (+1.4%), status: Underspend.",
    "Legal & Compliance with -2.8% overspend (₹40 Lakh).",
    "Total headcount 485 employees.",
    "Sales department, due to higher commissions from exceeding revenue targets.",
    "CapEx 13.6% vs OpEx 86.4%.",
    "Cloud contract renegotiation to save ₹150–200L, sales commission plan review, and RPA expansion.",
    "Legal & Compliance overspent ₹20L on outside counsel for international expansion and data privacy.",
    "Average total compensation ₹25.1 Lakh per employee.",
    "Operations department — ₹14.7 Lakh per employee cost, most efficient.",
    "Sales Commissions (₹20L) and Legal Counsel (₹20L) were top unfavorable variance drivers.",
    "FY2025 proposed company total budget ₹27,880 Lakh (+15.2%).",
    "HR department budget proposed +12.0% increase for FY2025.",
    "Approximately 31.1% of total FY2024 company budget.",
    "Sales department, ₹47.9 Lakh average compensation.",
    "Robotic Process Automation (RPA) reduced need for external hires.",
    "Renegotiation of AWS/Azure contracts and use of reserved instances for cost optimization.",

    "A microservices-based, cloud-native architecture.",
    "PostgreSQL for transactional data and MongoDB for user profiles.",
    "To route, authenticate, and rate-limit all client API traffic centrally.",
    "OAuth 2.0 with JWT tokens.",
    "Swift (iOS) and Kotlin (Android).",
    "P95 target latency for Payment Service: 250ms.",
    "Disaster Recovery RTO is 4 hours.",
    "Uses AWS EC2, ECS, Lambda, RDS, S3, and CloudFront services.",
    "Total Engineering headcount: 57.",
    "19 direct reports across backend engineering managers and teams.",

    "Three days in office and two days remote per week.",
    "10:00 AM to 4:00 PM IST are mandatory core hours.",
    "15–21 days per year (state-dependent).",
    "26 weeks paid leave for first two children.",
    "2 paid volunteering days per year.",
    "₹500 per day for meals for junior/mid-level employees.",
    "90 days’ notice period for managers and above.",
    "Up to 30 days per year remotely from another city or country.",
    "Up to ₹50,000 per year for relevant tuition or certification.",
    "Written warning or escalation for repeated unauthorized absenteeism.",

    "28 campaigns executed in 2024.",
    "‘Video Content – Year-End Review’ campaign with 4.58x ROI.",
    "Average campaign ROI: 2.67x.",
    "Acquisition campaigns delivered the highest average ROI (3.26x).",
    "ROI was 0.73x for Event Sponsorships – LATAM (underperformed).",
    "Test 3: Video Content Length confirmed longer (60s) videos drove 73% higher engagement.",
    "Evening send time (7 PM) achieved the highest open rates and clicks.",
    "Localized content produced 40–80% performance improvement over direct translations.",
    "Digital Ads and Video/Social channels both averaged above 3x ROI.",
    "Outcome-focused messaging improved LinkedIn reply rates by 81% over feature-based messaging."
]

mock_ques = [
    "How many total campaigns did FinSolve execute in 2024?",
    "What is FinSolve’s standard hybrid work model arrangement per week?",
    "What type of architecture does FinSolve’s system follow?",
    "What was the total operating expense variance for FinSolve Technologies in 2024?"
]

mock_ground_truth = [
    "28 campaigns executed in 2024.",
    "Three days in office and two days remote per week.",
    "A microservices-based, cloud-native architecture.",
    "Total operating expense variance: ₹13 Lakh underspend (+1.4%), status: Underspend."
]
