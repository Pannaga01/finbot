"""
router.py
---------
Defines the SemanticRouter for FinBot.

Routes:
- engineering
- finance
- general
- hr
- marketing
- off_topic (guardrail)
- potentially_harmful (guardrail)
"""

import re
from collections import defaultdict

from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder

from config import PII_PATTERNS, INJECTION_PATTERNS, MAX_QUERIES


# -------------------------------
# SESSION RATE LIMITING
# -------------------------------
SESSION_COUNTER = defaultdict(int)


def check_rate_limit(session_id: str):
    SESSION_COUNTER[session_id] += 1
    if SESSION_COUNTER[session_id] > MAX_QUERIES:
        raise ValueError(
            "❌ Rate limit exceeded (20 queries per session). Please start a new session."
        )


# -------------------------------
# SECURITY CHECKS
# -------------------------------
def detect_prompt_injection(text: str):
    text_lower = text.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            raise ValueError("❌ Prompt injection detected. Request blocked.")


def detect_pii(text: str):
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            raise ValueError(
                f"❌ Detected sensitive data ({pii_type}). Please remove personal information."
            )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
encoder = HuggingFaceEncoder(name="sentence-transformers/all-mpnet-base-v2")


# ---------------------------------------------------------------------------
# Route Definitions
# ---------------------------------------------------------------------------
engineering = Route(
    name="engineering",
    utterances=[
        "What is the current status of the production deployment?",
        "Show me the latest system architecture diagram",
        "How does the microservices setup work?",
        "What tech stack are we using for the backend?",
        "Explain the CI/CD pipeline for the project",
        "What are the open engineering tickets this sprint?",
        "How do I set up the local development environment?",
        "What is the SLA for our API uptime?",
        "What is the software development lifecycle?",
        "Explain SDLC",
    ],
    score_threshold=0.3,
)

finance = Route(
    name="finance",
    utterances=[
        "What is the company's total revenue this quarter?",
        "Show me the profit and loss statement",
        "What is the operating budget for this year?",
        "Give me the balance sheet summary",
        "What are the company's financial statements?",
        "How much cash runway do we have?",
        "What are the total expenses by department?",
        "Explain the variance in financial expenses",
        "What are the key financial KPIs like profit margin and EBITDA?",
        "What is the company's net profit?",
        "Show me accounting reports for this quarter",
        "What are the capital expenditures this year?",
    ],
)

general = Route(
    name="general",
    utterances=[
        "What is company's vision and mission?",
        "Explain company core values like integrity and innovation",
        "What is the onboarding process for new employees?",
        "Which statutory benefits are provided such as EPF or ESI?",
        "What wellness programs and insurance benefits are offered?",
        "How many types of leave are available?",
        "What is the leave application process in HRMS?",
        "What are the standard work hours?",
        "How is overtime handled?",
        "What is the dress code policy?",
        "How are harassment complaints handled?",
        "What are workplace safety guidelines?",
        "What is IT usage policy?",
        "How are salaries structured?",
        "What is the resignation process?",
    ],
)

hr = Route(
    name="hr",
    utterances=[
        "What is the leave balance for Pavan Krishnan?",
        "How many leaves has Pavan Krishnan taken?",
        "What is the employment status of the HR Executive?",
        "Who is the HR Executive in Lucknow?",
        "How can I contact the HR Executive?",
        "How many female employees are in HR?",
        "What is the average salary for Senior HR Executives?",
        "How many employees are in Lucknow?",
        "What is the total leave balance for employees in Technology department?",
        "How many employees have performance rating above 4?",
        "How many employees are in the Finance department?",
    ],
    score_threshold=0.3,
)

marketing = Route(
    name="marketing",
    utterances=[
        "What is marketing campaign performance?",
        "Show social media engagement metrics",
        "How many leads were generated?",
        "What is cost per lead?",
        "Which channels drive conversions?",
        "What is campaign ROI?",
        "Show website traffic stats",
        "Email campaign performance",
        "Customer acquisition cost",
        "Content calendar",
        "Ad campaign analytics",
        "Engagement rates",
    ],
)

off_topic_route = Route(
    name="off_topic",
    utterances=[
        "Tell me a joke",
        "What's the weather today?",
        "How do I bake a cake?",
        "Who won the football game?",
        "Write me a poem",
        "Help me with homework",
        "Tell me about history",
        "How do I party at Nvidia headquarters",
    ],
)

harmful_route = Route(
    name="potentially_harmful",
    utterances=[
        "How can I hide losses in financial reports?",
        "How to manipulate financial data?",
        "Ways to adjust numbers to avoid audit",
        "How to bypass compliance?",
        "How to get around internal controls?",
        "How to access restricted data?",
        "How to view unauthorized documents?",
    ],
)


# ---------------------------------------------------------------------------
# Assemble Routes
# ---------------------------------------------------------------------------
routes = [
    engineering,
    finance,
    general,
    hr,
    marketing,
    off_topic_route,
    harmful_route,
]


# ---------------------------------------------------------------------------
# Router Instance
# ---------------------------------------------------------------------------
rl = SemanticRouter(
    encoder=encoder,
    routes=routes,
    auto_sync="local",
)


# ---------------------------------------------------------------------------
# Guardrail Wrapper
# ---------------------------------------------------------------------------
OFF_TOPIC_REPLY = (
    "❌ I can only help with company-related queries across Engineering, "
    "Finance, HR, or Marketing. Please rephrase your question."
)

HARMFUL_REPLY = "❌ I cannot help with that request."


class SimpleSemanticRouterGuardrail:
    """
    Wrapper over SemanticRouter with:
    - Rate limiting
    - PII detection
    - Prompt injection protection
    - Off-topic & harmful filtering
    """

    def __init__(self, semantic_router: SemanticRouter):
        self.router = semantic_router

    def __call__(self, user_input: str, session_id: str = None) -> tuple:
        if session_id:
            check_rate_limit(session_id)

        detect_pii(user_input)
        detect_prompt_injection(user_input)

        print(f"Routing query: '{user_input}'")

        if not user_input or not user_input.strip():
            return "hr_general", False, None

        result = self.router(user_input)

        print(f"Result: {result}")
        if result:
            print(f"Name: {result.name}, Score: {getattr(result, 'score', None)}")

        if result and result.name == "off_topic":
            return "off_topic", True, OFF_TOPIC_REPLY

        if result and result.name == "potentially_harmful":
            return "potentially_harmful", True, HARMFUL_REPLY

        route_name = result.name if (result and result.name) else "hr_general"
        return route_name, False, None


# ---------------------------------------------------------------------------
# Singleton Instance
# ---------------------------------------------------------------------------
guardrail = SimpleSemanticRouterGuardrail(rl)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def route_query(query: str, session_id: str = None) -> str:
    """
    Route query through guardrails.

    Raises:
        ValueError → if blocked
    """

    print(f"Received query: '{query}'")
    route_name, blocked, reply = guardrail(query, session_id)

    print(f"Routed to: '{route_name}', Blocked: {blocked}")

    if blocked:
        raise ValueError(reply)

    return route_name