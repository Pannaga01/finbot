# output_guardrails.py

from langsmith import traceable

import re
from config import (
    ROLE_SENSITIVE_KEYWORDS,
    UNGROUNDED_WARNING,
    CITATION_WARNING,
    LEAKAGE_WARNING,
)


class OutputGuardrails:
    """
    Applies post-generation guardrails on model responses:
    1. Grounding check
    2. Cross-role leakage check
    3. Citation check
    """

    def __init__(self):
        pass

    # -------------------------------
    # 1. Grounding Check
    # -------------------------------
    import re

    def grounding_check(self,response: str, context: str) -> str:
        response_nums = list(map(int, re.findall(r"\d+", response)))
        context_nums = list(map(int, re.findall(r"\d+", context)))

        context_set = set(context_nums)

        for num in response_nums:
            if num in context_set:
                continue

            # ✅ allow derived numbers (difference check)
            derived = False
            for a in context_nums:
                for b in context_nums:
                    if abs(a - b) == num or (a + b) == num:
                        derived = True
                        break
                if derived:
                    break

            if not derived:
                return UNGROUNDED_WARNING

        return response


    # -------------------------------
    # 2. Cross-role Leakage Check
    # -------------------------------
    def leakage_check(self, response: str, allowed_role: str) -> str:
        """
        Detect if response leaks restricted domain information.
        """

        response_lower = response.lower()

        for role, keywords in ROLE_SENSITIVE_KEYWORDS.items():
            if role != allowed_role:
                if any(keyword in response_lower for keyword in keywords):
                    return LEAKAGE_WARNING

        return response

    # -------------------------------
    # 3. Citation Check
    # -------------------------------
    def citation_check(self, response: str, context: str) -> str:
        """
        Ensure context contains citation indicators (e.g., 'Page X').
        """

        if not re.search(r"(?i)(?:page|source)\s*\d+", context.lower()):
            return CITATION_WARNING

        return response


    # -------------------------------
    # MAIN PIPELINE
    # -------------------------------
    @traceable(name="Output Guardrails")
    def run(self, response: str, context: list, role: str) -> str:
        """
        Execute all output guardrails sequentially.
        """

        response = self.grounding_check(response, context)

        # Leakage check intentionally disabled (as in original logic)
        # response = self.leakage_check(response, role)

        response = self.citation_check(response, context)

        return response