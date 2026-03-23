"""
language.py — Lightweight language detection and prompt instruction helpers.

No external dependencies — Hebrew is detected by Unicode block membership.
Hebrew Unicode block: U+0590–U+05FF

Public API
----------
detect_language(text)            → "Hebrew" | "English"
choose_response_language(q, ctx) → "Hebrew" | "English"
language_instruction(lang)       → strong prompt directive string
"""


def _hebrew_ratio(text: str) -> float:
    """Fraction of alphabetic characters in *text* that are Hebrew."""
    if not text:
        return 0.0
    hebrew = sum(1 for c in text if "\u0590" <= c <= "\u05FF")
    alpha = sum(1 for c in text if c.isalpha())
    return hebrew / alpha if alpha else 0.0


def detect_language(text: str) -> str:
    """
    Return "Hebrew" if the text is predominantly Hebrew, otherwise "English".

    Threshold: Hebrew chars > 20 % of all alphabetic chars.
    This handles mixed content gracefully — a Hebrew lecture transcript
    commonly contains a few English technical terms, names, or numbers.
    """
    return "Hebrew" if _hebrew_ratio(text) > 0.20 else "English"


def choose_response_language(question: str, context: str = "") -> str:
    """
    Decide which language the AI should respond in.

    Priority order (strongest signal first):
      1. User's question — if the student asked in Hebrew, answer in Hebrew.
      2. Context / transcript — if the lecture is Hebrew and the question
         is short or ambiguous, default to Hebrew.
      3. Fall back to English.
    """
    if detect_language(question) == "Hebrew":
        return "Hebrew"
    if context and detect_language(context) == "Hebrew":
        return "Hebrew"
    return "English"


def language_instruction(lang: str) -> str:
    """
    Return a strong prompt directive that tells the LLM which language to use.

    The instruction is deliberately assertive — soft phrases like
    "please respond in Hebrew" are frequently overridden by the model
    when the surrounding context is in English.
    """
    if lang == "Hebrew":
        return (
            "IMPORTANT: Your entire response must be written in Hebrew (עברית). "
            "Do not use English anywhere in your response."
        )
    return (
        "IMPORTANT: Your entire response must be written in English. "
        "Do not use Hebrew anywhere in your response."
    )
