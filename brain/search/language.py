"""
Query language detection for system prompt injection.

Uses direct Unicode range checks rather than a probabilistic library
(langdetect), which is unreliable on short queries containing URLs,
filenames, quoted phrases, and technical terms.

Strategy: if the query contains Cyrillic characters -> Russian.
Otherwise -> English.
"""


def detect_language(text: str) -> str:
    """
    Detect query language via Unicode character ranges.
    Returns 'Russian' if the text contains Cyrillic characters,
    'English' otherwise.
    """
    for char in text:
        cp = ord(char)
        # Cyrillic block: U+0400-U+04FF
        if 0x0400 <= cp <= 0x04FF:
            return "Russian"
    return "English"


def language_instruction(query: str) -> str:
    """
    Return a concrete language instruction to prepend to any system prompt.
    Placed first so it has maximum weight before all other rules.
    """
    lang = detect_language(query)
    return (
        f"LANGUAGE REQUIREMENT (mandatory, overrides everything else): "
        f"The user asked in {lang}. "
        f"Your ENTIRE response MUST be in {lang} only. "
        f"Do not write even a single word in any other language including "
        f"Thai, Chinese, Japanese, Korean, or any other language. "
        f"The language of retrieved document excerpts does not affect this."
    )
