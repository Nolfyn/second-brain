"""
Определение языка запроса для предотвращения внедрения системных инструкций
(system prompt injection). Используется прямая проверка символов Unicode.
"""


def detect_language(text: str) -> str:
    """
    Определяет язык запроса на основе диапазонов символов Unicode.
    Возвращает 'Russian', если текст содержит кириллические символы,
    в противном случае - 'English'.
    """
    for char in text:
        cp = ord(char)
        # Cyrillic block: U+0400-U+04FF
        if 0x0400 <= cp <= 0x04FF:
            return "Russian"
    return "English"


def language_instruction(query: str) -> str:
    """
    Возвращает языковую инструкцию, которую нужно добавить в начало любого
    системного промпта. Она размещается первой, чтобы иметь максимальный
    приоритет по сравнению со всеми остальными правилами.
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
