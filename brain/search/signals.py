"""Извлечение сигналов из запроса.

Сигнал - детерминированный якорь: год, URL/домен, фраза в кавычках,
имя файла, тематическая папка. Контентные слова не перечисляются -
их общность определяется частотой в корпусе (is_common), грамматические
слова - списками функциональных слов ниже.
"""

import re
from collections.abc import Callable

_YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
_URL_PATTERN = re.compile(r"https?://[^\s\"'\)]+")
_DOMAIN_PATTERN = re.compile(
    r"\b([a-z0-9-]+\.(?:com|org|net|ru|io|dev|app|info|biz|substack|blog))\b",
    re.IGNORECASE,
)
_QUOTED_PATTERN = re.compile(r'"([^"]{2,})"|(?:(?<=\s)|^)\'([^\']{2,})\'')
_EXT_PATTERN = re.compile(r"[\w\- ]*?\.(docx|doc|pdf)", re.IGNORECASE)
_BARE_SITE = re.compile(
    r"\b(?:from|on|at)\s+([a-z][a-z0-9-]{4,})\b", re.IGNORECASE
)

_FUNC_EN = frozenset("""
i me my myself we our ours ourselves you your yours yourself yourselves
he him his himself she her hers herself it its itself they them their
themselves what which who whom whose this that these those am is are was
were be been being have has had having do does did doing will would shall
should can could may might must ought a an the and but if or nor as while
of at by for with about into through during before after above below to
from up down in out on off over under again then once here there when
where why how all any both each few more most other some such no not only
own same so than too very just now because until against between further
give tell find check note notes think feel main brief get got show look
list name describe explain summarize summarise mention mentions remember
recall reading read writing wrote write written
""".split())

_FUNC_RU = frozenset("""
я мы ты вы он она оно они мой моя моё мои моего моей моих моим мою наш
ваш твой его её их свой своя своё свои своих это этот эта эти того тот та
те все всё весь каждый сама сами другой какой который когда где куда
откуда зачем почему сколько кто что чем как был была было были есть буду
будет бы же ли не ни да нет и а но или либо чтобы хотя если потому
поэтому при про для без над под из у о об от до в на с по к ко между
через около после перед вместо внутри вне против благодаря вдоль мимо
ещё еще уже тоже также только лишь даже вот ведь просто совсем почти
более менее очень так там здесь тут теперь потом затем сейчас всегда
никогда иногда опять снова вдруг вместе конечно именно прямо вокруг
назад вперёд много мало кто-то что-то куда-то давай дай расскажи сделай
опиши покажи спасибо пожалуйста скажи напомни вспомни главные краткую
справку краткий расскажи опиши
""".split())

_REFLECTION_TRIGGERS = frozenset("""
conclusions conclusion reflections reflection thoughts summary overview
impressions impression takeaways takeaway recap review learned
выводы итоги мысли впечатления обзор пережил узнал опыт
""".split())


def extract_year(query: str) -> str | None:
    m = _YEAR_PATTERN.search(query)
    return m.group(1) if m else None


def extract_keywords(query: str) -> list[str]:
    """Точные якоря: URL, домены, фразы в кавычках, 'голые' имена сайтов."""
    kws = []
    kws += _URL_PATTERN.findall(query)
    kws += _DOMAIN_PATTERN.findall(query)
    for dq, sq in _QUOTED_PATTERN.findall(query):
        phrase = dq or sq
        if phrase and not re.search(r"\.(docx|doc|pdf)$", phrase, re.I):
            kws.append(phrase)
    for m in _BARE_SITE.finditer(query):
        word = m.group(1)
        if word.lower() not in _FUNC_EN:
            kws.append(word)
    return _dedup_lower(kws)


def find_named_file(query: str, indexed_paths: list[str]) -> str | None:
    """Имя файла из запроса (с расширением или в кавычках), если оно в индексе."""
    indexed = [p.lower() for p in indexed_paths]

    m = _EXT_PATTERN.search(query)
    if m:
        full = m.group(0).strip()
        for start in range(len(full.split())):
            candidate = " ".join(full.split()[start:])
            if any(candidate.lower() in p for p in indexed):
                return candidate

    for dq, sq in _QUOTED_PATTERN.findall(query):
        phrase = dq or sq
        if phrase and any(phrase.lower() in p for p in indexed):
            return phrase

    return None


def find_topic_scope(query: str, indexed_paths: list[str]) -> str | None:
    """Тематическая папка для запросов-размышлений (выводы, итоги, обзор)."""
    query_lower = query.lower()
    if not any(t in query_lower for t in _REFLECTION_TRIGGERS):
        return None

    paths = [p.replace("\\", "/") for p in indexed_paths]
    candidates = re.findall(r"\b[A-ZА-ЯЁ][a-zа-яёA-Za-z]{2,}\b", query)
    candidates += [
        w
        for w in re.findall(r"\b[а-яё]{3,}\b", query_lower)
        if w not in _FUNC_RU
    ]

    for word in candidates:
        wl = word.lower()
        exact = [p for p in paths if wl in [s.lower() for s in p.split("/")]]
        if exact:
            return word
        docx_folder = [
            p
            for p in paths
            if p.lower().endswith(".docx")
            and wl in p.rsplit("/", 1)[0].lower()
        ]
        if docx_folder:
            return word

    return None


def extract_entities(
    query: str, is_common: Callable[[str], bool]
) -> list[str]:
    """Кандидаты для keyword-буста.
    is_common(word_lower) - True, если слово слишком частотное для корпуса.
    """
    entities: list[str] = []

    for dq, sq in _QUOTED_PATTERN.findall(query):
        entities.append(dq or sq)

    for m in re.finditer(
        r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z0-9][a-zA-Z0-9]*){0,3})\b", query
    ):
        words = m.group(1).split()
        if all(w.lower() in _FUNC_EN for w in words):
            continue
        entities.append(m.group(1))

    for m in re.finditer(r"\b([А-ЯЁ][а-яё]{3,})\b", query):
        entities.append(m.group(1))

    for word in re.findall(r"\b([a-zа-яё]{4,})\b", query):
        wl = word.lower()
        if wl not in _FUNC_EN and wl not in _FUNC_RU and not is_common(wl):
            entities.append(word)

    return _dedup_lower(entities)


def _dedup_lower(items: list[str]) -> list[str]:
    seen = set()
    return [
        x for x in items if not (x.lower() in seen or seen.add(x.lower()))
    ]
