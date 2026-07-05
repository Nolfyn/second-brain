"""
Retriever: single entry point for finding relevant document chunks.

Priority order:
1. Exact signals (URLs, domains, quoted phrases, bare site names)
   -> keyword search, optionally scoped to a named file
2. Explicit filename (.docx/.pdf extension or quoted path)
   -> file-scoped retrieval
3. Year mention (4-digit year in query)
   -> year-scoped search, returning all chunks from that year
4. Reflection/conclusion query about a known topic folder
   -> topic-scoped search within that folder's .docx files
5. Everything else
   -> semantic search with LangChain MultiQueryRetriever expansion,
      augmented by keyword search for any specific named things in query

Design principle: no hardcoded word lists for content (activities, topics,
translations). Either use the embedding model (semantic search), exact match
(keyword search), or structural signals (year, filename, folder name).
"""

import os
import re
from collections.abc import Generator

from brain.index import Embedder, VectorStore, DocumentStore, SearchResult


# ---------------------------------------------------------------------------
# Compiled patterns for signal extraction
# ---------------------------------------------------------------------------

_YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
_URL_PATTERN = re.compile(r"https?://[^\s\"'\)]+")
_DOMAIN_PATTERN = re.compile(
    r"\b([a-z0-9-]+\.(?:com|org|net|ru|io|substack|blog))\b",
    re.IGNORECASE,
)
_QUOTED_PATTERN = re.compile(r'"([^"]{2,})"|(?:(?<=\s)|^)\'([^\']{2,})\'')
_EXT_PATTERN = re.compile(r"[\w\- ]*?\.(docx|doc|pdf)", re.IGNORECASE)

_BARE_SITE = re.compile(
    r"\b(?:from|on|at)\s+([a-z][a-z0-9-]{4,})\b", re.IGNORECASE
)
_NOT_SITES = {
    "from",
    "have",
    "read",
    "articles",
    "were",
    "where",
    "which",
    "doing",
    "what",
    "that",
    "this",
    "with",
    "been",
    "more",
    "when",
    "them",
    "they",
    "here",
    "there",
    "about",
    "after",
    "back",
    "only",
    "some",
    "also",
    "just",
    "your",
    "their",
}

_REFLECTION_TRIGGERS = {
    "conclusions",
    "conclusion",
    "reflections",
    "reflection",
    "thoughts",
    "summary",
    "выводы",
    "выводам",
    "мысли",
    "итоги",
    # impression / experience / overview style queries also want topic scoping
    "impressions",
    "impression",
    "overview",
    "experience",
    "experienced",
    "recap",
    "takeaways",
    "takeaway",
    "review",
    "learned",
    "впечатления",
    "впечатлениях",
    "впечатлений",
    "пережил",
    "узнал",
    "опыт",
    "обзор",
}

# Maps Russian country/topic names to their English folder-name equivalents,
# so a Russian query can match an English-named folder. These are illustrative
# common examples.
_TOPIC_TRANSLATIONS = {
    "китай": "china",
    "япония": "japan",
    "корея": "korea",
    "германия": "germany",
    "франция": "france",
    "италия": "italy",
    "испания": "spain",
    "турция": "turkey",
    "индия": "india",
    "египет": "egypt",
}

_QUESTION_WORDS = {
    "What",
    "When",
    "Where",
    "Who",
    "Why",
    "How",
    "Did",
    "Does",
    "Have",
    "Has",
    "Was",
    "Were",
    "Can",
    "Could",
    "Would",
    "Should",
    "Is",
    "Are",
    "Do",
    "The",
    "My",
    "Your",
    "About",
    "Give",
    "Tell",
    "Find",
    "Check",
    "Note",
    "Notes",
    "Think",
    "Feel",
    "Experience",
    "Impression",
    "Thought",
    "Main",
    "Brief",
    "Some",
    "Any",
    "All",
    "Get",
    "Got",
}


class Retriever:
    MAX_RESULTS = 20
    MAX_FILE = 60
    MAX_YEAR = 60
    MAX_KEYWORD = 30
    SMALL_FILE = 60
    MIN_SCORE = 0.35
    SEMANTIC_TOPK = 15

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        document_store: DocumentStore | None = None,
    ):
        self._emb = embedder or Embedder()
        self._vs = vector_store or VectorStore()
        self._ds = document_store or DocumentStore()
        self._ignore_patterns = self._load_ignore_patterns()

    def _load_ignore_patterns(self) -> list[str]:
        """
        Load glob patterns from .brainignore (like .gitignore). Files matching
        any pattern are filtered from general search results -- useful for
        reference books and other non-personal content that pollutes answers.
        The files stay indexed, so explicit "Check <filename>" queries still
        reach them; they're only hidden from semantic/keyword/topic results.
        """
        candidates = [
            os.path.join(os.getcwd(), ".brainignore"),
            os.path.join(
                os.path.dirname(__file__), "..", "..", ".brainignore"
            ),
        ]
        for path in candidates:
            if os.path.exists(path):
                patterns = []
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            patterns.append(line.lower())
                print(
                    f"[RETRIEVER] loaded {len(patterns)} .brainignore patterns"
                )
                return patterns
        return []

    def _is_ignored(self, file_path: str) -> bool:
        if not self._ignore_patterns:
            return False
        import fnmatch

        p = file_path.replace("\\", "/").lower()
        basename = p.rsplit("/", 1)[-1]
        for pat in self._ignore_patterns:
            if (
                fnmatch.fnmatch(p, pat)
                or fnmatch.fnmatch(basename, pat)
                or pat in p
            ):
                return True
        return False

    def _filter_ignored(
        self, results: list[SearchResult]
    ) -> list[SearchResult]:
        if not self._ignore_patterns:
            return results
        return [r for r in results if not self._is_ignored(r.file_path)]

    def retrieve(self, query: str) -> list[SearchResult]:
        year = _YEAR_PATTERN.search(query)
        named_file = self._named_file(query)
        kws = self._exact_keywords(query)
        topic = self._topic_scope(query)

        print(
            f"[RETRIEVER] kws={kws} file={named_file!r} "
            f"year={year and year.group(1)!r} topic={topic!r}"
        )

        if kws:
            results = self._keyword_search(kws, scope_file=named_file)
            if results:
                return self._filter_ignored(results)[: self.MAX_KEYWORD]

        # Explicit named file: do NOT filter -- the user asked for it by name
        if named_file:
            return self._file_search(query, named_file)

        if year:
            return self._filter_ignored(
                self._year_search(query, year.group(1))
            )[: self.MAX_YEAR]

        if topic:
            return self._filter_ignored(self._topic_search(query, topic))[
                : self.MAX_FILE
            ]

        return self._filter_ignored(self._semantic_with_entities(query))[
            : self.MAX_RESULTS
        ]

    def _keyword_search(
        self, kws: list[str], scope_file: str | None
    ) -> list[SearchResult]:
        chunks = []
        for kw in kws:
            found = self._ds.keyword_search(kw, limit=self.MAX_KEYWORD * 2)
            if scope_file:
                found = [
                    c
                    for c in found
                    if scope_file.lower() in c.file_path.lower()
                ]
            chunks.extend(found)

        seen, unique = set(), []
        for c in chunks:
            k = (c.file_path, c.chunk_index)
            if k not in seen:
                seen.add(k)
                unique.append(c)
        unique.sort(key=lambda c: (c.file_path, c.chunk_index))
        return self._to_results(unique)

    def _file_search(self, query: str, named_file: str) -> list[SearchResult]:
        chunks = self._ds.get_chunks_for_file_matching(named_file)
        print(f"[RETRIEVER] file '{named_file}' -> {len(chunks)} chunks")
        if not chunks:
            return []
        if len(chunks) <= self.SMALL_FILE:
            return self._to_results(chunks)
        vec = self._emb.embed_one(query)
        pool = self._vs.search(vec, top_k=300)
        in_file = [
            r for r in pool if named_file.lower() in r.file_path.lower()
        ]
        return in_file[: self.MAX_FILE]

    def _year_search(self, query: str, year: str) -> list[SearchResult]:
        all_paths = self._ds.list_file_paths()
        year_paths = [p for p in all_paths if year in p]
        print(f"[RETRIEVER] year {year} -> {len(year_paths)} files")

        if not year_paths:
            return self._semantic_with_entities(query)

        _REVIEW = {
            "review",
            "summary",
            "overview",
            "recap",
            "справку",
            "итог",
            "обзор",
            "расскажи",
            "doing",
            "занимался",
            "занималась",
        }
        is_review = any(t in query.lower() for t in _REVIEW)

        vec = self._emb.embed_one(query)
        pool = self._vs.search(vec, top_k=300)
        in_year = [r for r in pool if year in r.file_path]

        if is_review or not in_year:
            all_chunks = []
            for p in sorted(year_paths):
                fname = p.replace("\\", "/").split("/")[-1]
                chunks = self._ds.get_chunks_for_file_matching(fname)
                all_chunks.extend(chunks)
                if len(all_chunks) >= self.MAX_YEAR:
                    break
            return self._to_results(all_chunks[: self.MAX_YEAR])

        return in_year[: self.MAX_YEAR]

    def _topic_search(self, query: str, topic: str) -> list[SearchResult]:
        all_paths = self._ds.list_file_paths()
        topic_paths = [p for p in all_paths if topic.lower() in p.lower()]
        print(f"[RETRIEVER] topic '{topic}' -> {len(topic_paths)} files")

        if not topic_paths:
            return self._semantic_with_entities(query)

        docx_paths = [p for p in topic_paths if p.lower().endswith(".docx")]

        vec = self._emb.embed_one(query)
        pool = self._vs.search(vec, top_k=300)
        docx_results = [
            r
            for r in pool
            if topic.lower() in r.file_path.lower()
            and r.file_path.lower().endswith(".docx")
        ]

        if docx_results:
            target_files = list({r.file_path for r in docx_results})
            for stem in ("вывод", "итог", "заключени", "резюм"):
                kw_chunks = self._ds.keyword_search_in_files(
                    stem, target_files
                )
                for c in kw_chunks:
                    key = (c.file_path, c.chunk_index)
                    if key not in {
                        (r.file_path, r.chunk_index) for r in docx_results
                    }:
                        docx_results.insert(
                            0,
                            SearchResult(
                                chunk_id=f"{c.file_path}::{c.chunk_index}",
                                text=c.raw_text,
                                file_path=c.file_path,
                                file_type="docx",
                                chunk_index=c.chunk_index,
                                score=0.95,
                            ),
                        )

            print(
                f"[RETRIEVER] topic docx: {len(docx_results)} chunks, "
                f"top scores: {[round(r.score, 2) for r in docx_results[:5]]}"
            )
            return docx_results[: self.MAX_FILE]

        all_chunks = []
        for p in sorted(docx_paths):
            fname = p.replace("\\", "/").split("/")[-1]
            chunks = self._ds.get_chunks_for_file_matching(fname)
            all_chunks.extend(chunks)
            if len(all_chunks) >= self.MAX_FILE:
                break
        return self._to_results(all_chunks[: self.MAX_FILE])

    def _semantic_with_entities(self, query: str) -> list[SearchResult]:
        queries = self._expand_query(query)
        all_results: dict[tuple, SearchResult] = {}

        for q in queries:
            vec = self._emb.embed_one(q)
            results = self._vs.search(vec, top_k=self.SEMANTIC_TOPK)
            for r in results:
                if r.score >= self.MIN_SCORE:
                    key = (r.file_path, r.chunk_index)
                    if (
                        key not in all_results
                        or r.score > all_results[key].score
                    ):
                        all_results[key] = r

        sem = sorted(all_results.values(), key=lambda r: r.score, reverse=True)
        print(
            f"[RETRIEVER] semantic ({len(queries)} queries) -> {len(sem)} results, "
            f"top scores: {[round(r.score, 2) for r in sem[:5]]}"
        )

        entities = self._named_entities(query)
        if entities:
            print(f"[RETRIEVER] named entities for keyword boost: {entities}")
            kw_chunks = []
            for entity in entities:
                hits = self._ds.keyword_search(entity, limit=200)
                if len(hits) <= 15:
                    kw_chunks.extend(hits)
                    print(
                        f"[RETRIEVER]   entity '{entity}': {len(hits)} hits (kept)"
                    )
                else:
                    print(
                        f"[RETRIEVER]   entity '{entity}': {len(hits)} hits (too common, skipped)"
                    )
            seen_kw: set[tuple] = set()
            unique_kw = []
            for c in kw_chunks:
                k = (c.file_path, c.chunk_index)
                if k not in seen_kw:
                    seen_kw.add(k)
                    unique_kw.append(c)

            if unique_kw:
                kw_results = self._to_results(unique_kw)
                sem_seen = {(r.file_path, r.chunk_index) for r in kw_results}
                extra_sem = [
                    r
                    for r in sem
                    if (r.file_path, r.chunk_index) not in sem_seen
                ]
                merged = kw_results + extra_sem
                print(
                    f"[RETRIEVER] entity keyword: {len(kw_results)} hits "
                    f"+ {len(extra_sem)} semantic = {len(merged)} total"
                )
                return merged[: self.MAX_RESULTS]

        return sem[: self.MAX_RESULTS]

    def _exact_keywords(self, query: str) -> list[str]:
        kws = []
        kws += _URL_PATTERN.findall(query)
        kws += _DOMAIN_PATTERN.findall(query)
        for dq, sq in _QUOTED_PATTERN.findall(query):
            phrase = dq or sq
            if phrase and not re.search(r"\.(docx|doc|pdf)$", phrase, re.I):
                kws.append(phrase)
        for m in _BARE_SITE.finditer(query):
            word = m.group(1)
            if word.lower() not in _NOT_SITES:
                kws.append(word)
        seen = set()
        return [
            k for k in kws if not (k.lower() in seen or seen.add(k.lower()))
        ]

    def _named_file(self, query: str) -> str | None:
        indexed = [p.lower() for p in self._ds.list_file_paths()]

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

    def _topic_scope(self, query: str) -> str | None:
        query_lower = query.lower()
        if not any(t in query_lower for t in _REFLECTION_TRIGGERS):
            return None

        all_paths = self._ds.list_file_paths()

        candidates = list(
            re.findall(r"\b[A-ZА-Я][a-zа-яёA-Za-z]{2,}\b", query)
        )
        for ru_word in re.findall(r"[а-яёА-ЯЁ]{3,}", query_lower):
            en = _TOPIC_TRANSLATIONS.get(ru_word)
            if en:
                candidates.append(en.capitalize())

        for word in candidates:
            wl = word.lower()
            exact = [
                p
                for p in all_paths
                if wl
                in [seg.lower() for seg in p.replace("\\", "/").split("/")]
            ]
            if exact:
                print(f"[RETRIEVER] topic scope (exact folder): '{word}'")
                return word

            docx_folder = [
                p
                for p in all_paths
                if p.lower().endswith(".docx")
                and wl in p.replace("\\", "/").rsplit("/", 1)[0].lower()
            ]
            if docx_folder:
                print(f"[RETRIEVER] topic scope (folder prefix): '{word}'")
                return word

        return None

    def _named_entities(self, query: str) -> list[str]:
        entities = []

        for dq, sq in _QUOTED_PATTERN.findall(query):
            entities.append(dq or sq)

        for m in re.finditer(
            r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z0-9][a-zA-Z0-9]*){0,3})\b", query
        ):
            phrase = m.group(1)
            words = phrase.split()
            if len(words) == 1 and words[0] in _QUESTION_WORDS:
                continue
            if all(w in _QUESTION_WORDS for w in words):
                continue
            entities.append(phrase)

        for m in re.finditer(r"\b([A-Z][a-z]{3,})\b", query):
            word = m.group(1)
            if word not in _QUESTION_WORDS and word not in {
                e.split()[0] for e in entities
            }:
                entities.append(word)

        for m in re.finditer(r"\b([А-ЯЁ][а-яё]{3,})\b", query):
            entities.append(m.group(1))

        # Lowercase content words: only keep ones distinctive enough to be a
        # useful keyword anchor. We decide that by HIT COUNT at search time
        # (below), not by maintaining an ever-growing stopword list. Collect
        # candidate words here; the caller filters by how many chunks they hit.
        _COMMON = {
            # question / framing words
            "what",
            "were",
            "where",
            "when",
            "which",
            "whom",
            "whose",
            "this",
            "that",
            "these",
            "those",
            "with",
            "from",
            "about",
            "have",
            "has",
            "had",
            "been",
            "being",
            "will",
            "shall",
            "your",
            "yours",
            "they",
            "them",
            "their",
            "here",
            "there",
            "some",
            "many",
            "much",
            "also",
            "just",
            "only",
            "more",
            "most",
            "main",
            "such",
            "than",
            "then",
            "into",
            "over",
            "under",
            "between",
            "across",
            "through",
            # generic verbs of asking / recalling / writing
            "give",
            "given",
            "tell",
            "find",
            "found",
            "look",
            "show",
            "note",
            "notes",
            "noted",
            "think",
            "thought",
            "feel",
            "felt",
            "remember",
            "recall",
            "reading",
            "read",
            "writing",
            "wrote",
            "write",
            "written",
            "mention",
            "mentions",
            "mentioned",
            "describe",
            "explain",
            "summarize",
            "summarise",
            "list",
            "name",
            "named",
            "called",
            # generic activity verbs (catch the noise)
            "start",
            "started",
            "starting",
            "begin",
            "began",
            "learning",
            "learn",
            "learnt",
            "learned",
            "study",
            "studied",
            "studying",
            "play",
            "playing",
            "played",
            "watch",
            "watching",
            "watched",
            "visit",
            "visited",
            "went",
            "going",
            "make",
            "made",
            "take",
            "took",
            "spend",
            "spent",
            "used",
            "using",
            # vague content placeholders
            "something",
            "anything",
            "everything",
            "nothing",
            "someone",
            "anyone",
            "anything",
            "really",
            "actually",
            "maybe",
            "perhaps",
            "thing",
            "things",
            "stuff",
            "kind",
            "type",
            "sort",
            # time / quantity words
            "during",
            "trip",
            "trips",
            "time",
            "times",
            "year",
            "years",
            "back",
            "ago",
            "after",
            "before",
            "while",
            "since",
            "today",
            "recent",
            "recently",
            "first",
            "last",
            "next",
            "early",
            "late",
            "could",
            "would",
            "should",
            "might",
            "must",
            "book",
            "books",
            "article",
            "articles",
            "page",
            "pages",
            # Russian function / framing words
            "мои",
            "моих",
            "моей",
            "моего",
            "моим",
            "каковы",
            "какой",
            "какие",
            "какую",
            "каком",
            "главные",
            "главных",
            "главное",
            "результатам",
            "результаты",
            "поездки",
            "поездку",
            "поездке",
            "краткую",
            "краткий",
            "справку",
            "справка",
            "занимался",
            "занималась",
            "расскажи",
            "опиши",
            "что",
            "как",
            "где",
            "когда",
            "почему",
            "зачем",
            "это",
            "этот",
            "эта",
            "был",
            "была",
            "было",
            "были",
            "есть",
            "мной",
            "меня",
            "мне",
            "свои",
            "свой",
            "своих",
            "также",
            "очень",
            "более",
            "менее",
            "после",
            "перед",
            "около",
            "между",
        }
        for word in re.findall(r"\b([a-zа-яё]{4,})\b", query):
            if word.lower() not in _COMMON:
                entities.append(word)

        seen = set()
        return [
            e
            for e in entities
            if not (e.lower() in seen or seen.add(e.lower()))
        ]

    def _expand_query(self, query: str) -> list[str]:
        import ollama

        model = os.getenv("OLLAMA_MODEL", "llama3")

        prompt = (
            f"Generate 3 short search queries to find personal notes about:\n"
            f'"{query}"\n\n'
            f"Rules:\n"
            f"- If the query is in English, write 2 English variants and "
            f"1 Russian translation\n"
            f"- If the query is in Russian, write 2 Russian variants and "
            f"1 English translation\n"
            f"- For film/book/show/game titles, include the Russian "
            f"transliteration (e.g. 'Twin Peaks' -> 'Твин Пикс')\n"
            f"- Each query should be 2-6 words, focused on key nouns\n"
            f"- Output ONLY the queries, one per line, no numbering"
        )

        try:
            from brain.search.synthesiser import _model_options

            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    **_model_options(),
                    "temperature": 0.3,
                    "num_predict": 80,
                },
            )
            raw = response["message"]["content"].strip()
            extras = [
                line.strip().strip('"').strip("'")
                for line in raw.split("\n")
                if line.strip() and len(line.strip()) > 3
            ][:3]
            queries = [query] + extras
            print(f"[RETRIEVER] expanded: {extras}")
            return queries
        except Exception as e:
            print(f"[RETRIEVER] expansion failed ({e})")
            return [query]

    def _to_results(self, chunks: list) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id=f"{c.file_path}::{c.chunk_index}",
                text=c.raw_text,
                file_path=c.file_path,
                file_type=c.file_path.rsplit(".", 1)[-1],
                chunk_index=c.chunk_index,
                score=1.0,
            )
            for c in chunks
        ]
