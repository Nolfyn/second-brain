"""Retriever: точка входа поиска по документам.

Приоритет стратегий:
1. Точные якоря (URL, домены, фразы в кавычках, сайты) -> keyword
2. Имя файла -> выдача всего файла
3. Год в запросе -> чанки файлов с годом в пути
4. Тематическая папка (запросы-размышления) -> чанки .docx папки
5. Иначе -> семантика с LLM-расширением запроса + keyword-буст сущностей
"""

import logging
import math
import re

import brain.config as config
from brain.index import Embedder, VectorStore, DocumentStore, SearchResult
from brain.search import signals

log = logging.getLogger(__name__)

# Минимальная косинусная близость расширения к оригиналу: ниже -
# мусор от LLM, отсекаем
_MIN_EXPANSION_SIM = 0.4

_REVIEW_WORDS = {
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

# Стемы секций итогов: такие чанки поднимаем вперёд в тематическом поиске
_CONCLUSION_STEMS = ("вывод", "итог", "заключени", "резюм")
# Сколько соседних чанков тянем вокруг keyword-попадания: упоминание
# часто сидит в одном чанке, а нужная секция - рядом
_NEIGHBOUR_WINDOW = 3


class Retriever:
    MAX_RESULTS = 20
    MAX_FILE = 60
    MAX_YEAR = 60
    MAX_KEYWORD = 30
    SMALL_FILE = 60
    MIN_SCORE = 0.35
    SEMANTIC_TOPK = 15
    RERANK_THRESHOLD = 10
    RERANK_FLOOR = 0.02
    RERANK_MIN_KEEP = 8

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        document_store: DocumentStore | None = None,
        reranker=None,
    ):
        self._emb = embedder or Embedder()
        self._vs = vector_store or VectorStore()
        self._ds = document_store or DocumentStore()
        self._reranker = reranker
        self._ignore_patterns = self._load_ignore_patterns()

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[SearchResult]:
        kws = signals.extract_keywords(query)
        indexed = self._ds.list_file_paths()
        named_file = signals.find_named_file(query, indexed)
        year = signals.extract_year(query)
        topic = signals.find_topic_scope(query, indexed)

        log.info(
            "signals: kws=%s file=%r year=%r topic=%r",
            kws,
            named_file,
            year,
            topic,
        )

        if kws:
            results = self._keyword_search(kws, scope_file=named_file)
            if results:
                # Порядок прогонов намеренный (связное чтение секций) -
                # реранк его размазал бы
                return self._finalize(
                    query,
                    self._mark(results, "keyword"),
                    max_results=self.MAX_KEYWORD,
                    rerank=False,
                )

        if named_file:
            return self._finalize(
                query,
                self._file_search(query, named_file),
                keep_ignored=True,
                max_results=self.MAX_FILE,
            )

        if year:
            results = self._year_search(query, year)
            return self._finalize(
                query, results, max_results=self.MAX_YEAR
            )

        if topic:
            results = self._topic_search(query, topic)
            return self._finalize(
                query,
                results,
                max_results=self.MAX_FILE,
                rerank=False,
            )

        return self._finalize(
            query, *self._semantic_with_entities_ex(query)
        )

    # ------------------------------------------------------------------
    # Стратегии
    # ------------------------------------------------------------------

    @staticmethod
    def _order_in_runs(hits: list, expanded: list) -> list:
        """Упорядочивает расширенные попадания непрерывными прогонами.

        Прогон - соседние чанки одного файла; прогоны идут в порядке первого
        попадания, внутри прогона - по chunk_index. Выдача читается связным
        текстом: заголовок секции оказывается сразу перед её содержимым,
        а не размазан по списку excerpts.
        """
        from collections import defaultdict

        hit_keys = {(h.file_path, h.chunk_index) for h in hits}
        hit_rank = {
            k: i for i, k in enumerate(sorted(hit_keys))
        }

        by_file: dict[str, list] = defaultdict(list)
        for c in expanded:
            by_file[c.file_path].append(c)

        runs = []
        for path, cs in by_file.items():
            cs.sort(key=lambda c: c.chunk_index)
            run = [cs[0]]
            run_rank = hit_rank.get((path, cs[0].chunk_index))
            for c in cs[1:]:
                if c.chunk_index > run[-1].chunk_index + 1:
                    runs.append((run_rank, run))
                    run = []
                    run_rank = None
                run.append(c)
                r = hit_rank.get((path, c.chunk_index))
                if run_rank is None or (r is not None and r < run_rank):
                    run_rank = r
            runs.append((run_rank, run))

        runs.sort(key=lambda item: item[0] if item[0] is not None else 10**9)
        return [c for _, run in runs for c in run]

    def _keyword_search(
        self, kws: list[str], scope_file: str | None
    ) -> list:
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

        unique = self._dedup_chunks(chunks)
        expanded = self._ds.expand_neighbours(unique, window=_NEIGHBOUR_WINDOW)
        return self._order_in_runs(unique, expanded)

    def _file_search(self, query: str, named_file: str) -> list[SearchResult]:
        paths = self._ds.find_files_matching(named_file)
        if not paths:
            return []
        chunks = self._ds.get_chunks_for_paths(paths)
        log.info("file %r -> %d chunks", named_file, len(chunks))
        if len(chunks) <= self.SMALL_FILE:
            return self._to_results(chunks, "file")
        # Большой файл: ранжируем чанки семантикой внутри файла
        vec = self._emb.embed_one(query)
        pool = self._vs.search(vec, top_k=300)
        return [
            r for r in pool if named_file.lower() in r.file_path.lower()
        ]

    def _year_search(self, query: str, year: str) -> list[SearchResult]:
        year_paths = [p for p in self._ds.list_file_paths() if year in p]
        log.info("year %s -> %d files", year, len(year_paths))
        if not year_paths:
            return self._semantic_with_entities(query)

        is_review = any(t in query.lower() for t in _REVIEW_WORDS)
        vec = self._emb.embed_one(query)
        pool = self._vs.search(vec, top_k=300)
        in_year = [
            self._with_source(r, "year")
            for r in pool
            if year in r.file_path
        ]

        if is_review or not in_year:
            chunks = self._ds.get_chunks_for_paths(sorted(year_paths))
            return self._to_results(chunks[: self.MAX_YEAR], "year")

        return in_year[: self.MAX_YEAR]

    def _topic_search(self, query: str, topic: str) -> list[SearchResult]:
        topic_paths = [
            p
            for p in self._ds.list_file_paths()
            if topic.lower() in p.lower()
        ]
        log.info("topic %r -> %d files", topic, len(topic_paths))
        if not topic_paths:
            return self._semantic_with_entities(query)

        docx_paths = [p for p in topic_paths if p.lower().endswith(".docx")]

        # Небольшой набор: отдаем все чанки
        all_chunks = self._ds.get_chunks_for_paths(sorted(docx_paths))
        if 0 < len(all_chunks) <= self.MAX_FILE:
            return self._to_results(all_chunks, "topic")

        vec = self._emb.embed_one(query)
        pool = self._vs.search(vec, top_k=300)
        docx_results = [
            self._with_source(r, "topic")
            for r in pool
            if topic.lower() in r.file_path.lower()
            and r.file_path.lower().endswith(".docx")
        ]

        if docx_results:
            # Секции итогов поднимаем вперед - для запросов типа 'сделай выводы'
            target_files = list({r.file_path for r in docx_results})
            seen = {(r.file_path, r.chunk_index) for r in docx_results}
            boosted = []
            for stem in _CONCLUSION_STEMS:
                for c in self._ds.keyword_search_in_files(stem, target_files):
                    key = (c.file_path, c.chunk_index)
                    if key not in seen:
                        seen.add(key)
                        boosted.append(self._to_result(c, "topic"))
            return boosted + docx_results

        chunks = self._ds.get_chunks_for_paths(sorted(docx_paths))
        return self._to_results(chunks[: self.MAX_FILE], "topic")

    @staticmethod
    def _cluster_first_order(hits: list, expanded_map: dict) -> list:
        """Плотные кластеры попаданий (заметки) - первыми, связным
        прогоном по chunk_index; одиночные упоминания - после.

        Плотность = число попаданий в непрерывном прогоне: кластер из
        эссе о Швеции важнее трех разрозненных всплывших упоминаний.
        """
        from collections import defaultdict

        pool = dict(expanded_map)
        for h in hits:
            pool.setdefault((h.file_path, h.chunk_index), h)
        hit_keys = {(h.file_path, h.chunk_index) for h in hits}

        by_file: dict[str, list] = defaultdict(list)
        for c in pool.values():
            by_file[c.file_path].append(c)

        runs = []
        for path, cs in by_file.items():
            cs.sort(key=lambda c: c.chunk_index)
            run = [cs[0]]
            for c in cs[1:]:
                if c.chunk_index > run[-1].chunk_index + 1:
                    runs.append(run)
                    run = []
                run.append(c)
            runs.append(run)

        def run_rank(run):
            n_hits = sum(
                1
                for c in run
                if (c.file_path, c.chunk_index) in hit_keys
            )
            return (-n_hits, run[0].file_path, run[0].chunk_index)

        runs.sort(key=run_rank)
        return [c for run in runs for c in run]

    def _semantic_with_entities(self, query: str) -> list[SearchResult]:
        results, _ = self._semantic_with_entities_ex(query)
        return results

    def _semantic_with_entities_ex(
        self, query: str
    ) -> tuple[list[SearchResult], int]:
        """Семантика + keyword-буст сущностей.

        Возвращает (результаты, число закрепленных): закрепленные
        (keyword-кластеры) сохраняют порядок при финализации - реранкер
        оценивает только семантический хвост.
        """
        queries = self._expand_query(query)
        all_results: dict[tuple, SearchResult] = {}

        for q in queries:
            vec = self._emb.embed_one(q)
            results = self._vs.search(vec, top_k=self.SEMANTIC_TOPK)
            for r in results:
                if r.score is not None and r.score >= self.MIN_SCORE:
                    key = (r.file_path, r.chunk_index)
                    if (
                        key not in all_results
                        or r.score > all_results[key].score
                    ):
                        all_results[key] = r

        sem = sorted(
            all_results.values(),
            key=lambda r: r.score if r.score is not None else 0,
            reverse=True,
        )
        log.info(
            "semantic (%d queries) -> %d results",
            len(queries),
            len(sem),
        )

        entities = signals.extract_entities(query, self._ds.is_corpus_common)
        if entities:
            log.info("entities for keyword boost: %s", entities)
            kw_chunks = []
            for entity in entities:
                hits = self._ds.keyword_search(entity, limit=200)
                # Слишком частотные сущности режут точность - пропускаем
                if len(hits) <= 15:
                    kw_chunks.extend(hits)
                log.info("  entity %r: %d hits", entity, len(hits))

            if kw_chunks:
                unique = self._dedup_chunks(kw_chunks)
                dense = self._ds.expand_dense_clusters(unique, window=2)
                ordered = self._cluster_first_order(unique, dense)
                kw_results = self._mark(ordered, "keyword")
                kw_keys = {(r.file_path, r.chunk_index) for r in kw_results}
                extra_sem = [
                    r
                    for r in sem
                    if (r.file_path, r.chunk_index) not in kw_keys
                ]
                return (kw_results + extra_sem)[: self.MAX_RESULTS], len(
                    kw_results
                )

        return sem[: self.MAX_RESULTS], 0

    def _expand_query(self, query: str) -> list[str]:
        """LLM-расширение запроса: кросс-языковые варианты."""
        from brain.llm import get_llm
        from brain.search import prompts

        try:
            llm = get_llm()
            raw = llm.chat(
                [
                    {
                        "role": "user",
                        "content": prompts.render("expansion", query=query),
                    }
                ],
                temperature=0.3,
                max_tokens=80,
            )
            extras = [
                line.strip().strip('"').strip("'")
                for line in raw.strip().split("\n")
                if line.strip() and len(line.strip()) > 3
            ][:3]
            extras = self._filter_expansions(query, extras)
            log.info("expanded: %s", extras)
            return [query] + extras
        except Exception as e:
            log.warning("expansion failed: %s", e)
            return [query]

    def _filter_expansions(
        self, query: str, extras: list[str]
    ) -> list[str]:
        """Оставляет варианты, близкие к оригиналу по косинусу."""
        if not extras:
            return []
        try:
            vectors = self._emb.embed([query] + extras)
            origin = vectors[0]

            def norm(v):
                return math.sqrt(sum(x * x for x in v)) or 1e-9

            origin_norm = norm(origin)
            kept = []
            for extra, vec in zip(extras, vectors[1:]):
                cos = sum(a * b for a, b in zip(origin, vec)) / (
                    origin_norm * norm(vec)
                )
                log.info("expansion sim %r: %.3f", extra, cos)
                if cos >= _MIN_EXPANSION_SIM:
                    kept.append(extra)
            return kept
        except Exception as e:
            log.warning("expansion filter failed: %s", e)
            return extras

    # ------------------------------------------------------------------
    # Финализация
    # ------------------------------------------------------------------

    def _finalize(
        self,
        query: str,
        results: list[SearchResult],
        pinned: int = 0,
        keep_ignored: bool = False,
        max_results: int | None = None,
        rerank: bool = True,
    ) -> list[SearchResult]:
        """Фильтр .brainignore + реранк + обрезка.

        Первые `pinned` результатов закреплены: реранкер оценивает
        только хвост - иначе он разрывает кластеры и переставляет
        связные прогоны.
        """
        if not keep_ignored:
            results = self._filter_ignored(results)
        cap = max_results or self.MAX_RESULTS
        head, tail = results[:pinned], results[pinned:]
        if rerank and self._reranker and len(tail) > self.RERANK_THRESHOLD:
            try:
                tail = self._reranker.rerank(query, tail, top_n=cap)
                kept = [
                    r for r in tail if (r.score or 0) >= self.RERANK_FLOOR
                ]
                if len(kept) >= self.RERANK_MIN_KEEP:
                    tail = kept
            except Exception as e:
                # Модель недоступна/не скачалась - выдача без реранка
                # лучше упавшего запроса
                log.warning("rerank failed: %s", e)
        return (head + tail)[:cap]

    def _load_ignore_patterns(self) -> list[str]:
        """Паттерны .brainignore: файлы скрыты из общей выдачи,
        но остаются доступны по прямому запросу имени."""
        path = config.brainignore_path()
        if not path.exists():
            return []
        patterns = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line.lower())
        log.info("loaded %d .brainignore patterns", len(patterns))
        return patterns

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

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------

    @staticmethod
    def _with_source(r: SearchResult, source: str) -> SearchResult:
        r.source = source
        return r

    @staticmethod
    def _mark(chunks: list, source: str) -> list[SearchResult]:
        return [Retriever._to_result(c, source) for c in chunks]

    @staticmethod
    def _to_result(chunk, source: str) -> SearchResult:
        return SearchResult(
            chunk_id=f"{chunk.file_path}::{chunk.chunk_index}",
            text=chunk.raw_text,
            file_path=chunk.file_path,
            file_type=chunk.file_path.rsplit(".", 1)[-1],
            chunk_index=chunk.chunk_index,
            score=None,
            source=source,
            section=getattr(chunk, "section", None),
        )

    @staticmethod
    def _to_results(chunks: list, source: str) -> list[SearchResult]:
        return [Retriever._to_result(c, source) for c in chunks]

    @staticmethod
    def _dedup_chunks(chunks: list) -> list:
        seen, unique = set(), []
        for c in chunks:
            k = (c.file_path, c.chunk_index)
            if k not in seen:
                seen.add(k)
                unique.append(c)
        unique.sort(key=lambda c: (c.file_path, c.chunk_index))
        return unique
