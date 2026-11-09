import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

import brain.config as config

Base = declarative_base()
log = logging.getLogger(__name__)


class FileRecord(Base):
    """Один проиндексированный файл и его контрольная сумма."""

    __tablename__ = "files"

    file_path = Column(String, primary_key=True)
    file_type = Column(String, nullable=False)
    checksum = Column(String, nullable=False)
    chunk_count = Column(Integer, default=0)
    indexed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ChunkRecord(Base):
    """Сырой текст чанка - для keyword-поиска и отладки."""

    __tablename__ = "chunks"

    chunk_id = Column(
        String, primary_key=True
    )
    file_path = Column(String, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    raw_text = Column(Text, nullable=False)
    section = Column(String, nullable=True)  # заголовок секции, если был


class FileTerm(Base):
    """Термин файла для частотной фильтрации: df(term) = число файлов с ним."""

    __tablename__ = "file_terms"

    file_path = Column(String, primary_key=True)
    term = Column(String, primary_key=True)


_WORD_PATTERN = re.compile(r"[a-zа-яё]{3,}")
# Термин считается общим для корпуса, если встречается минимум в 3 файлах
# и в доле файлов не ниже порога
_CORPUS_COMMON_MIN_FILES = 3
_CORPUS_COMMON_RATIO = 0.15


def _terms_of(text: str) -> set[str]:
    return set(_WORD_PATTERN.findall(text.lower()))


class DocumentStore:
    """SQLite-хранилище: чек-суммы файлов, тексты чанков, частоты терминов."""

    def __init__(self, db_path: str | None = None):
        path = db_path or str(config.sqlite_path())
        self._engine = create_engine(f"sqlite:///{path}")
        Base.metadata.create_all(self._engine)
        self._migrate()
        self._Session = sessionmaker(bind=self._engine)

    def _migrate(self) -> None:
        """Добавляет колонки, появившиеся после создания старой БД."""
        from sqlalchemy import text

        with self._engine.begin() as conn:
            columns = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(chunks)"))
            }
            if "section" not in columns:
                conn.execute(
                    text("ALTER TABLE chunks ADD COLUMN section VARCHAR")
                )

    def compute_checksum(self, path: Path) -> str:
        """MD5 содержимого файла - достаточно для выявления изменений."""
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                hasher.update(block)
        return hasher.hexdigest()

    def needs_reindex(self, path: Path) -> bool:
        """True, если файл новый или менялся с прошлой индексации."""
        session = self._Session()
        try:
            record = session.get(FileRecord, str(path))
            if record is None:
                return True
            return self.compute_checksum(path) != record.checksum
        finally:
            session.close()

    def record_file(
        self, path: Path, file_type: str, chunk_count: int
    ) -> None:
        """Upsert записи файла после успешной индексации."""
        session = self._Session()
        try:
            checksum = self.compute_checksum(path)
            record = session.get(FileRecord, str(path))
            if record is None:
                record = FileRecord(file_path=str(path))
                session.add(record)
            record.file_type = file_type
            record.checksum = checksum
            record.chunk_count = chunk_count
            record.indexed_at = datetime.now(timezone.utc)
            session.commit()
        finally:
            session.close()

    def save_chunks(
        self, file_path: str, chunks: list[tuple]
    ) -> None:
        """Сохраняет тексты чанков и пересобирает термины файла."""
        session = self._Session()
        try:
            session.query(ChunkRecord).filter_by(file_path=file_path).delete()
            file_terms: set[str] = set()
            for item in chunks:
                chunk_index, text = item[0], item[1]
                section = item[2] if len(item) > 2 else None
                session.add(
                    ChunkRecord(
                        chunk_id=f"{file_path}::{chunk_index}",
                        file_path=file_path,
                        chunk_index=chunk_index,
                        raw_text=text,
                        section=section,
                    )
                )
                file_terms |= _terms_of(text)
            session.query(FileTerm).filter_by(file_path=file_path).delete()
            session.add_all(
                FileTerm(file_path=file_path, term=t) for t in file_terms
            )
            session.commit()
        finally:
            session.close()

    def find_files_matching(self, substring: str) -> list[str]:
        """Пути файлов, содержащих подстроку (регистронезависимо)."""
        session = self._Session()
        try:
            pattern = f"%{substring.lower()}%"
            rows = (
                session.query(FileRecord.file_path)
                .filter(FileRecord.file_path.ilike(pattern))
                .all()
            )
            return [r[0] for r in rows]
        finally:
            session.close()

    def get_chunks_for_paths(
        self, file_paths: list[str]
    ) -> list[ChunkRecord]:
        """Все чанки указанных файлов, упорядоченные по файлу и индексу."""
        if not file_paths:
            return []
        session = self._Session()
        try:
            chunks = (
                session.query(ChunkRecord)
                .filter(ChunkRecord.file_path.in_(file_paths))
                .order_by(ChunkRecord.file_path, ChunkRecord.chunk_index)
                .all()
            )
            session.expunge_all()
            return chunks
        finally:
            session.close()

    def prune_missing(self) -> list[str]:
        """Удаляет записи файлов, которых больше нет на диске.
        Возвращает удалённые пути.
        """
        session = self._Session()
        try:
            missing = [
                r.file_path
                for r in session.query(FileRecord).all()
                if not Path(r.file_path).exists()
            ]
            for path in missing:
                session.query(ChunkRecord).filter_by(file_path=path).delete()
                session.query(FileTerm).filter_by(file_path=path).delete()
                session.query(FileRecord).filter_by(file_path=path).delete()
            session.commit()
            return missing
        finally:
            session.close()

    def delete_records(self, file_path: str) -> None:
        """Удаляет все записи файла: чанки, термины, метаданные."""
        session = self._Session()
        try:
            session.query(ChunkRecord).filter_by(file_path=file_path).delete()
            session.query(FileTerm).filter_by(file_path=file_path).delete()
            session.query(FileRecord).filter_by(file_path=file_path).delete()
            session.commit()
        finally:
            session.close()

    def is_corpus_common(self, term: str) -> bool:
        """True, если термин встречается в значимой доле корпуса."""
        session = self._Session()
        try:
            df = (
                session.query(FileTerm)
                .filter_by(term=term.lower())
                .count()
            )
            if df < _CORPUS_COMMON_MIN_FILES:
                return False
            total = session.query(FileRecord).count()
            return df >= max(
                _CORPUS_COMMON_MIN_FILES, total * _CORPUS_COMMON_RATIO
            )
        finally:
            session.close()

    def list_file_paths(self) -> list[str]:
        """Все проиндексированные пути - для проверки упоминаний файлов."""
        session = self._Session()
        try:
            return [
                f.file_path for f in session.query(FileRecord.file_path).all()
            ]
        finally:
            session.close()

    def keyword_search_in_files(
        self, keyword: str, file_paths: list[str]
    ) -> list[ChunkRecord]:
        """Поиск подстроки в пределах заданных файлов."""
        if not file_paths:
            return []
        session = self._Session()
        try:
            pattern = f"%{keyword}%"
            results = (
                session.query(ChunkRecord)
                .filter(
                    ChunkRecord.raw_text.ilike(pattern),
                    ChunkRecord.file_path.in_(file_paths),
                )
                .order_by(ChunkRecord.file_path, ChunkRecord.chunk_index)
                .all()
            )
            session.expunge_all()
            return results
        finally:
            session.close()

    def keyword_search(
        self, keyword: str, limit: int = 30
    ) -> list[ChunkRecord]:
        """Поиск подстроки по текстам чанков (SQL LIKE).

        Векторные эмбеддинги плохо ищут точные совпадения (URL, имена,
        коды) - это второй, детерминированный путь поиска.
        """
        session = self._Session()
        try:
            pattern = f"%{keyword}%"
            results = (
                session.query(ChunkRecord)
                .filter(ChunkRecord.raw_text.ilike(pattern))
                .order_by(ChunkRecord.file_path, ChunkRecord.chunk_index)
                .limit(limit)
                .all()
            )
            session.expunge_all()
            return results
        finally:
            session.close()

    def expand_neighbours(
        self, chunks: list[ChunkRecord], window: int = 3
    ) -> list[ChunkRecord]:
        """Добавляет к попаданиям соседние чанки (±window) в том же файле.

        Keyword-попадание часто сидит в одном чанке, а нужный контекст
        (секция заметки) - в соседних. Попадания разбиваются на кластеры
        (разрыв больше 2*window - отдельный кластер), каждый кластер -
        одним range-запросом.
        """
        if not chunks:
            return []

        # file_path -> отсортированные индексы попаданий
        by_file: dict[str, list[int]] = {}
        for c in chunks:
            by_file.setdefault(c.file_path, []).append(c.chunk_index)

        session = self._Session()
        try:
            result_map: dict[tuple, ChunkRecord] = {}
            for file_path, indices in by_file.items():
                indices.sort()
                # кластеры: соседние попадания с маленьким разрывом -
                # одним диапазоном, иначе выдача раздувается
                cluster_start = indices[0]
                prev = indices[0]
                for idx in indices[1:] + [None]:
                    if idx is None or idx - prev > 2 * window:
                        found = (
                            session.query(ChunkRecord)
                            .filter(
                                ChunkRecord.file_path == file_path,
                                ChunkRecord.chunk_index
                                >= cluster_start - window,
                                ChunkRecord.chunk_index <= prev + window,
                            )
                            .all()
                        )
                        for c in found:
                            result_map[(c.file_path, c.chunk_index)] = c
                        if idx is not None:
                            cluster_start = idx
                    if idx is not None:
                        prev = idx

            session.expunge_all()
            hit_keys = {(h.file_path, h.chunk_index) for h in chunks}
            neighbours = sorted(
                (c for k, c in result_map.items() if k not in hit_keys),
                key=lambda c: (c.file_path, c.chunk_index),
            )
            return list(chunks) + neighbours
        finally:
            session.close()

    def expand_dense_clusters(
        self, chunks: list[ChunkRecord], window: int = 2
    ) -> dict:
        """Расширяет соседями только ПЛОТНЫЕ кластеры попаданий
        (несколько попаданий рядом) - вокруг одиночного упоминания
        контекст не тянем: это почти всегда соседние темы.

        Возвращает dict: (file_path, chunk_index) -> чанк.
        """
        if not chunks:
            return {}

        by_file: dict[str, list[int]] = {}
        for c in chunks:
            by_file.setdefault(c.file_path, []).append(c.chunk_index)

        session = self._Session()
        try:
            result_map: dict[tuple, ChunkRecord] = {}
            for file_path, indices in by_file.items():
                indices.sort()
                # кластеры с разрывом <= 2*window; плотные (>= 2 попаданий)
                # расширяем, одиночные пропускаем
                cluster = [indices[0]]
                clusters = []
                for idx in indices[1:]:
                    if idx - cluster[-1] > 2 * window:
                        clusters.append(cluster)
                        cluster = []
                    cluster.append(idx)
                clusters.append(cluster)

                ranges = {
                    i
                    for cl in clusters
                    if len(cl) >= 2
                    for i in range(cl[0] - window, cl[-1] + window + 1)
                }
                if not ranges:
                    continue
                found = (
                    session.query(ChunkRecord)
                    .filter(
                        ChunkRecord.file_path == file_path,
                        ChunkRecord.chunk_index.in_(sorted(ranges)),
                    )
                    .all()
                )
                for c in found:
                    result_map[(c.file_path, c.chunk_index)] = c
            session.expunge_all()
            return result_map
        finally:
            session.close()

    def stats(self) -> dict:
        """Сводная статистика для сайдбара и CLI."""
        session = self._Session()
        try:
            files = session.query(FileRecord).all()
            return {
                "total_files": len(files),
                "by_type": self._count_by_type(files),
                "total_chunks": sum(f.chunk_count for f in files),
            }
        finally:
            session.close()

    def _count_by_type(self, files: list[FileRecord]) -> dict:
        counts: dict[str, int] = {}
        for f in files:
            counts[f.file_type] = counts.get(f.file_type, 0) + 1
        return counts
