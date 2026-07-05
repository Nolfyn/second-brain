import hashlib
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class FileRecord(Base):
    """Tracks one indexed file and its checksum, for incremental re-indexing."""

    __tablename__ = "files"

    file_path = Column(String, primary_key=True)
    file_type = Column(String, nullable=False)
    checksum = Column(String, nullable=False)
    chunk_count = Column(Integer, default=0)
    indexed_at = Column(DateTime, default=datetime.utcnow)


class ChunkRecord(Base):
    """Stores the raw text of each chunk, for display/debugging purposes."""

    __tablename__ = "chunks"

    chunk_id = Column(
        String, primary_key=True
    )  # f"{file_path}::{chunk_index}"
    file_path = Column(String, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    raw_text = Column(Text, nullable=False)


class DocumentStore:
    """
    SQLite-backed store for file checksums and chunk text.
    The checksum lets the indexer skip files that haven't changed
    since the last run.
    """

    def __init__(self, db_path: str = "second_brain.db"):
        self._engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

    def compute_checksum(self, path: Path) -> str:
        """MD5 checksum of file contents — fast and sufficient for change detection."""
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                hasher.update(block)
        return hasher.hexdigest()

    def needs_reindex(self, path: Path) -> bool:
        """True if the file is new or has changed since last indexed."""
        session = self._Session()
        try:
            record = session.get(FileRecord, str(path))
            if record is None:
                return True
            current_checksum = self.compute_checksum(path)
            return current_checksum != record.checksum
        finally:
            session.close()

    def record_file(
        self, path: Path, file_type: str, chunk_count: int
    ) -> None:
        """Upsert the file's checksum record after successful indexing."""
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
            record.indexed_at = datetime.utcnow()
            session.commit()
        finally:
            session.close()

    def save_chunks(
        self, file_path: str, chunks: list[tuple[int, str]]
    ) -> None:
        """Save raw chunk text. chunks is a list of (chunk_index, text) tuples."""
        session = self._Session()
        try:
            # Remove old chunks for this file first (in case chunk count changed)
            session.query(ChunkRecord).filter_by(file_path=file_path).delete()
            for chunk_index, text in chunks:
                chunk_id = f"{file_path}::{chunk_index}"
                session.add(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        file_path=file_path,
                        chunk_index=chunk_index,
                        raw_text=text,
                    )
                )
            session.commit()
        finally:
            session.close()

    def expand_with_neighbours(
        self, chunks: list[ChunkRecord], window: int = 1
    ) -> list[ChunkRecord]:
        """
        For each chunk, also fetch up to `window` chunks before and after it
        in the same file. This ensures that content straddling a chunk boundary
        (e.g. a URL at the end of chunk N and its note at the start of chunk N+1)
        is always returned together.

        Returns a flat deduplicated list; does not re-sort (caller sorts).
        """
        if not chunks:
            return []

        session = self._Session()
        try:
            result_map: dict[tuple, ChunkRecord] = {}

            for chunk in chunks:
                for offset in range(-window, window + 1):
                    target_index = chunk.chunk_index + offset
                    if target_index < 0:
                        continue
                    key = (chunk.file_path, target_index)
                    if key in result_map:
                        continue
                    neighbour = (
                        session.query(ChunkRecord)
                        .filter_by(
                            file_path=chunk.file_path, chunk_index=target_index
                        )
                        .first()
                    )
                    if neighbour:
                        result_map[key] = neighbour

            session.expunge_all()
            return list(result_map.values())
        finally:
            session.close()

    def list_file_paths(self) -> list[str]:
        """Return all indexed file paths -- used to verify filename mentions in queries."""
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
        """
        Keyword search limited to specific file paths. More efficient than
        fetching all matches and filtering when the target file set is known.
        """
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
        """
        Literal substring search over raw chunk text (SQL LIKE).
        Vector embeddings are bad at exact-match recall for things like
        URLs, domains, codes, or proper nouns -- this is a deliberate
        second retrieval path used alongside semantic search, not a
        replacement for it.
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

    def get_chunks_for_file_matching(self, filename: str) -> list[ChunkRecord]:
        """
        Return all chunks for the file whose path contains `filename`
        (case-insensitive), ordered by chunk_index. Used when a query
        explicitly names a file -- we want complete coverage, not a
        similarity-ranked sample, since list-style content (e.g. a log
        of entries) can be spread thinly across many chunks.
        """
        session = self._Session()
        try:
            all_chunks = session.query(ChunkRecord).all()
            matching_paths = {
                c.file_path
                for c in all_chunks
                if filename.lower() in c.file_path.lower()
            }
            if not matching_paths:
                return []
            results = (
                session.query(ChunkRecord)
                .filter(ChunkRecord.file_path.in_(matching_paths))
                .order_by(ChunkRecord.file_path, ChunkRecord.chunk_index)
                .all()
            )
            # Detach from session so callers can use these after session closes
            session.expunge_all()
            return results
        finally:
            session.close()

    def stats(self) -> dict:
        """Summary stats for the sidebar / CLI status display."""
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
