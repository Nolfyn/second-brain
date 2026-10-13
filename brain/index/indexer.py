from pathlib import Path

from rich.console import Console

from brain.parsers import get_parser, SUPPORTED_EXTENSIONS
from .chunker import chunk_text
from .embedder import Embedder
from .vector_store import VectorStore
from .document_store import DocumentStore

console = Console()


class Indexer:
    """
    Orchestrates the full pipeline:
    walk folder -> parse -> chunk -> embed -> store (vector + sqlite)

    Skips files whose checksum hasn't changed since the last run.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        document_store: DocumentStore | None = None,
    ):
        self._embedder = embedder or Embedder()
        self._vector_store = vector_store or VectorStore()
        self._document_store = document_store or DocumentStore()

    def index_folder(self, folder: Path) -> dict:
        """
        Index every supported file in folder (recursively).
        Returns a summary dict: {"indexed": n, "skipped": n, "failed": n}
        """
        files = self._discover_files(folder)
        console.print(
            f"Found [bold]{len(files)}[/bold] supported files in {folder}"
        )

        summary = {"indexed": 0, "skipped": 0, "failed": 0}

        for i, path in enumerate(files, 1):
            console.print(f"[{i}/{len(files)}] {path.name}", end=" ")

            if not self._document_store.needs_reindex(path):
                console.print("[dim]unchanged, skipping[/dim]")
                summary["skipped"] += 1
                continue

            try:
                self.index_file(path)
                console.print("[green]indexed[/green]")
                summary["indexed"] += 1
            except Exception as e:
                console.print(f"[red]failed: {e}[/red]")
                summary["failed"] += 1

        console.print()
        console.print(
            f"[bold green]Done.[/bold green] "
            f"Indexed: {summary['indexed']}, "
            f"Skipped (unchanged): {summary['skipped']}, "
            f"Failed: {summary['failed']}"
        )
        return summary

    def index_file(self, path: Path) -> None:
        """Index a single file: parse, chunk, embed, store."""
        parser = get_parser(path)
        parsed_doc = parser.parse(path)

        chunks = chunk_text(parsed_doc.raw_text)
        if not chunks:
            raise ValueError("No text content extracted")

        texts = [c.text for c in chunks]
        vectors = self._embedder.embed(texts)

        chunk_ids = [f"{path}::{c.chunk_index}" for c in chunks]
        metadatas = [
            {
                "file_path": str(path),
                "file_type": parsed_doc.file_type,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        # Remove old chunks for this file first, then add fresh ones.
        # Important for files that changed and now have a different chunk count.
        self._vector_store.delete_by_file(str(path))
        self._vector_store.add(chunk_ids, texts, vectors, metadatas)

        self._document_store.save_chunks(
            str(path),
            [(c.chunk_index, c.text) for c in chunks],
        )
        self._document_store.record_file(
            path, parsed_doc.file_type, len(chunks)
        )

    def _discover_files(self, folder: Path) -> list[Path]:
        """Recursively find all files with supported extensions."""
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(folder.rglob(f"*{ext}"))
        return sorted(files)
