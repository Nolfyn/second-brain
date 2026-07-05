import threading
import time
from pathlib import Path

from rich.console import Console
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from brain.parsers import SUPPORTED_EXTENSIONS
from .indexer import Indexer

console = Console()

# Wait this long after the last change to a file before indexing it.
# Prevents triggering mid-save (e.g. while Word is still writing the file).
_DEBOUNCE_SECONDS = 2.0


class _DebouncedHandler(FileSystemEventHandler):
    """
    Reacts to file create/modify events for supported extensions.
    Debounces rapid repeated events on the same file (common during
    saves) by waiting for a quiet period before actually indexing.
    """

    def __init__(self, indexer: Indexer):
        self._indexer = indexer
        self._pending_timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def on_created(self, event):
        self._handle_event(event)

    def on_modified(self, event):
        self._handle_event(event)

    def _handle_event(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        self._schedule_debounced_index(path)

    def _schedule_debounced_index(self, path: Path) -> None:
        """Reset the debounce timer each time this file changes again."""
        key = str(path)
        with self._lock:
            existing_timer = self._pending_timers.get(key)
            if existing_timer:
                existing_timer.cancel()

            timer = threading.Timer(
                _DEBOUNCE_SECONDS, self._index_now, args=[path]
            )
            timer.daemon = True
            self._pending_timers[key] = timer
            timer.start()

    def _index_now(self, path: Path) -> None:
        with self._lock:
            self._pending_timers.pop(str(path), None)

        if not path.exists():
            return  # file was deleted/moved before the debounce fired

        console.print(f"[dim]Change detected:[/dim] {path.name}", end=" ")
        try:
            self._indexer.index_file(path)
            console.print("[green]re-indexed[/green]")
        except Exception as e:
            console.print(f"[red]failed: {e}[/red]")


class Watcher:
    """Watches a folder recursively and keeps the index in sync automatically."""

    def __init__(self, indexer: Indexer | None = None):
        self._indexer = indexer or Indexer()
        self._observer = Observer()

    def watch(self, folder: Path) -> None:
        """Block and watch `folder` for changes until interrupted (Ctrl+C)."""
        handler = _DebouncedHandler(self._indexer)
        self._observer.schedule(handler, str(folder), recursive=True)
        self._observer.start()

        console.print(
            f"[bold]Watching[/bold] {folder} for changes... (Ctrl+C to stop)"
        )

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self._observer.stop()
            console.print("\n[dim]Watcher stopped.[/dim]")
        self._observer.join()
