import threading
import time
from pathlib import Path

from rich.console import Console
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from brain.parsers import SUPPORTED_EXTENSIONS
from .indexer import Indexer

console = Console()

# Пауза после последнего изменения перед индексацией: защита от
# индексации недописанного файла (Word и т.п. пишут не атомарно).
_DEBOUNCE_SECONDS = 2.0


class _DebouncedHandler(FileSystemEventHandler):
    """
    Реагирует на create/modify/move/delete файлов с поддерживаемыми расширениями.
    Серия событий подряд по одному файлу схлопывается в одну индексацию —
    ждём тихого периода _DEBOUNCE_SECONDS.
    """

    def __init__(self, indexer: Indexer):
        self._indexer = indexer
        self._pending_timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def on_created(self, event):
        self._handle_event(event)

    def on_modified(self, event):
        self._handle_event(event)

    def on_moved(self, event):
        # Старое имя убираем из индекса, новое планируем на индексацию
        self._remove(Path(event.src_path))
        if event.is_directory:
            return
        dest = Path(event.dest_path)
        if dest.suffix.lower() in SUPPORTED_EXTENSIONS:
            self._schedule_debounced_index(dest)

    def on_deleted(self, event):
        self._remove(Path(event.src_path))

    def _remove(self, path: Path) -> None:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        console.print(f"[dim]Удалён:[/dim] {path.name}", end=" ")
        try:
            self._indexer.remove_file(path)
            console.print("[green]убран из индекса[/green]")
        except Exception as e:
            console.print(f"[red]failed: {e}[/red]")

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
            return  # файл успели удалить/переместить до срабатывания таймера

        console.print(f"[dim]Изменение:[/dim] {path.name}", end=" ")
        try:
            self._indexer.index_file(path)
            console.print("[green]переиндексирован[/green]")
        except Exception as e:
            console.print(f"[red]failed: {e}[/red]")


class Watcher:
    """Рекурсивно следит за папкой и держит индекс синхронизированным."""

    def __init__(self, indexer: Indexer | None = None):
        self._indexer = indexer or Indexer()
        self._observer = Observer()

    def watch(self, folder: Path) -> None:
        """Блокирующий цикл наблюдения до Ctrl+C."""
        handler = _DebouncedHandler(self._indexer)
        self._observer.schedule(handler, str(folder), recursive=True)
        self._observer.start()

        console.print(
            f"[bold]Наблюдение[/bold] за {folder}... (Ctrl+C — остановить)"
        )

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self._observer.stop()
            console.print("\n[dim]Наблюдение остановлено.[/dim]")
        self._observer.join()
