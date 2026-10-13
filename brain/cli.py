from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

app = typer.Typer(
    help="Second Brain — поиск по личным документам с помощью ИИ."
)
console = Console()


@app.command()
def index(
    folder: str = typer.Argument(..., help="Папка для индексации (рекурсивно)."),
):
    """Индексирует все поддерживаемые документы в папке."""
    from brain.index import Indexer

    path = Path(folder).expanduser().resolve()
    if not path.exists():
        console.print(f"[red]Папка не найдена: {path}[/red]")
        raise typer.Exit(1)

    indexer = Indexer()
    indexer.index_folder(path)


@app.command()
def stats():
    """Сводная статистика текущего индекса."""
    from brain.index import DocumentStore

    store = DocumentStore()
    s = store.stats()

    console.print(f"[bold]Всего файлов:[/bold] {s['total_files']}")
    console.print(f"[bold]Всего чанков:[/bold] {s['total_chunks']}")
    console.print("[bold]По типам:[/bold]")
    for file_type, count in s["by_type"].items():
        console.print(f"  {file_type}: {count}")


@app.command()
def check(
    file: str = typer.Argument(
        ..., help="Имя файла или подстрока для поиска в индексе."
    ),
):
    """Проверяет, проиндексирован ли файл, и показывает превью первого чанка."""
    from brain.index import DocumentStore
    from brain.index.document_store import FileRecord, ChunkRecord

    store = DocumentStore()
    session = store._Session()
    try:
        all_records = session.query(FileRecord).all()
        matches = [
            r for r in all_records if file.lower() in r.file_path.lower()
        ]

        if not matches:
            console.print(
                f"[red]Не проиндексирован:[/red] файлы по запросу '{file}' не найдены."
            )
            console.print(
                "Проверьте написание или запустите 'brain index' для папки с файлом."
            )
            raise typer.Exit(1)

        for record in matches:
            console.print(f"[green]Проиндексирован[/green]: {record.file_path}")
            console.print(f"  Тип:          {record.file_type}")
            console.print(f"  Чанков:       {record.chunk_count}")
            console.print(f"  Дата индекса: {record.indexed_at}")

            chunks = (
                session.query(ChunkRecord)
                .filter_by(file_path=record.file_path)
                .order_by(ChunkRecord.chunk_index)
                .all()
            )
            if chunks:
                preview = chunks[0].raw_text[:300]
                console.print(f"\n  [bold]Превью первого чанка:[/bold]")
                console.print(f"  {preview}...\n")
            else:
                console.print(
                    "  [red]Чанки отсутствуют — запись о файле есть, а векторов может не быть. "
                    'Запустите: brain reindex "<полный путь>"[/red]\n'
                )
    finally:
        session.close()


@app.command()
def reindex(
    path: str = typer.Argument(
        ..., help="Файл или папка для принудительной переиндексации."
    ),
):
    """Переиндексирует файл или папку, игнорируя контрольные суммы."""
    from brain.index import Indexer, DocumentStore
    from brain.index.document_store import FileRecord, ChunkRecord

    target = Path(path).expanduser().resolve()
    store = DocumentStore()
    session = store._Session()

    try:
        if target.is_file():
            paths_to_clear = [str(target)]
        else:
            paths_to_clear = [
                r.file_path
                for r in session.query(FileRecord).all()
                if r.file_path.startswith(str(target))
            ]

        if not paths_to_clear:
            console.print(
                f"[yellow]В индексе нет записей под {target}[/yellow]"
            )
            raise typer.Exit(0)

        console.print(
            f"Очистка {len(paths_to_clear)} устаревшей записи (записей) в SQLite..."
        )
        for p in paths_to_clear:
            session.query(ChunkRecord).filter_by(file_path=p).delete()
            session.query(FileRecord).filter_by(file_path=p).delete()
        session.commit()
        console.print("[green]Очищено.[/green]")
    finally:
        session.close()

    indexer = Indexer()
    if target.is_file():
        try:
            indexer.index_file(target)
            console.print(f"[green]Переиндексирован:[/green] {target.name}")
        except Exception as e:
            console.print(f"[red]Ошибка:[/red] {e}")
    else:
        indexer.index_folder(target)


@app.command()
def watch(
    folder: str = typer.Argument(
        ..., help="Папка для наблюдения за изменениями (рекурсивно)."
    ),
):
    """Следит за папкой и автоматически переиндексирует изменённые файлы."""
    from brain.index import Watcher, DocumentStore

    path = Path(folder).expanduser().resolve()
    if not path.exists():
        console.print(f"[red]Папка не найдена: {path}[/red]")
        raise typer.Exit(1)

    store = DocumentStore()
    s = store.stats()
    console.print(
        f"[dim]Уже в индексе: {s['total_files']} файлов, {s['total_chunks']} чанков.[/dim]"
    )

    watcher = Watcher()
    watcher.watch(path)


@app.command()
def grep(
    keyword: str = typer.Argument(
        ..., help="Подстрока для поиска по тексту проиндексированных чанков."
    ),
):
    """Поиск подстроки по индексу без LLM — для отладки retrieval."""
    from brain.index import DocumentStore

    store = DocumentStore()
    chunks = store.keyword_search(keyword, limit=20)

    if not chunks:
        console.print(f"[red]Ни один чанк не содержит '{keyword}'.[/red]")
        console.print(
            "Текста нет в проиндексированных файлах, либо извлечение изменило его (ссылки, кодировка)."
        )
        raise typer.Exit(1)

    console.print(
        f"[green]Найдено {len(chunks)} чанков с '{keyword}':[/green]\n"
    )
    for c in chunks:
        console.print(f"[bold]{c.file_path}[/bold] (чанк {c.chunk_index})")
        idx = c.raw_text.lower().find(keyword.lower())
        start = max(0, idx - 60)
        end = min(len(c.raw_text), idx + len(keyword) + 60)
        snippet = c.raw_text[start:end].replace("\n", " ")
        # print(), а не console.print(): Rich на Windows падает на Unicode при пайпе
        print(f"  ...{snippet}...\n")


@app.command()
def search_test(
    query: str = typer.Argument(
        ..., help="Запрос для проверки всего пайплайна поиска."
    ),
):
    """Показывает, что именно вернул ретривер, без вызова LLM."""
    from brain.index import Embedder, VectorStore, DocumentStore
    from brain.search.retriever import Retriever

    embedder = Embedder()
    vector_store = VectorStore()
    document_store = DocumentStore()
    searcher = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        document_store=document_store,
    )

    console.print(f"\n[bold]Запрос:[/bold] {query}\n")
    results = searcher.retrieve(query)

    if not results:
        console.print("[red]Результатов нет.[/red]")
        return

    console.print(f"[green]{len(results)} результатов:[/green]\n")
    for i, r in enumerate(results):
        console.print(
            f"[bold][{i}][/bold] {r.file_path} (чанк {r.chunk_index}, score={r.score})"
        )
        print(f"     {r.text[:150].replace(chr(10), ' ')}...")
        console.print()


if __name__ == "__main__":
    app()
