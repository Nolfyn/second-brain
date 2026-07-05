from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

app = typer.Typer(
    help="Second Brain — your personal AI-powered document search."
)
console = Console()


@app.command()
def index(
    folder: str = typer.Argument(..., help="Folder to index (recursively)."),
):
    """Index all supported documents in a folder."""
    from brain.index import Indexer

    path = Path(folder).expanduser().resolve()
    if not path.exists():
        console.print(f"[red]Folder not found: {path}[/red]")
        raise typer.Exit(1)

    indexer = Indexer()
    indexer.index_folder(path)


@app.command()
def stats():
    """Show summary statistics about the current index."""
    from brain.index import DocumentStore

    store = DocumentStore()
    s = store.stats()

    console.print(f"[bold]Total files indexed:[/bold] {s['total_files']}")
    console.print(f"[bold]Total chunks:[/bold] {s['total_chunks']}")
    console.print("[bold]By type:[/bold]")
    for file_type, count in s["by_type"].items():
        console.print(f"  {file_type}: {count}")


@app.command()
def check(
    file: str = typer.Argument(
        ..., help="Filename or substring to look up in the index."
    ),
):
    """Check whether a file was indexed and show a preview of its first chunk."""
    from brain.index import DocumentStore
    from brain.index.document_store import FileRecord, ChunkRecord

    store = DocumentStore()
    session = store._Session()
    try:
        # Match by substring so "report.docx" finds
        # "/path/to/documents/report.docx"
        all_records = session.query(FileRecord).all()
        matches = [
            r for r in all_records if file.lower() in r.file_path.lower()
        ]

        if not matches:
            console.print(
                f"[red]Not indexed:[/red] no file matching '{file}' found."
            )
            console.print(
                "Check the spelling, or run 'brain index' on its containing folder."
            )
            raise typer.Exit(1)

        for record in matches:
            console.print(f"[green]Indexed[/green]: {record.file_path}")
            console.print(f"  Type:       {record.file_type}")
            console.print(f"  Chunks:     {record.chunk_count}")
            console.print(f"  Indexed at: {record.indexed_at}")

            chunks = (
                session.query(ChunkRecord)
                .filter_by(file_path=record.file_path)
                .order_by(ChunkRecord.chunk_index)
                .all()
            )
            if chunks:
                preview = chunks[0].raw_text[:300]
                console.print(f"\n  [bold]First chunk preview:[/bold]")
                console.print(f"  {preview}...\n")
            else:
                console.print(
                    "  [red]Chunks missing — file record exists but vectors may not. "
                    'Run: brain reindex "<full path>"[/red]\n'
                )
    finally:
        session.close()


@app.command()
def reindex(
    path: str = typer.Argument(
        ..., help="File or folder to force re-index, ignoring checksums."
    ),
):
    """Force re-index a file or folder, ignoring cached checksums."""
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
                f"[yellow]No indexed records found under {target}[/yellow]"
            )
            raise typer.Exit(0)

        console.print(
            f"Clearing {len(paths_to_clear)} stale record(s) from SQLite..."
        )
        for p in paths_to_clear:
            session.query(ChunkRecord).filter_by(file_path=p).delete()
            session.query(FileRecord).filter_by(file_path=p).delete()
        session.commit()
        console.print("[green]Cleared.[/green]")
    finally:
        session.close()

    indexer = Indexer()
    if target.is_file():
        try:
            indexer.index_file(target)
            console.print(f"[green]Re-indexed:[/green] {target.name}")
        except Exception as e:
            console.print(f"[red]Failed:[/red] {e}")
    else:
        indexer.index_folder(target)


@app.command()
def watch(
    folder: str = typer.Argument(
        ..., help="Folder to watch for changes (recursively)."
    ),
):
    """Watch a folder and automatically re-index files as they change."""
    from brain.index import Watcher, DocumentStore

    path = Path(folder).expanduser().resolve()
    if not path.exists():
        console.print(f"[red]Folder not found: {path}[/red]")
        raise typer.Exit(1)

    store = DocumentStore()
    s = store.stats()
    console.print(
        f"[dim]{s['total_files']} files already indexed, {s['total_chunks']} chunks.[/dim]"
    )

    watcher = Watcher()
    watcher.watch(path)


@app.command()
def grep(
    keyword: str = typer.Argument(
        ..., help="Literal substring to search for in indexed chunk text."
    ),
):
    """Run a raw keyword search against the index (no LLM involved) -- useful for debugging retrieval."""
    from brain.index import DocumentStore

    store = DocumentStore()
    chunks = store.keyword_search(keyword, limit=20)

    if not chunks:
        console.print(f"[red]No chunks contain '{keyword}'.[/red]")
        console.print(
            "This means either the text isn't in any indexed file, or extraction altered it (e.g. hyperlinks, encoding)."
        )
        raise typer.Exit(1)

    console.print(
        f"[green]Found {len(chunks)} chunk(s) containing '{keyword}':[/green]\n"
    )
    for c in chunks:
        console.print(f"[bold]{c.file_path}[/bold] (chunk {c.chunk_index})")
        idx = c.raw_text.lower().find(keyword.lower())
        start = max(0, idx - 60)
        end = min(len(c.raw_text), idx + len(keyword) + 60)
        snippet = c.raw_text[start:end].replace("\n", " ")
        # Use print() not console.print() for raw text content --
        # Rich's legacy Windows renderer crashes on Unicode chars
        # outside cp1251 when piped through | head or similar.
        print(f"  ...{snippet}...\n")


@app.command()
def ask_agent(
    query: str = typer.Argument(
        ..., help="Question to ask the agentic retrieval loop."
    ),
):
    """Ask a question using the agentic tool-calling loop (prints tool calls + final answer)."""
    from brain.search import Agent

    agent = Agent()
    console.print(f"[dim]Query: {query}[/dim]\n")

    answer_parts = []
    for event in agent.run(query):
        if event["type"] == "tool_call":
            console.print(
                f"[bold cyan]→ calling {event['tool']}[/bold cyan]({event['arguments']})"
            )
        elif event["type"] == "tool_result":
            count = event["result"].get("result_count", "?")
            console.print(f"  [dim]got {count} result(s)[/dim]")
        elif event["type"] == "token":
            answer_parts.append(event["content"])

    console.print(
        f"\n[bold green]Answer:[/bold green]\n{''.join(answer_parts)}"
    )


@app.command()
def search_test(
    query: str = typer.Argument(
        ..., help="Query to test the full search pipeline."
    ),
):
    """Test the full search pipeline without the LLM -- shows exactly what retriever returns."""
    from dotenv import load_dotenv

    load_dotenv()
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

    console.print(f"\n[bold]Query:[/bold] {query}\n")
    results = searcher.retrieve(query)

    if not results:
        console.print("[red]No results returned.[/red]")
        return

    console.print(f"[green]{len(results)} result(s):[/green]\n")
    for i, r in enumerate(results):
        console.print(
            f"[bold][{i}][/bold] {r.file_path} (chunk {r.chunk_index}, score={r.score})"
        )
        print(f"     {r.text[:150].replace(chr(10), ' ')}...")
        console.print()


if __name__ == "__main__":
    app()
