"""New CLI commands for Revise_AI features.

Commands added:
- hybrid-eval: Compare vector vs hybrid search
- detect-new: Check for new lectures
- auto-index: Automatically index new lectures
- show-where: Show timestamps for a topic
"""

import typer

app_features = typer.Typer(help="Revise_AI features")


# ------------------------------------------------------------------
# Hybrid Search Evaluation
# ------------------------------------------------------------------
@app_features.command("hybrid-eval")
def hybrid_eval(
    path: str = typer.Option(..., "--path", help="Golden set JSON"),
    k: int = typer.Option(5, "--k", help="Retrieval depth"),
):
    """Compare vector vs hybrid search on the golden set."""
    from rich.console import Console
    from ytrag.evaluate import compare_retrievers

    console = Console()
    console.print("[bold]🔄 Evaluating retrieval methods...[/bold]\n")

    results = compare_retrievers(path, k)

    console.print(f"[bold]Results:[/bold]\n")
    for method, metrics in results["methods"].items():
        console.print(
            f"  [cyan]{method:8}[/cyan] hit_rate={metrics['hit_rate']:.1%}  "
            f"retrieval={metrics.get('retrieval_hit_rate', 0):.1%}  "
            f"refusal={metrics.get('refusal_rate', 0):.1%}"
        )



# ------------------------------------------------------------------
# New Lecture Detection
# ------------------------------------------------------------------
@app_features.command("detect-new")
def detect_new(
    playlist: str = typer.Option(..., "--playlist", "-p", help="Playlist URL"),
):
    """Check for new lectures in a playlist."""
    from rich.console import Console
    from ytrag.detection import detect_new_lectures

    console = Console()

    console.print("[bold]🔍 Checking for new lectures...[/bold]\n")
    result = detect_new_lectures(playlist)

    console.print(
        f"[bold]Playlist:[/bold] {result.total_in_playlist} total\n"
        f"[bold]Indexed:[/bold] {result.indexed_count}\n"
        f"[bold green]New:[/bold green] {result.new_count}\n"
    )

    if result.new_videos:
        console.print("[bold]New lectures found:[/bold]")
        for video in result.new_videos:
            duration_min = video["duration"] / 60
            console.print(f"  • {video['title']} ({duration_min:.0f}m)")
        console.print("\n[dim]Run [bold]ytrag auto-index --playlist <URL>[/bold] to index them[/dim]")
    else:
        console.print("[green]No new lectures[/green]")


# ------------------------------------------------------------------
# Auto-Index New Lectures
# ------------------------------------------------------------------
@app_features.command("auto-index")
def auto_index(
    playlist: str = typer.Option(..., "--playlist", "-p", help="Playlist URL"),
):
    """Automatically index new lectures from a playlist."""
    from rich.console import Console
    from ytrag.detection import auto_index_new_lectures

    console = Console()

    def on_progress(msg: str):
        console.print(msg)

    console.print("[bold]🚀 Auto-indexing new lectures[/bold]\n")

    results = auto_index_new_lectures(
        playlist,
        on_progress=on_progress,
    )

    console.print(
        f"\n[bold]Summary:[/bold]\n"
        f"  Indexed: {results['new_indexed']}\n"
        f"  Success: {results['success']}\n"
        f"  Failed: {results['failed']}\n"
        f"  Skipped: {results['skipped']}"
    )


# ------------------------------------------------------------------
# Show Me Where
# ------------------------------------------------------------------
@app_features.command("show-where")
def show_where_cmd(
    question: str = typer.Argument(..., help="What to find"),
    k: int = typer.Option(6, "--k", help="Number of results"),
    video_id: str = typer.Option("", "--video-id", help="Search in specific video"),
):
    """Show where in the lectures a topic was explained."""
    from rich.console import Console
    from rich.table import Table
    from ytrag.show_me_where import show_where, show_where_in_lecture

    console = Console()

    if video_id:
        result = show_where_in_lecture(question, video_id, top_k=k)
    else:
        result = show_where(question, top_k=k)

    moments = result.get("moments", [])

    if not moments:
        console.print(f"[yellow]Nothing found for: {question}[/yellow]")
        return

    confidence_text = (
        "[green]Strong match[/green]" if result.get("confident") else "[yellow]Weak match[/yellow]"
    )
    console.print(f"\n{confidence_text} — {len(moments)} moments found:\n")

    for moment in moments:
        console.print(
            f"  [{moment['rank']}] [bold cyan]{moment['lecture']}[/bold cyan] "
            f"[dim]@[/dim] [bold]{moment['timestamp']}[/bold]\n"
            f"       {moment.get('preview', '')[:100]}\n"
        )


if __name__ == "__main__":
    app_features()
