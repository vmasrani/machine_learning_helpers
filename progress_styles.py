from typing import Any
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.style import Style
from rich.text import Text

def make_job_description(job_num: int, total_cpus: int, active_cpus: int) -> Text:
    """Create a styled job description with minimal, elegant CPU info."""
    return Text.assemble(
        ("", "dim cyan"),  # Subtle process icon
        (f" Job {job_num:02d}", "bold white"),  # Zero-padded job number
    )

def create_progress_columns(disable_tqdm: bool = False) -> tuple[Progress, Progress]:
    """Create styled progress bars for overall and job-specific progress."""
    # Job-specific progress with minimal styling
    job_progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        SpinnerColumn("dots", style="cyan", speed=1.0),
        BarColumn(
            bar_width=50,
            style="dim cyan",
            complete_style="cyan",
            finished_style="bright_cyan",
        ),
        TextColumn("[cyan]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        disable=disable_tqdm,
        expand=True,
        auto_refresh=False,
    )

    # Overall progress with subtle styling
    overall_progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(
            bar_width=50,
            style="dim blue",
            complete_style="blue",
            finished_style="bright_blue",
        ),
        TextColumn("[blue]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        disable=disable_tqdm,
        expand=True,
    )

    return job_progress, overall_progress

def create_progress_table(
    job_progress: Progress,
    overall_progress: Progress,
    total_cpus: int,
    completed_tasks: int,
    total_tasks: int,
) -> Any:
    """Create a styled table with progress information."""
    from rich.panel import Panel
    from rich.table import Table

    progress_table = Table.grid(expand=False)

    progress_table.add_row(
        Panel(
            overall_progress,
            title="[bold blue]Progress",
            border_style="dim blue",
            padding=(1, 1),  # Increased vertical padding
            title_align="left",
            width=100,
        )
    )

    progress_table.add_row(
        Panel(
            job_progress,
            title=f"[cyan bold]Tasks ({completed_tasks}/{total_tasks}) • {total_cpus} CPUs",
            border_style="dim cyan",
            padding=(1, 1),  # Increased vertical padding
            height=15,       # Increased height
            width=100,
            title_align="left",
        )
    )

    return progress_table

