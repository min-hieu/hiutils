from pathlib import Path
from rich.progress import Progress, MofNCompleteColumn, TimeElapsedColumn

def dirtybar():
    return Progress(
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    return Progress

def wrapbar(title, total, callback):
    with dirtybar() as progress:
        task = progress.add_task(f"[green]{title}", total=total)
        while not progress.finished:
            progress.update(task, completed=callback)
