import time
from threading import Lock, Thread

from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn


class ProgressLogger:
    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed:3.0f}/{task.total:3.0f}"),
            "[progress.percentage]{task.percentage:>3.0f}%",
        )
        self.task = self.progress.add_task("[green]Processing...", total=100)
        self.live = Live(self.progress, console=self.console, refresh_per_second=10)
        self.lock = Lock()

    def start(self):
        self.live.start()
        self._progress_thread = Thread(target=self._dummy_progress_update)
        self._progress_thread.start()

    def log_message(self, text: str):
        with self.lock:
            self.console.log(text)

    def update_progress(self, percentage: float):
        self.progress.update(self.task, advance=percentage)

    def _dummy_progress_update(self):
        # Dummy function to represent background progress updates.
        # Replace or remove this with actual work and progress updates.
        while not self.progress.finished:
            time.sleep(1)

    def stop(self):
        with self.lock:
            self.progress.update(self.task, completed=100)
            self.live.stop()

        if self._progress_thread.is_alive():
            self._progress_thread.join()
