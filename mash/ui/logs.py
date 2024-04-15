import contextlib
import sys

from loguru import logger as logging


class _RedirectedStdout:
    def write(self, s):
        logging.info(s)

    def flush(self):
        pass


@contextlib.contextmanager
def redirected_stdout_to_loguru():
    """Convenience context manager to redirect stdout and stderr to loguru logger.

    Useful for capturing output from external libraries that write to stdout or stderr.
    """
    stdout = sys.stdout
    stderr = sys.stderr
    try:
        sys.stdout = _RedirectedStdout()
        sys.stderr = _RedirectedStdout()
        yield
    finally:
        sys.stdout = stdout
        sys.stderr = stderr
