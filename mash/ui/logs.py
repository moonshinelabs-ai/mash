import contextlib
import sys

from loguru import logger as logging


class RedirectedStdout:
    def write(self, s):
        logging.info(s)

    def flush(self):
        pass


@contextlib.contextmanager
def redirected_stdout_to_loguru():
    stdout = sys.stdout
    stderr = sys.stderr
    try:
        sys.stdout = RedirectedStdout()
        sys.stderr = RedirectedStdout()
        yield
    finally:
        sys.stdout = stdout
        sys.stderr = stderr
