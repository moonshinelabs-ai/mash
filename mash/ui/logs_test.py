import sys
import unittest
from io import StringIO

from loguru import logger

from mash import ui


class TestRedirectedStdoutToLoguru(unittest.TestCase):
    def setUp(self):
        self.held_output = StringIO()
        logger.remove()
        logger.add(self.held_output, format="{message}")

    def assertLogMessage(self, test_message):
        self.held_output.seek(0)
        self.assertIn(test_message, self.held_output.read())

    def test_redirected_stdout(self):
        test_message = "This is a test message."
        with ui.redirected_stdout_to_loguru():
            print(test_message)
        self.assertLogMessage(test_message)

    def test_redirected_stderr(self):
        test_message = "This is a test error message."
        with ui.redirected_stdout_to_loguru():
            print(test_message, file=sys.stderr)
        self.assertLogMessage(test_message)


class TestMultipleMessages(unittest.TestCase):
    def setUp(self):
        self.held_output = StringIO()
        logger.remove()
        logger.add(self.held_output, format="{message}")

    def assertLogMessages(self, messages):
        self.held_output.seek(0)
        log_content = self.held_output.read()
        for msg in messages:
            self.assertIn(msg, log_content)

    def test_multiple_messages(self):
        messages = ["Message 1", "Message 2", "Message 3"]
        with ui.redirected_stdout_to_loguru():
            for msg in messages:
                print(msg)
        self.assertLogMessages(messages)


if __name__ == "__main__":
    unittest.main()
