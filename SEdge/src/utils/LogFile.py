import sys


class Echo_STDIO_to_File():
    """
    Echo_STDIO_to_File is a class that duplicates (echoes) all output written to sys.stdout to a specified log file.

    Attributes:
        terminal (file-like object): The original sys.stdout stream.
        log (file object): The file where output is also written.
        line_written (bool): Tracks if a line has been written to the log to manage newlines.

    Args:
        file (str): The path to the file where output will be logged.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Methods:
        write(message):
            Writes the message to both sys.stdout and the log file, unless the message contains 'IGNORE: '.
            Handles newlines to avoid extra blank lines in the log file.

        flush():
            Flushes both sys.stdout and the log file.

        close():
            Closes the log file.
    """

    def __init__(self, file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terminal = sys.stdout
        self.log = open(file, "w+")
        self.line_written=False

    def write(self, message):
        self.terminal.write(message)
        if 'IGNORE: ' not in message:
            if len(message.split()) != 0:
                self.log.write(message)
                self.line_written = True
            elif self.line_written:
                self.log.write('\n')
                self.line_written = False
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        pass
    def close(self):
        self.log.close()
        pass