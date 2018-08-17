#!/usr/bin/env python3

import contextlib
import time
import sys
from logging import warning, info, debug, error


class Tee:
    """
    Redirect stdin+stdout to a file and at the same time to the orig stdin+stdout.
    Use as a context manager or with .start() and .stop().
    """

    def __init__(self, name, mode="at"):
        self.filename = name
        self.mode = mode

    def __enter__(self):
        self.start()

    def __exit__(self, *exceptinfo):
        self.stop()

    def start(self):
        self.file = open(self.filename, self.mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def stop(self, *exceptinfo):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


@contextlib.contextmanager
def log_time(prefix=""):
    '''log the time usage in a code block
    prefix: the prefix text to show
    '''
    start = time.time()
    try:
        yield
    finally:
        dt = time.time() - start
        if dt < 300:
            info('{} took {:.2f} s'.format(prefix, dt))
        else:
            info('{} took {:.2f} s ({:.2f} min)'.format(prefix, dt, dt / 60.0))
