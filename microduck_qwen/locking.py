"""Nonblocking controller lock; closing the file releases it."""

import errno
import os


def lock_exclusive(file):
    if os.name == "nt":
        import msvcrt

        file.seek(0)
        try:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EDEADLK):
                raise BlockingIOError(error.errno, str(error)) from error
            raise
    else:
        import fcntl

        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
