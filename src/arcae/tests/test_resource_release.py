import os
import time
from pathlib import Path

import pytest

import arcae

FD_DIR = Path("/proc/self/fd")
TASK_DIR = Path("/proc/self/task")

requires_procfs = pytest.mark.skipif(
    not FD_DIR.exists(), reason="file descriptor inspection requires procfs"
)

# Reads and writes may drop the last reference to a table on an arrow thread,
# in which case the close, or the destruction of the table's I/O pools, is
# completed there. Both happen promptly, but neither is synchronous with the
# Python statement that dropped the reference
SETTLE_TIMEOUT = 30.0
ITERATIONS = 10


def table_fds(table_name):
    """File descriptors held open on the given table"""
    fds = []

    for fd in FD_DIR.iterdir():
        try:
            target = os.readlink(fd)
        except OSError:  # closed while we were looking at it
            continue

        if target.startswith(str(table_name)):
            fds.append(target)

    return fds


def wait_until(predicate, timeout=SETTLE_TIMEOUT):
    deadline = time.monotonic() + timeout

    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)

    return True


@requires_procfs
def test_read_releases_table(tau_ms):
    """Dropping a table that has been read releases its file descriptors.

    The read pipeline holds a reference to the table's IsolatedTableProxy in
    callbacks that are torn down on its own isolation thread once the read
    completes. Closing the table from that thread used to deadlock it, leaving
    the table open for the lifetime of the process"""
    for _ in range(ITERATIONS):
        table = arcae.table(tau_ms)
        table.to_arrow()
        del table

    assert wait_until(
        lambda: not table_fds(tau_ms)
    ), f"{len(table_fds(tau_ms))} file descriptors still open on {tau_ms}"


@requires_procfs
def test_write_releases_table(column_case_table):
    """Dropping a table that has been written to releases its file descriptors.

    The write pipeline holds the same reference as the read pipeline does,
    see :func:`test_read_releases_table`"""
    for _ in range(ITERATIONS):
        table = arcae.table(column_case_table, readonly=False)
        table.putcol("FIXED", table.getcol("FIXED") + 1)
        del table

    assert wait_until(lambda: not table_fds(column_case_table)), (
        f"{len(table_fds(column_case_table))} file descriptors still open "
        f"on {column_case_table}"
    )


@requires_procfs
def test_read_releases_io_threads(tau_ms):
    """Dropping a table that has been read releases its I/O pool threads.

    Each table instance owns a single-threaded arrow I/O pool. A table left
    open by a deadlocked close leaks that thread too"""
    # Read once to start the pools that arrow itself maintains, which are
    # created on demand and shared by every table
    arcae.table(tau_ms).to_arrow()
    threads = len(list(TASK_DIR.iterdir()))

    for _ in range(ITERATIONS):
        table = arcae.table(tau_ms, ninstances=2)
        table.to_arrow()
        del table

    assert wait_until(lambda: len(list(TASK_DIR.iterdir())) <= threads), (
        f"{len(list(TASK_DIR.iterdir())) - threads} threads leaked over "
        f"{ITERATIONS} reads"
    )
