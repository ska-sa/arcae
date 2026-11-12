"""Exercise arcae from many Python threads at once.

Under free-threaded CPython these run truly in parallel. They don't depend on
python-casacore, which isn't available for the free-threaded builds.
"""

import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from arcae.lib.arrow_tables import Configuration, Table
from arcae.testing import DMINFO, NCHAN, NCORR, TABLE_DESC

NTHREADS = 8
NITERATIONS = 50
NROWS = 100


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="Only meaningful on free-threaded CPython",
)
def test_gil_stays_disabled():
    assert not sys._is_gil_enabled()


@pytest.fixture
def sanity_table(tmp_path):
    data = np.arange(NROWS * NCHAN * NCORR).reshape(NROWS, NCHAN, NCORR)
    data = data + data * 1j
    names = np.array([f"name-{i}" for i in range(NROWS)])

    with Table.ms_from_descriptor(
        str(tmp_path / "threads.ms"), "MAIN", table_desc=TABLE_DESC, dminfo=DMINFO
    ) as T:
        T.addrows(NROWS)
        T.putcol("TIME", np.arange(NROWS, dtype=np.float64))
        T.putcol("DATA", data)
        T.putcol("NAMES", names)
        yield T, data, names


def test_concurrent_reads(sanity_table):
    T, data, names = sanity_table
    barrier = Barrier(NTHREADS)

    def read(t):
        barrier.wait()
        for i in range(NITERATIONS):
            start = (t * NITERATIONS + i) % (NROWS - 10)
            rows = np.arange(start, start + 10)
            assert_array_equal(T.getcol("DATA", index=(rows,)), data[rows])
            assert_array_equal(T.getcol("NAMES", index=(rows,)), names[rows])
            assert T.nrow() == NROWS

    with ThreadPoolExecutor(NTHREADS) as pool:
        list(pool.map(read, range(NTHREADS)))


def test_concurrent_configuration():
    config = Configuration()
    barrier = Barrier(NTHREADS)

    def hammer(t):
        key = f"arcae.test.free-threading.{t}"
        barrier.wait()
        for i in range(NITERATIONS):
            config[key] = str(i)
            assert config[key] == str(i)
            assert key in list(config)
            del config[key]
            assert key not in config

    with ThreadPoolExecutor(NTHREADS) as pool:
        list(pool.map(hammer, range(NTHREADS)))
