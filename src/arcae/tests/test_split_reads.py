import shutil

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import arcae

NROW = 70_000
NCHAN = 16
NCORR = 4
# Reads are split into pieces of at least this many bytes
MIN_SPLIT_BYTES = 8 * 1024 * 1024


@pytest.fixture(scope="module")
def split_table(tmp_path_factory):
    """A table whose DATA column is large enough to be split over instances"""
    ct = pytest.importorskip("casacore.tables")
    path = tmp_path_factory.mktemp("split") / "split.table"
    desc = ct.maketabdesc(
        [
            ct.makescacoldesc("TIME", 0.0),
            ct.makearrcoldesc(
                "DATA",
                0j,
                shape=[NCHAN, NCORR],
                datamanagertype="TiledColumnStMan",
                datamanagergroup="DATA",
            ),
        ]
    )
    rows = np.arange(NROW)
    data = rows[:, None, None] + 1j * np.arange(NCHAN * NCORR).reshape(NCHAN, NCORR)
    with ct.table(str(path), desc, nrow=NROW, ack=False) as T:
        T.putcol("TIME", rows.astype(np.float64))
        T.putcol("DATA", data)
    return str(path), data


rng = np.random.default_rng(42)
GAPPED = np.setdiff1d(np.arange(NROW), np.arange(0, NROW, 1000))
UNSORTED = rng.permutation(NROW)[: NROW // 2]


@pytest.mark.parametrize(
    "index",
    [
        None,
        (GAPPED,),
        (UNSORTED,),
        (slice(100, NROW - 100), np.array([5, 1, 2])),
        (UNSORTED, slice(2, 10), np.array([3, 0])),
    ],
    ids=["full", "gapped", "unsorted", "rows+chan", "unsorted+chan+corr"],
)
@pytest.mark.parametrize("ninstances", [2, 4, 8])
def test_split_reads_match(split_table, index, ninstances):
    path, data = split_table
    expected = data
    # Index each axis in turn: numpy broadcasts multiple index arrays together
    for axis, axis_index in enumerate(index or ()):
        expected = expected[(slice(None),) * axis + (axis_index,)]

    with arcae.table(path, ninstances=1) as T:
        assert_array_equal(T.getcol("DATA", index=index), expected)

    with arcae.table(path, ninstances=ninstances) as T:
        before = T._instance_stats()
        assert_array_equal(T.getcol("DATA", index=index), expected)
        after = T._instance_stats()

        # Supplying the result array reads into it
        result = np.zeros_like(expected)
        before_result = T._instance_stats()
        assert T.getcol("DATA", index=index, result=result) is result
        after_result = T._instance_stats()
        assert_array_equal(result, expected)

    # Reads of at least MIN_SPLIT_BYTES per instance are spread over the instances
    if expected.nbytes >= ninstances * MIN_SPLIT_BYTES:
        busy = [a["io_bytes"] - b["io_bytes"] for a, b in zip(after, before)]
        assert all(b > 0 for b in busy)
        # in one task, and so one lock, per instance: the partitioning
        # task reads a piece itself. Reading into a result array skips
        # the complex conversion, which looks up the column descriptor
        tasks = [
            a["completed"] - b["completed"] for a, b in zip(after_result, before_result)
        ]
        assert tasks == [1] * ninstances


NEWCOL_DESC = {
    "NEWCOL": {
        "valueType": "double",
        "option": 0,
        "maxlen": 0,
        "comment": "x",
        "keywords": {},
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": "SSM",
    }
}


@pytest.mark.parametrize("ninstances", [2, 4])
def test_split_read_after_column_added_elsewhere(split_table, tmp_path, ninstances):
    """A read split over instances that another handle's new column made stale"""
    src, data = split_table
    path = str(tmp_path / "split.table")
    shutil.copytree(src, path)

    with arcae.table(path, readonly=False, ninstances=ninstances) as B:
        # Every instance of B is out of date once A adds a column
        with arcae.table(path, readonly=False, ninstances=1) as A:
            A.addcols(NEWCOL_DESC, {})

        before = B._instance_stats()
        assert_array_equal(B.getcol("DATA"), data)
        after = B._instance_stats()
        assert "NEWCOL" in B.columns()

    busy = [a["io_bytes"] - b["io_bytes"] for a, b in zip(after, before)]
    assert all(b > 0 for b in busy)


def test_writes_count_against_writer_instance(split_table, tmp_path):
    """Writes run on instance 0, sharing its statistics with reads"""
    src, data = split_table
    path = str(tmp_path / "split.table")
    shutil.copytree(src, path)
    rows = np.arange(1000)

    with arcae.table(path, readonly=False, ninstances=4) as T:
        before = T._instance_stats()
        T.putcol("DATA", 2 * data[rows], index=(rows,))
        after = T._instance_stats()
        assert_array_equal(T.getcol("DATA", index=(rows,)), 2 * data[rows])

    written = [a["io_bytes"] - b["io_bytes"] for a, b in zip(after, before)]
    assert written[0] == data[rows].nbytes
    assert written[1:] == [0] * 3
