import concurrent.futures as cf

import numpy as np
import pyarrow as pa
import pytest
from numpy.testing import assert_array_equal

import arcae
from arcae.lib.arrow_tables import Table

NROW = 200

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


@pytest.fixture
def ms(tmp_path):
    path = str(tmp_path / "addcols.ms")
    Table.ms_from_descriptor(path, subtable="MAIN").close()
    return path


@pytest.mark.parametrize("ninstances", [1, 2, 4])
def test_addcols_then_close(ms, ninstances):
    """Adding a column must not leave sibling instances uncloseable"""
    T = arcae.table(ms, readonly=False, ninstances=ninstances)
    T.addcols(NEWCOL_DESC, {})
    T.close()


@pytest.mark.parametrize("ninstances", [2, 4])
def test_addcols_then_concurrent_reads(ms, ninstances):
    """Reads landing on sibling instances must see the new column"""
    with arcae.table(ms, readonly=False, ninstances=ninstances) as T:
        T.addrows(NROW)
        T.addcols(NEWCOL_DESC, {})
        assert "NEWCOL" in T.columns()
        data = np.arange(NROW, dtype=np.float64)
        T.putcol("NEWCOL", data)

        def read(i):
            column = "NEWCOL" if i % 2 == 0 else "TIME"
            return column, T.getcol(column)

        with cf.ThreadPoolExecutor(16) as pool:
            results = list(pool.map(read, range(200)))

        for column, result in results:
            if column == "NEWCOL":
                assert_array_equal(result, data)
            else:
                assert result.shape == (NROW,)


def test_addcols_twice(ms):
    """Refreshed instances must themselves survive a later column addition"""
    with arcae.table(ms, readonly=False, ninstances=4) as T:
        T.addrows(NROW)
        T.addcols(NEWCOL_DESC, {})
        T.addcols({"OTHERCOL": {**NEWCOL_DESC["NEWCOL"], "comment": "y"}}, {})

        def read(i):
            return T.getcol("OTHERCOL" if i % 2 == 0 else "NEWCOL")

        with cf.ThreadPoolExecutor(16) as pool:
            for result in pool.map(read, range(200)):
                assert result.shape == (NROW,)


def test_addcols_on_created_ms(tmp_path):
    """Refreshing instances of a table created, not opened, must reopen it,
    not create it again"""
    path = str(tmp_path / "created.ms")
    with Table.ms_from_descriptor(path, subtable="MAIN") as T:
        T.addrows(NROW)
        T.addcols(NEWCOL_DESC, {})
        assert "NEWCOL" in T.columns()
        assert T.nrow() == NROW

    with arcae.table(path) as T:
        assert "NEWCOL" in T.columns()
        assert T.nrow() == NROW


def test_addcols_with_live_taql_table(ms):
    """A TAQL table derived from this one keeps the stale table alive on the
    isolation threads it shares with it, which must not stop those threads'
    instances being refreshed"""
    T = arcae.table(ms, readonly=False, ninstances=4)
    T.addrows(NROW)
    Q = Table.from_taql("SELECT FROM $1 WHERE ROWID() < 50", [T])
    assert Q.nrow() == 50

    T.addcols(NEWCOL_DESC, {})
    data = np.arange(NROW, dtype=np.float64)
    T.putcol("NEWCOL", data)

    with cf.ThreadPoolExecutor(16) as pool:
        for result in pool.map(lambda i: T.getcol("NEWCOL"), range(200)):
            assert_array_equal(result, data)

    # The derived table's own instances are not refreshed, so it cannot
    # flush on close. It must still release them, leaving the parent
    # closeable. See design/addcols-stale-instances.md, fix 3.
    with pytest.raises(pa.lib.ArrowInvalid, match="number of columns"):
        Q.close()
    T.close()

    with arcae.table(ms) as T:
        assert_array_equal(T.getcol("NEWCOL"), data)
