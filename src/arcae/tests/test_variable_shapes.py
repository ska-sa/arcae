"""Reads of variably shaped columns, checked against python-casacore.

Covers StandardStMan and TiledShapeStMan columns,
in plain tables and TaQL reference tables.
"""

import numpy as np
import pyarrow as pa
import pytest
from numpy.testing import assert_array_equal

import arcae
from arcae.lib.arrow_tables import Table

# python-casacore provides the expected values, and has no wheels for some platforms
ct = pytest.importorskip("casacore.tables")

NROW = 40
SMALL = (4, 2)
LARGE = (6, 2)
# Rows with an undefined cell in the MISSING column
MISSING_ROWS = {3, 17, 18, 39}
# Rows not selected by the TaQL reference table
TAQL_WHERE = "ROWID() % 3 != 1"


def row_shape(r):
    """Shapes alternate in runs of 5 rows"""
    return SMALL if (r // 5) % 2 == 0 else LARGE


def cell(r, shape):
    return np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + r * 100


@pytest.fixture(scope="module", params=["StandardStMan", "TiledShapeStMan"])
def variable_table(request, tmp_path_factory):
    dm = request.param
    path = str(tmp_path_factory.mktemp(dm) / "variable.table")
    desc = ct.maketabdesc(
        [
            ct.makearrcoldesc(
                name, 0.0, ndim=2, datamanagertype=dm, datamanagergroup=name
            )
            for name in ("UNIFORM", "RAGGED", "MISSING")
        ]
    )

    with ct.table(path, desc, nrow=NROW, ack=False) as T:
        for r in range(NROW):
            T.putcell("UNIFORM", r, cell(r, SMALL))
            T.putcell("RAGGED", r, cell(r, row_shape(r)))
            if r not in MISSING_ROWS:
                T.putcell("MISSING", r, cell(r, SMALL))

        assert T.getdminfo("RAGGED")["TYPE"] == dm

    return path


@pytest.fixture(params=["plain", "taql"])
def tables(request, variable_table):
    """Yields matching arcae and python-casacore tables"""
    with arcae.table(variable_table) as A, ct.table(variable_table, ack=False) as C:
        if request.param == "plain":
            yield A, C
            return

        query = f"SELECT FROM $1 WHERE {TAQL_WHERE}"
        with Table.from_taql(query, A) as AR, ct.taql(query, tables=[C]) as CR:
            assert AR.nrow() == CR.nrows() < NROW
            yield AR, CR


def expected_cells(C, column, rows):
    return [C.getcell(column, int(r)) for r in rows]


def arcae_cells(A, column, index):
    array = A.to_arrow(index=index, columns=column).column(column).combine_chunks()
    assert len(array) > 0
    return [np.array(v) for v in array.to_pylist()]


def test_uniform_shapes(tables):
    """Variably shaped columns whose cells share a shape read as numpy arrays"""
    A, C = tables
    assert_array_equal(A.getcol("UNIFORM"), C.getcol("UNIFORM"))

    rows = np.array([5, 0, 9, 2])
    assert_array_equal(A.getcol("UNIFORM", index=(rows,)), C.getcol("UNIFORM")[rows])


@pytest.mark.parametrize("unsorted", [False, True])
def test_ragged_shapes(tables, unsorted):
    A, C = tables
    rows = np.arange(C.nrows())
    if unsorted:
        rows = np.random.default_rng(42).permutation(rows)

    actual = arcae_cells(A, "RAGGED", (rows,))
    expected = expected_cells(C, "RAGGED", rows)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert_array_equal(a, e)

    with pytest.raises(TypeError, match="variably shaped"):
        A.getcol("RAGGED")


def test_ragged_secondary_selection(tables):
    """Selecting the first and third elements of the first dimension
    clips every ragged cell to the same shape"""
    A, C = tables
    rows = np.arange(C.nrows())
    expected = np.stack([c[[0, 2], :] for c in expected_cells(C, "RAGGED", rows)])
    assert_array_equal(A.getcol("RAGGED", index=(None, np.array([0, 2]))), expected)


def test_selection_exceeds_shape(tables):
    """Element 5 exists in the larger cells only"""
    A, _ = tables
    with pytest.raises(IndexError, match="exceeds dimension"):
        A.getcol("RAGGED", index=(None, np.array([5])))


def test_missing_rows(tables):
    A, C = tables
    defined = [C.iscelldefined("MISSING", r) for r in range(C.nrows())]
    assert not all(defined)

    shapes = A.row_shapes("MISSING")
    assert shapes.null_count == defined.count(False)
    for r, is_defined in enumerate(defined):
        if is_defined:
            assert tuple(shapes[r].as_py()) == SMALL
        else:
            assert not shapes[r].is_valid

    with pytest.raises(pa.ArrowInvalid, match="not defined"):
        A.getcol("MISSING")

    # Selecting only defined rows succeeds
    rows = np.flatnonzero(defined)
    expected = np.stack(expected_cells(C, "MISSING", rows))
    assert_array_equal(A.getcol("MISSING", index=(rows,)), expected)
