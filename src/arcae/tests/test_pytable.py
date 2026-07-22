import os

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pad
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

import arcae
from arcae.lib.arrow_tables import Table


@pytest.mark.parametrize(
    "table_suffix, table_name",
    [
        ("", "MAIN"),
        ("::ANTENNA", "ANTENNA"),
        ("::FEED", "FEED"),
        ("::POLARIZATION", "POLARIZATION"),
        ("::SPECTRAL_WINDOW", "SPECTRAL_WINDOW"),
    ],
)
def test_parquet_write(tmp_path, tau_ms, table_suffix, table_name):
    """Test conversion of a representative MS and some of its subtables to parquet format"""
    T = arcae.table(f"{tau_ms}{table_suffix}").to_arrow()
    pq.write_table(T, str(tmp_path / f"{table_name}.parquet"))


def test_descriptors(column_case_table):
    with arcae.table(column_case_table) as T:
        td = T.tabledesc()

        for column in T.columns():
            assert td[column] == T.getcoldesc(column)

        dminfo = T.getdminfo()
        dminfo["*1"].pop("SPEC")  # Varies
        assert dminfo == {
            "*1": {
                "COLUMNS": [
                    "FIXED",
                    "FIXED_STRING",
                    "SCALAR",
                    "SCALAR_STRING",
                    "UNCONSTRAINED",
                    "UNCONSTRAINED_SAME_NDIM",
                    "VARIABLE",
                    "VARIABLE_STRING",
                ],
                "NAME": "StandardStMan",
                "SEQNR": 0,
                "TYPE": "StandardStMan",
            }
        }


def test_add_columns(column_case_table):
    with arcae.table(column_case_table) as T:
        shape = [16, 4]
        tile_shape = tuple(reversed(shape)) + (T.nrow(),)

        # Add ACK and BAR to the ACKBAR_GROUP at the same time succeeds
        desc = {
            "ACK": {
                "dataManagerGroup": "ACKBAR_GROUP",
                "dataManagerType": "TiledColumnStMan",
                "ndim": len(shape),
                "shape": shape,
                "valueType": "BOOLEAN",
            },
            "BAR": {
                "dataManagerGroup": "ACKBAR_GROUP",
                "dataManagerType": "TiledColumnStMan",
                "ndim": len(shape),
                "shape": shape,
                "valueType": "COMPLEX",
            },
        }

        dminfo = {
            "*1": {
                "NAME": "ACKBAR_GROUP",
                "TYPE": "TiledColumnStMan",
                "SPEC": {"DEFAULTTILESHAPE": tile_shape},
                "COLUMNS": ["ACK", "BAR"],
            }
        }

        T.addcols(desc, dminfo)
        groups = {g["NAME"]: g for g in T.getdminfo().values()}
        assert set(["ACKBAR_GROUP", "StandardStMan"]) == set(groups.keys())
        assert set(groups["ACKBAR_GROUP"]["COLUMNS"]) == set(["ACK", "BAR"])
        assert {"ACK", "BAR"}.issubset(T.columns())


def test_column_selection(column_case_table):
    with arcae.table(column_case_table) as T:
        assert sorted(T.to_arrow().column_names) == [
            "FIXED",
            "FIXED_STRING",
            "SCALAR",
            "SCALAR_STRING",
            # Even though the column is unconstrained, ndim is the same
            "UNCONSTRAINED_SAME_NDIM",
            "VARIABLE",
            "VARIABLE_STRING",
        ]

    with arcae.table(column_case_table) as T:
        assert sorted(T.to_arrow([[0, 1]]).column_names) == [
            "FIXED",
            "FIXED_STRING",
            "SCALAR",
            "SCALAR_STRING",
            "UNCONSTRAINED_SAME_NDIM",
            "VARIABLE",
            "VARIABLE_STRING",
        ]

    with arcae.table(column_case_table) as T:
        assert sorted(T.to_arrow([[0, 1]], "VARIABLE").column_names) == ["VARIABLE"]

    with arcae.table(column_case_table) as T:
        assert sorted(T.to_arrow([[0, 1]], ["VARIABLE", "FIXED"]).column_names) == [
            "FIXED",
            "VARIABLE",
        ]


def test_column_cases(column_case_table):
    """Test code paths"""
    T = arcae.table(column_case_table).to_arrow()

    assert T.column("VARIABLE").to_pylist() == [
        [[[0, 0]], [[0, 0]], [[0, 0]]],
        [[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]],
        [[[2, 2], [2, 2], [2, 2]], [[2, 2], [2, 2], [2, 2]], [[2, 2], [2, 2], [2, 2]]],
    ]

    assert T.column("FIXED").to_pylist() == [
        [[0, 0, 0, 0], [0, 0, 0, 0]],
        [[1, 1, 1, 1], [1, 1, 1, 1]],
        [[2, 2, 2, 2], [2, 2, 2, 2]],
    ]

    assert T.column("SCALAR").to_pylist() == [0, 1, 2]

    assert T.column("VARIABLE_STRING").to_pylist() == [
        [[["0", "0"]], [["0", "0"]], [["0", "0"]]],
        [[["1", "1"], ["1", "1"]], [["1", "1"], ["1", "1"]], [["1", "1"], ["1", "1"]]],
        [
            [["2", "2"], ["2", "2"], ["2", "2"]],
            [["2", "2"], ["2", "2"], ["2", "2"]],
            [["2", "2"], ["2", "2"], ["2", "2"]],
        ],
    ]

    assert T.column("FIXED_STRING").to_pylist() == [
        [["0", "0", "0", "0"], ["0", "0", "0", "0"]],
        [["1", "1", "1", "1"], ["1", "1", "1", "1"]],
        [["2", "2", "2", "2"], ["2", "2", "2", "2"]],
    ]

    assert T.column("SCALAR_STRING").to_pylist() == ["0", "1", "2"]

    # Unconstrained columns not yet handled
    assert "UNCONSTRAINED" not in T.column_names


def test_row_shapes(column_case_table):
    table = arcae.table(column_case_table)

    assert isinstance(table.row_shapes("SCALAR"), pa.UInt8Array)
    assert table.row_shapes("SCALAR").to_pylist() == [1, 1, 1]

    assert isinstance(table.row_shapes("VARIABLE"), pa.FixedSizeListArray)
    assert table.row_shapes("VARIABLE").to_pylist() == [[3, 1, 2], [3, 2, 2], [3, 3, 2]]
    assert table.row_shapes("VARIABLE", ([1, 2],)).to_pylist() == [[3, 2, 2], [3, 3, 2]]
    assert table.row_shapes("VARIABLE", ([0],)).to_pylist() == [[3, 1, 2]]

    assert isinstance(table.row_shapes("FIXED"), pa.FixedSizeListArray)
    assert table.row_shapes("FIXED").to_pylist() == [[2, 4], [2, 4], [2, 4]]
    assert table.row_shapes("FIXED", ([1, 2],)).to_pylist() == [[2, 4], [2, 4]]
    assert table.row_shapes("FIXED", ([0],)).to_pylist() == [[2, 4]]


def test_complex_cases(complex_case_table):
    from arcae import config

    table = arcae.table(complex_case_table)

    with config.set(**{"casa.convert.strategy": "fixed"}):
        T = table.to_arrow()
        assert T.column("COMPLEX").type == pa.list_(
            pa.list_(pa.list_(pa.float64(), 2), 4), 2
        )

    with config.set(**{"casa.convert.strategy": "list"}):
        T = table.to_arrow()
        assert T.column("COMPLEX").type == pa.list_(pa.list_(pa.list_(pa.float64())))


def test_unordered_select_roundtrip(tmp_path):
    """Tests writing and reading using indexing"""
    from arcae.lib.arrow_tables import Table, ms_descriptor

    ms = str(tmp_path / "unorded_select.ms")
    table_desc = ms_descriptor("MAIN", complete=False)

    table_desc["DATA"] = {
        "comment": "Antenna number",
        "dataManagerGroup": "StandardStMan",
        "dataManagerType": "StandardStMan",
        "keywords": {},
        "ndim": 2,
        "shape": [3, 3],
        "maxlen": 0,
        "option": 0,
        "valueType": "float",
    }

    with Table.ms_from_descriptor(ms, table_desc=table_desc) as T:
        T.addrows(3)
        zeros = np.zeros((3, 3, 3), dtype=np.float32)
        T.putcol("DATA", np.arange(3 * 3 * 3, dtype=np.float32).reshape(3, 3, 3))

        assert_array_equal(
            T.getcol("DATA"),
            [
                [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                [[18, 19, 20], [21, 22, 23], [24, 25, 26]],
            ],
        )

        index = (np.array([2, 0]), np.array([2, 0]), np.array([2, 0]))
        expected = np.array([[[26, 24], [20, 18]], [[8, 6], [2, 0]]], np.float32)
        assert_array_equal(T.getcol("DATA", index=index), expected)

        T.putcol("DATA", zeros)
        assert_array_equal(T.getcol("DATA"), 0)

        T.putcol("DATA", expected, index=index)
        assert_array_equal(
            T.getcol("DATA"),
            [
                [[0, 0, 2], [0, 0, 0], [6, 0, 8]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[18, 0, 20], [0, 0, 0], [24, 0, 26]],
            ],
        )

        T.putcol("DATA", zeros)
        assert_array_equal(T.getcol("DATA"), 0)

        index = (np.array([0, 2]),) * 3
        T.putcol("DATA", expected, index=index)
        assert_array_equal(
            T.getcol("DATA"),
            [
                [[26, 0, 24], [0, 0, 0], [20, 0, 18]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[8, 0, 6], [0, 0, 0], [2, 0, 0]],
            ],
        )


def test_getcol(getcol_table):
    T = arcae.table(getcol_table, readonly=False)

    assert_array_equal(T.getcol("TIME"), [0, 1, 2])
    assert_array_equal(T.getcol("TIME", (slice(0, 2),)), [0, 1])
    assert_array_equal(T.getcol("TIME", (np.array([0, 1]),)), [0, 1])
    assert_array_equal(T.getcol("TIME", (np.array([2, 0]),)), [2, 0])
    assert_array_equal(T.getcol("TIME", (np.array([0, 2]),)), [0, 2])
    assert_array_equal(T.getcol("TIME", (None,)), [0, 1, 2])
    assert_array_equal(T.getcol("TIME", (slice(None),)), [0, 1, 2])
    assert_array_equal(T.getcol("STRING"), ["0", "1", "2"])

    index = (None, None, np.array([0, -1, -1, 2], np.int64))
    result = np.zeros((3, 2, 4), dtype=np.complex128)
    assert_array_equal(
        T.getcol("COMPLEX_DATA", index=index, result=result),
        [
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[1 + 1j, 0, 0, 1 + 1j], [1 + 1j, 0, 0, 1 + 1j]],
            [[2 + 2j, 0, 0, 2 + 2j], [2 + 2j, 0, 0, 2 + 2j]],
        ],
    )

    assert_array_equal(
        T.getcol("FLOAT_DATA"),
        [
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1]],
            [[2, 2, 2, 2], [2, 2, 2, 2]],
        ],
    )

    assert_array_equal(
        T.getcol("FLOAT_DATA", (slice(0, 2), slice(0, 2))),
        [[[0, 0, 0, 0], [0, 0, 0, 0]], [[1, 1, 1, 1], [1, 1, 1, 1]]],
    )

    assert_array_equal(
        T.getcol("FLOAT_DATA", (np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))),
        [[[0, 0], [0, 0]], [[1, 1], [1, 1]]],
    )

    # Test partial round-trip
    index = (np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))
    float_data = np.array([[[2, 3], [4, 5]], [[9, 8], [7, 6]]], np.float32)

    T.putcol("FLOAT_DATA", float_data + 1, index)
    assert_array_equal(T.getcol("FLOAT_DATA", index), float_data + 1)

    # Test complete round-trips
    float_data = np.array(
        [
            [[3, 2, 1, 0], [3, 2, 1, 0]],
            [[0, 1, 2, 3], [0, 1, 2, 3]],
            [[2, 2, 2, 2], [2, 2, 2, 2]],
        ],
        np.float32,
    )

    complex_data = np.array(
        [
            [[3 + 3j, 2 + 2j, 1 + 1j, 0 + 0j], [3 + 3j, 2 + 2j, 1 + 1j, 0 + 0j]],
            [[0 + 0j, 1 + 1j, 2 + 2j, 3 + 3j], [0 + 0j, 1 + 1j, 2 + 2j, 3 + 3j]],
            [[2 + 2j, 2 + 2j, 2 + 2j, 2 + 2j], [2 + 2j, 2 + 2j, 2 + 2j, 2 + 2j]],
        ],
        np.complex128,
    )

    T.putcol("FLOAT_DATA", float_data + 1)
    assert_array_equal(T.getcol("FLOAT_DATA"), float_data + 1)

    T.putcol("COMPLEX_DATA", complex_data + 1 + 1j)
    assert_array_equal(T.getcol("COMPLEX_DATA"), complex_data + 1 + 1j)

    for r in range(T.nrow()):
        row_data = T.getcol("VARDATA", index=(slice(r, r + 1),))
        assert_array_equal(row_data, np.full((1, r + 1, r + 1), r))

    with pytest.raises(TypeError, match="variably shaped column VARDATA"):
        T.getcol("VARDATA")

    with pytest.raises(pa.lib.ArrowException, match="NONEXISTENT does not exist"):
        T.getcol("NONEXISTENT")


def test_partial_read(sorting_table):
    """Tests that partial reads work"""
    T = arcae.table(sorting_table)
    full = T.to_arrow()
    nrows = [1, 2, 3, 4]
    assert sum(nrows) == len(full) == 10

    start = 0

    for nrow in nrows:
        assert full.take(list(range(start, start + nrow))) == T.to_arrow(
            [slice(start, start + nrow)]
        )
        start += nrow


def test_table_name(sorting_table):
    T = arcae.table(sorting_table)
    assert T.name() == sorting_table


def test_table_taql(sorting_table):
    """Tests that basic taql queries work"""
    with arcae.table(sorting_table) as T:
        AT = T.to_arrow()

    with Table.from_taql(
        f"SELECT * FROM {sorting_table} ORDER BY TIME, ANTENNA1, ANTENNA2"
    ) as Q:
        assert (
            AT.sort_by(
                [
                    ("TIME", "ascending"),
                    ("ANTENNA1", "ascending"),
                    ("ANTENNA2", "ascending"),
                ]
            )
            == Q.to_arrow()
        )

    with Table.from_taql(
        f"SELECT * FROM {sorting_table} ORDER BY ANTENNA1, ANTENNA2, TIME"
    ) as Q:
        assert (
            AT.sort_by(
                [
                    ("ANTENNA1", "ascending"),
                    ("ANTENNA2", "ascending"),
                    ("TIME", "ascending"),
                ]
            )
            == Q.to_arrow()
        )


def test_complex_taql(sorting_table):
    """Test a relatively complex taql query"""

    query = f"""
    SELECT
        ANTENNA1,
        ANTENNA2,
        GROWID() as ROW,
        GAGGR(TIME) as TIME,
        GAGGR(FIELD_ID) as FIELD_ID,
        GAGGR(SCAN_NUMBER) as SCAN_NUMBER
    FROM
        {sorting_table}
    GROUPBY
        ANTENNA1,
        ANTENNA2
    """

    with arcae.table(sorting_table) as T:
        AT = T.to_arrow()
        group_cols = ["ANTENNA1", "ANTENNA2"]
        agg_cols = [
            ("TIME", "list"),
            ("FIELD_ID", "list"),
            ("ROW", "list"),
            ("SCAN_NUMBER", "list"),
        ]
        AT = AT.append_column("ROW", pa.array(np.arange(len(AT))))
        AT = AT.group_by(group_cols).aggregate(agg_cols)
        new_names = [
            c[: -len("_list")] if c.endswith("_list") else c for c in AT.column_names
        ]
        AT = AT.rename_columns(new_names)

    with Table.from_taql(query) as T:
        QT = T.to_arrow()
        # Ensure fields are ordered the same and have same types
        # TAQL returns indexing columns as int64 instead of original int32
        assert AT.select(QT.column_names).cast(QT.schema).equals(QT)


def test_taql_table_arg(sorting_table):
    with arcae.table(sorting_table) as T:
        query = """
        SELECT
            TIME,
            ANTENNA1,
            ANTENNA2,
            ROWID() AS ROW
        FROM
            $1
        ORDER BY
            TIME,
            ANTENNA1,
            ANTENNA2
        """
        with Table.from_taql(query, [T]) as Q:
            result = Q.to_arrow()

    assert_array_almost_equal(result["TIME"], np.linspace(0.1, 1.0, 10))
    assert_array_equal(result["ANTENNA1"], [1, 0, 0, 1, 2, 1, 1, 1, 0, 0])
    assert_array_equal(result["ANTENNA2"], [2, 1, 1, 0, 1, 2, 3, 2, 2, 1])
    assert_array_equal(result["ROW"], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])


@pytest.mark.skip(reason="https://github.com/apache/arrow/issues/43696")
def test_print_dataset_structure(partitioned_dataset):
    """Print the directory stucture of the partitioned dataset"""
    partitioned_dataset = str(partitioned_dataset)

    print("\nDATASET TREE\n")

    for root, dirs, files in os.walk(partitioned_dataset):
        level = root.replace(partitioned_dataset, "").count(os.sep)
        indent = " " * 4 * (level)
        print("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * 4 * (level + 1)
        for f in sorted(files):
            print("{}{}".format(subindent, f))


@pytest.mark.skip(reason="https://github.com/apache/arrow/issues/43696")
def test_dataset_predicates(partitioned_dataset):
    """Illustrate native arrow dataset predicates"""
    partitioned_dataset = pad.dataset(partitioned_dataset)
    predicate = (
        (pad.field("ANTENNA1") >= 0)
        & (pad.field("ANTENNA1") < 2)
        & (pad.field("ANTENNA2") >= 2)
        & (pad.field("ANTENNA2") <= 3)
    )

    T = partitioned_dataset.to_table(filter=predicate)
    antenna1 = T.column("ANTENNA1").to_numpy()
    antenna2 = T.column("ANTENNA2").to_numpy()

    assert antenna1.size > 0 and np.all((antenna1 >= 0) & (antenna1 < 2))
    assert antenna2.size > 0 and np.all((antenna2 >= 2) & (antenna2 <= 3))


@pytest.mark.skip(reason="https://github.com/apache/arrow/issues/43696")
def test_duckdb(partitioned_dataset):
    """Illustrate integation between dataset and duckdb"""
    partitioned_dataset = pad.dataset(partitioned_dataset)
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()

    query = con.execute(
        "SELECT TIME, ANTENNA1, ANTENNA2, DATA FROM partitioned_dataset "
        "WHERE ANTENNA1 >= 0 AND ANTENNA1 < 2 AND "
        "ANTENNA2 >= 2 AND ANTENNA2 <=3"
    )
    rbr = query.fetch_record_batch()

    chunks = []

    while True:
        try:
            chunks.append(rbr.read_next_batch())
        except StopIteration:
            break

    query_table = pa.Table.from_batches(chunks)
    antenna1 = query_table.column("ANTENNA1").to_numpy()
    antenna2 = query_table.column("ANTENNA2").to_numpy()

    assert antenna1.size > 0 and np.all((antenna1 >= 0) & (antenna1 < 2))
    assert antenna2.size > 0 and np.all((antenna2 >= 2) & (antenna2 <= 3))


def test_config():
    from arcae.lib.arrow_tables import Configuration

    global_config = Configuration()

    assert global_config["validation-level"] == "full"

    global_config["blah"] = "foo"
    assert global_config["blah"] == "foo"
    assert len(global_config) == 2

    global_config["qux"] = "bar"
    assert global_config["qux"] == "bar"
    assert len(global_config) == 3

    try:
        global_config["foo"] == "bar"
    except KeyError as e:
        assert "foo" in e.args[0]

    assert global_config.get("foo") is None
    assert global_config.get("foo", "bar") == "bar"

    assert list(global_config.items()) == [
        ("blah", "foo"),
        ("qux", "bar"),
        ("validation-level", "full"),
    ]

    try:
        global_config["foo"] = 1
    except TypeError as e:
        assert "(expected str, got int)" in e.args[0]

    del global_config["blah"]
    del global_config["qux"]

    assert list(global_config.items()) == [("validation-level", "full")]


def test_config_context_mgr():
    from arcae import config
    from arcae.lib.arrow_tables import Configuration

    global_config = Configuration()
    assert list(global_config.items()) == [("validation-level", "full")]

    with config.set(**{"foo": "bar", "qux-baz": "blah"}):
        assert global_config["foo"] == "bar"
        assert global_config["qux-baz"] == "blah"

    with pytest.raises(KeyError):
        assert global_config["foo"] == "foo"

    with pytest.raises(KeyError):
        assert global_config["qux-baz"] == "blah"

    with config.set(**{"foo": "bar"}):
        assert global_config["foo"] == "bar"

        with config.set(**{"qux": "baz"}):
            assert global_config["foo"] == "bar"
            assert global_config["qux"] == "baz"

        with pytest.raises(KeyError):
            global_config["qux"]

    with pytest.raises(KeyError):
        global_config["foo"]

    assert list(global_config.items()) == [("validation-level", "full")]


def test_cache_size_reads_are_consistent(tau_ms):
    """Bounding storage-manager caches must not change the data that is read."""
    baseline = arcae.table(tau_ms).to_arrow()

    # Single global default (MiB)
    assert arcae.table(tau_ms, cache_size=64).to_arrow().equals(baseline)

    # Unbounded escape hatch (the historical casacore behaviour)
    assert arcae.table(tau_ms, cache_size=0).to_arrow().equals(baseline)

    # Three levels combined, keyed off the table's real storage-manager and
    # column names (column: overrides stman: overrides default).
    groups = {g["NAME"]: g for g in arcae.table(tau_ms).getdminfo().values()}
    data_sm = next(name for name, g in groups.items() if "DATA" in g["COLUMNS"])
    spec = {"default": 128, f"stman:{data_sm}": 64, "column:DATA": 32}
    assert arcae.table(tau_ms, cache_size=spec).to_arrow().equals(baseline)


def test_cache_size_validation(tau_ms):
    """Malformed specs and unknown names are rejected."""
    # Unknown key prefix rejected during Python normalisation.
    with pytest.raises(ValueError, match="Invalid cache_size key"):
        arcae.table(tau_ms, cache_size={"bogus": 1})

    # Non-integer value rejected during normalisation.
    with pytest.raises(TypeError):
        arcae.table(tau_ms, cache_size={"default": "big"})

    # bool is not accepted as an int.
    with pytest.raises(TypeError):
        arcae.table(tau_ms, cache_size=True)

    # Unknown storage-manager / column names are rejected in the C++ layer.
    with pytest.raises(Exception, match="Unknown storage manager"):
        arcae.table(tau_ms, cache_size={"stman:NoSuchSm": 1})
    with pytest.raises(Exception, match="Unknown column"):
        arcae.table(tau_ms, cache_size={"column:NOSUCHCOLUMN": 1})
