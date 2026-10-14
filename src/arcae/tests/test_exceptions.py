"""casacore errors must raise Python exceptions rather than abort the process.

See https://github.com/casangi/xradio/issues/637
"""

import pyarrow as pa
import pytest

from arcae.lib.arrow_tables import Table

TABLE_DESC = {
    "F": {
        "comment": "",
        "dataManagerGroup": "SSM",
        "dataManagerType": "StandardStMan",
        "keywords": {},
        "maxlen": 0,
        "option": 0,
        "valueType": "INT",
    },
}

UNKNOWN_DMINFO = {
    "*1": {
        "NAME": "SSM",
        "TYPE": "NoSuchStMan",
        "SPEC": {},
        "COLUMNS": ["F"],
    }
}


@pytest.fixture
def f_table(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("exceptions") / "test.table")
    Table.from_descriptor(path, table_desc=TABLE_DESC, nrow=2).close()
    return path


def test_taql_parse_error(f_table):
    """F is a TaQL keyword, so this query does not parse"""
    with Table.from_filename(f_table, readonly=False) as T:
        with pytest.raises(pa.ArrowInvalid, match="parse error"):
            Table.from_taql("UPDATE $1 SET F=1", T)

        # The table remains usable
        assert T.nrow() == 2


def test_taql_missing_table():
    with pytest.raises(pa.ArrowInvalid, match="does not exist"):
        Table.from_taql("SELECT FROM arcae_nonexistent_table.table")


def test_unknown_data_manager_from_descriptor(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("exceptions") / "test.table")

    with pytest.raises(pa.ArrowInvalid, match="nosuchstman"):
        Table.from_descriptor(
            path, table_desc=TABLE_DESC, dminfo=UNKNOWN_DMINFO, nrow=1
        )


def test_unknown_data_manager_ms_from_descriptor(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("exceptions") / "test.ms")
    dminfo = {**UNKNOWN_DMINFO, "*1": {**UNKNOWN_DMINFO["*1"], "COLUMNS": ["TIME"]}}

    with pytest.raises(pa.ArrowInvalid, match="nosuchstman"):
        Table.ms_from_descriptor(path, dminfo=dminfo)


def test_invalid_lockoptions(f_table):
    """casacore cannot represent a list of mixed types"""
    with pytest.raises(pa.ArrowInvalid, match="lockoptions"):
        Table.from_filename(f_table, lockoptions={"option": [1, {"x": "y"}]})
