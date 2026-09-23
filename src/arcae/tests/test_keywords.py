import numpy as np
import pytest

from arcae.lib.arrow_tables import Table

TABLE_DESC = {
    "TIME": {
        "comment": "TIME column",
        "dataManagerGroup": "StandardStMan",
        "dataManagerType": "StandardStMan",
        "keywords": {},
        "maxlen": 0,
        "option": 0,
        "valueType": "DOUBLE",
    },
}


@pytest.fixture
def keyword_table(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("keywords") / "test.tab")
    with Table.from_descriptor(path, table_desc=TABLE_DESC, nrow=2) as T:
        T.putcol("TIME", np.array([0.0, 1.0]))
    return path


def test_table_keywords_roundtrip(keyword_table):
    with Table.from_filename(keyword_table, readonly=False) as T:
        assert T.getkeywords() == {}
        T.putkeywords({"a_string": "foo", "an_int": 5, "a_double": 1.5})
        assert T.getkeywords() == {"a_string": "foo", "an_int": 5, "a_double": 1.5}

        # Absent keywords are left untouched
        T.putkeywords({"an_int": 6})
        assert T.getkeywords() == {"a_string": "foo", "an_int": 6, "a_double": 1.5}

        T.removekeyword("a_string")
        assert T.getkeywords() == {"an_int": 6, "a_double": 1.5}

    # Keywords persist across a re-open
    with Table.from_filename(keyword_table) as T:
        assert T.getkeywords() == {"an_int": 6, "a_double": 1.5}


def test_column_keywords_roundtrip(keyword_table):
    with Table.from_filename(keyword_table, readonly=False) as T:
        assert T.getcolkeywords("TIME") == {}
        T.putcolkeywords("TIME", {"QuantumUnits": ["s"], "MEASINFO": {"type": "epoch"}})
        keywords = T.getcolkeywords("TIME")
        assert keywords["QuantumUnits"] == ["s"]
        assert keywords["MEASINFO"] == {"type": "epoch"}

        # Table keywords and column keywords are distinct namespaces
        assert T.getkeywords() == {}

        T.removecolkeyword("TIME", "MEASINFO")
        assert set(T.getcolkeywords("TIME")) == {"QuantumUnits"}

    with Table.from_filename(keyword_table) as T:
        assert T.getcolkeywords("TIME")["QuantumUnits"] == ["s"]


def test_subtable_keyword(tmp_path_factory, keyword_table):
    """A "Table: <path>" string defines a subtable keyword, which is
    then resolvable through casacore's :: syntax"""
    path = str(tmp_path_factory.mktemp("subtable") / "parent.tab")
    with Table.from_descriptor(path, table_desc=TABLE_DESC, nrow=1) as T:
        T.putkeywords({"CHILD": f"Table: {keyword_table}"})

    with Table.from_filename(f"{path}::CHILD") as S:
        assert S.columns() == ["TIME"]
        np.testing.assert_array_equal(S.getcol("TIME"), [0.0, 1.0])


def test_keywords_on_missing_column(keyword_table):
    with Table.from_filename(keyword_table, readonly=False) as T:
        with pytest.raises(Exception, match="MISSING"):
            T.getcolkeywords("MISSING")

        with pytest.raises(Exception, match="MISSING"):
            T.putcolkeywords("MISSING", {"a": 1})

        with pytest.raises(ValueError, match="column name is required"):
            T.getcolkeywords("")


def test_remove_missing_keyword(keyword_table):
    with Table.from_filename(keyword_table, readonly=False) as T:
        with pytest.raises(Exception, match="does not exist"):
            T.removekeyword("nonexistent")
