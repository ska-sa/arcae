import numpy as np
import pytest
from numpy.testing import assert_array_equal

from arcae.lib.arrow_tables import Table

NCHAN = 8
NCORR = 4

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
    "NAMES": {
        "keywords": {},
        "maxlen": 0,
        "option": 0,
        "valueType": "STRING",
    },
    "DATA": {
        "_c_order": True,
        "comment": "DATA column",
        "dataManagerGroup": "DATA_GROUP",
        "dataManagerType": "TiledColumnStMan",
        "keywords": {},
        "maxlen": 0,
        "ndim": 2,
        "option": 0,
        "shape": [NCHAN, NCORR],
        "valueType": "COMPLEX",
    },
}

DMINFO = {
    "*1": {
        "NAME": "DATA_GROUP",
        "TYPE": "TiledColumnStMan",
        "SPEC": {"DEFAULTTILESHAPE": [NCORR, NCHAN, 2]},
        "COLUMNS": ["DATA"],
    }
}


def test_from_descriptor(tmp_path_factory):
    """A plain table contains exactly the described columns --
    no MeasurementSet columns are merged in"""
    path = str(tmp_path_factory.mktemp("plain") / "test.tab")

    with Table.from_descriptor(path, table_desc=TABLE_DESC, dminfo=DMINFO, nrow=3) as T:
        assert sorted(T.columns()) == ["DATA", "NAMES", "TIME"]
        assert T.nrow() == 3
        assert T.ncolumns() == 3

        time = np.array([0.0, 1.0, 2.0])
        names = np.array(["foo", "bar", "qux"])
        data = np.arange(3 * NCHAN * NCORR, dtype=np.float32)
        data = (data + data * 1j).astype(np.complex64)
        data = data.reshape(3, NCHAN, NCORR)

        T.putcol("TIME", time)
        T.putcol("NAMES", names)
        T.putcol("DATA", data)

        assert_array_equal(T.getcol("TIME"), time)
        assert_array_equal(T.getcol("NAMES"), names)
        assert_array_equal(T.getcol("DATA"), data)

        # The requested data manager was honoured
        dminfo = T.getdminfo()
        data_group = next(v for v in dminfo.values() if v["NAME"] == "DATA_GROUP")
        assert data_group["TYPE"] == "TiledColumnStMan"
        assert data_group["COLUMNS"] == ["DATA"]


def test_from_descriptor_no_rows(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("norows") / "test.tab")

    with Table.from_descriptor(path, table_desc=TABLE_DESC) as T:
        assert T.nrow() == 0
        T.addrows(2)
        assert T.nrow() == 2


def test_from_descriptor_reopen(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("reopen") / "test.tab")

    with Table.from_descriptor(path, table_desc=TABLE_DESC, dminfo=DMINFO, nrow=1) as T:
        T.putcol("TIME", np.array([7.0]))

    with Table.from_filename(path) as T:
        assert sorted(T.columns()) == ["DATA", "NAMES", "TIME"]
        assert_array_equal(T.getcol("TIME"), [7.0])


def test_from_descriptor_empty_name():
    with pytest.raises(Exception, match="must not be empty"):
        Table.from_descriptor("", table_desc=TABLE_DESC)


def test_from_descriptor_bad_descriptor(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("bad") / "test.tab")

    with pytest.raises(Exception):
        Table.from_descriptor(path, table_desc={"BAD": {"valueType": "NOSUCHTYPE"}})


def test_numpy_values_in_descriptors(tmp_path_factory):
    """casacore descriptors routinely carry numpy values, which the
    stdlib JSON encoder cannot serialise"""
    path = str(tmp_path_factory.mktemp("numpy_desc") / "test.tab")
    dminfo = {
        "*1": {
            "NAME": "DATA_GROUP",
            "TYPE": "TiledColumnStMan",
            # DEFAULTTILESHAPE is an int32 array in practice
            "SPEC": {"DEFAULTTILESHAPE": np.int32([NCORR, NCHAN, 2])},
            "COLUMNS": ["DATA"],
        }
    }
    table_desc = {
        "DATA": {**TABLE_DESC["DATA"], "shape": np.int32([NCHAN, NCORR])},
    }

    with Table.from_descriptor(path, table_desc=table_desc, dminfo=dminfo, nrow=1) as T:
        group = next(v for v in T.getdminfo().values() if v["NAME"] == "DATA_GROUP")
        assert group["TYPE"] == "TiledColumnStMan"


def test_numpy_values_in_keywords(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("numpy_kw") / "test.tab")
    with Table.from_descriptor(path, table_desc=TABLE_DESC, nrow=1) as T:
        T.putkeywords({"scalar": np.int32(7), "array": np.arange(3, dtype=np.int32)})
        keywords = T.getkeywords()
        assert keywords["scalar"] == 7
        assert list(keywords["array"]) == [0, 1, 2]
