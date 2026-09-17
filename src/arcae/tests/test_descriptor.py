import numpy as np
from numpy.testing import assert_array_equal

from arcae.lib.arrow_tables import Table, ms_descriptor


def test_descriptor_basic():
    assert isinstance(ms_descriptor("MAIN"), dict)
    assert isinstance(ms_descriptor("ANTENNA"), dict)
    assert isinstance(ms_descriptor("FEED"), dict)
    assert isinstance(ms_descriptor("SPECTRAL_WINDOW"), dict)
    assert isinstance(ms_descriptor("PHASED_ARRAY"), dict)


def test_ms_addrows(tmp_path_factory):
    ms = tmp_path_factory.mktemp("test") / "test.ms"
    with Table.ms_from_descriptor(str(ms)) as T:
        T.addrows(10)
        assert T.nrow() == 10
        AT = T.to_arrow()
        assert len(AT) == 10
        assert_array_equal(AT.column("TIME"), 0)
        assert_array_equal(AT.column("ANTENNA1"), 0)
        assert_array_equal(AT.column("ANTENNA2"), 0)


def test_ms_and_weather_subtable(tmp_path_factory):
    ms = tmp_path_factory.mktemp("test") / "test.ms"
    with Table.ms_from_descriptor(str(ms)) as T:
        assert (ms / "table.dat").exists()
        assert not (ms / "WEATHER").exists()
        assert "WEATHER" not in T.tabledesc()["_keywords_"]

    # Basic descriptor
    table_desc = ms_descriptor("WEATHER", complete=False)
    with Table.ms_from_descriptor(str(ms), "WEATHER", table_desc) as W:
        assert (ms / "WEATHER").exists()
        assert W.columns() == ["ANTENNA_ID", "INTERVAL", "TIME"]

    # Add a column to the basic descriptor
    table_desc = ms_descriptor("WEATHER", complete=False)
    table_desc["BLAH"] = table_desc["TIME"].copy()
    with Table.ms_from_descriptor(str(ms), "WEATHER", table_desc) as W:
        assert (ms / "WEATHER").exists()
        assert W.columns() == ["ANTENNA_ID", "BLAH", "INTERVAL", "TIME"]

    # Complete descriptor
    table_desc = ms_descriptor("WEATHER", complete=True)
    with Table.ms_from_descriptor(str(ms), "WEATHER", table_desc) as W:
        assert (ms / "WEATHER").exists()
        assert W.columns() == [
            "ANTENNA_ID",
            "DEW_POINT",
            "DEW_POINT_FLAG",
            "H2O",
            "H2O_FLAG",
            "INTERVAL",
            "IONOS_ELECTRON",
            "IONOS_ELECTRON_FLAG",
            "PRESSURE",
            "PRESSURE_FLAG",
            "REL_HUMIDITY",
            "REL_HUMIDITY_FLAG",
            "TEMPERATURE",
            "TEMPERATURE_FLAG",
            "TIME",
            "WIND_DIRECTION",
            "WIND_DIRECTION_FLAG",
            "WIND_SPEED",
            "WIND_SPEED_FLAG",
        ]

    # Weather table is linked in the Measurement Set
    with Table.from_filename(str(ms)) as T:
        td = T.tabledesc()
        assert "WEATHER" in td["_keywords_"]
        assert td["_keywords_"]["WEATHER"] == f"Table: {ms}/WEATHER"

    # Opening the table works with the subtable :: reference syntax
    with Table.from_filename(f"{ms}::WEATHER") as W:
        pass


def test_phased_array_descriptor_and_subtable(tmp_path_factory):
    required = ms_descriptor("PHASED_ARRAY", complete=False)
    complete = ms_descriptor("PHASED_ARRAY", complete=True)

    assert set(required) == {
        "ANTENNA_ID",
        "PHASED_ARRAY_ID",
        "POSITION",
        "COORDINATE_AXES",
        "ELEMENT_OFFSET",
        "ELEMENT_FLAG",
        "_define_hypercolumn_",
        "_keywords_",
        "_private_keywords_",
    }
    assert set(complete) == set(required) | {"BEAM_ID"}
    assert required["POSITION"]["shape"] == [3]
    assert required["COORDINATE_AXES"]["shape"] == [3, 3]
    assert required["ELEMENT_OFFSET"]["ndim"] == 2
    assert required["ELEMENT_FLAG"]["ndim"] == 2
    assert required["ELEMENT_OFFSET"]["keywords"] == {
        "MEASINFO": {"Ref": "ITRF", "type": "position"},
        "QuantumUnits": ["m"],
    }

    ms = tmp_path_factory.mktemp("test") / "test.ms"
    with Table.ms_from_descriptor(str(ms)) as main:
        assert main.nrow() == 0

    with Table.ms_from_descriptor(str(ms), "PHASED_ARRAY", required) as phased:
        assert phased.columns() == [
            "ANTENNA_ID",
            "COORDINATE_AXES",
            "ELEMENT_FLAG",
            "ELEMENT_OFFSET",
            "PHASED_ARRAY_ID",
            "POSITION",
        ]
        phased.addrows(2)
        phased.putcol("ANTENNA_ID", np.array([0, 1], dtype=np.int32))
        phased.putcol("PHASED_ARRAY_ID", np.array([10, 11], dtype=np.int32))
        phased.putcol(
            "POSITION",
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        )
        phased.putcol(
            "COORDINATE_AXES",
            np.arange(18, dtype=np.float64).reshape(2, 3, 3),
        )

        offsets = np.empty(2, dtype=object)
        offsets[0] = np.zeros((3, 2), dtype=np.float64).tolist()
        offsets[1] = np.ones((3, 3), dtype=np.float64).tolist()
        phased.putcol("ELEMENT_OFFSET", offsets)

        flags = np.empty(2, dtype=object)
        flags[0] = np.zeros((2, 2), dtype=bool).tolist()
        flags[1] = np.ones((2, 3), dtype=bool).tolist()
        phased.putcol("ELEMENT_FLAG", flags)

        assert phased.row_shapes("ELEMENT_OFFSET").to_pylist() == [[3, 2], [3, 3]]
        assert phased.row_shapes("ELEMENT_FLAG").to_pylist() == [[2, 2], [2, 3]]
        arrow = phased.to_arrow(columns=["ELEMENT_OFFSET", "ELEMENT_FLAG"])
        assert arrow.column("ELEMENT_OFFSET").to_pylist() == [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ]
        assert arrow.column("ELEMENT_FLAG").to_pylist() == [
            [[False, False], [False, False]],
            [[True, True, True], [True, True, True]],
        ]

    with Table.from_filename(str(ms)) as main:
        assert "PHASED_ARRAY" in main.tabledesc()["_keywords_"]

    with Table.from_filename(f"{ms}::PHASED_ARRAY") as phased:
        assert phased.nrow() == 2


def test_weather_subtable_descriptor():
    # Test required and complete descriptor for the WEATHER subtable
    assert ms_descriptor("WEATHER", complete=False) == {
        "ANTENNA_ID": {
            "comment": "Antenna number",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "int",
        },
        "INTERVAL": {
            "comment": "Interval over which data is relevant",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {"QuantumUnits": ["s"]},
            "maxlen": 0,
            "option": 0,
            "valueType": "double",
        },
        "TIME": {
            "comment": "An MEpoch specifying the midpoint of the time forwhich "
            "data is relevant",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {
                "MEASINFO": {"Ref": "UTC", "type": "epoch"},
                "QuantumUnits": ["s"],
            },
            "maxlen": 0,
            "option": 0,
            "valueType": "double",
        },
        "_define_hypercolumn_": {},
        "_keywords_": {},
        "_private_keywords_": {},
    }

    assert ms_descriptor("WEATHER", complete=True) == {
        "ANTENNA_ID": {
            "comment": "Antenna number",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "int",
        },
        "DEW_POINT": {
            "comment": "Dew point",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {"QuantumUnits": ["K"]},
            "maxlen": 0,
            "option": 0,
            "valueType": "float",
        },
        "DEW_POINT_FLAG": {
            "comment": "Flag for dew point",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "boolean",
        },
        "H2O": {
            "comment": "Average column density of water-vapor",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {"QuantumUnits": ["m-2"]},
            "maxlen": 0,
            "option": 0,
            "valueType": "float",
        },
        "H2O_FLAG": {
            "comment": "Flag for average column density of water-vapor",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "boolean",
        },
        "INTERVAL": {
            "comment": "Interval over which data is relevant",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {"QuantumUnits": ["s"]},
            "maxlen": 0,
            "option": 0,
            "valueType": "double",
        },
        "IONOS_ELECTRON": {
            "comment": "Average column density of electrons",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {"QuantumUnits": ["m-2"]},
            "maxlen": 0,
            "option": 0,
            "valueType": "float",
        },
        "IONOS_ELECTRON_FLAG": {
            "comment": "Flag for average column density of electrons",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "boolean",
        },
        "PRESSURE": {
            "comment": "Ambient atmospheric pressure",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {"QuantumUnits": ["hPa"]},
            "maxlen": 0,
            "option": 0,
            "valueType": "float",
        },
        "PRESSURE_FLAG": {
            "comment": "Flag for ambient atmospheric pressure",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "boolean",
        },
        "REL_HUMIDITY": {
            "comment": "Ambient relative humidity",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {"QuantumUnits": ["%"]},
            "maxlen": 0,
            "option": 0,
            "valueType": "float",
        },
        "REL_HUMIDITY_FLAG": {
            "comment": "Flag for ambient relative humidity",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "boolean",
        },
        "TEMPERATURE": {
            "comment": "Ambient Air Temperature for an antenna",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {"QuantumUnits": ["K"]},
            "maxlen": 0,
            "option": 0,
            "valueType": "float",
        },
        "TEMPERATURE_FLAG": {
            "comment": "Flag for ambient Air Temperature for an antenna",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "boolean",
        },
        "TIME": {
            "comment": "An MEpoch specifying the midpoint of the time forwhich "
            "data is relevant",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {
                "MEASINFO": {"Ref": "UTC", "type": "epoch"},
                "QuantumUnits": ["s"],
            },
            "maxlen": 0,
            "option": 0,
            "valueType": "double",
        },
        "WIND_DIRECTION": {
            "comment": "Average wind direction",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {"QuantumUnits": ["rad"]},
            "maxlen": 0,
            "option": 0,
            "valueType": "float",
        },
        "WIND_DIRECTION_FLAG": {
            "comment": "Flag for wind direction",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "boolean",
        },
        "WIND_SPEED": {
            "comment": "Average wind speed",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {"QuantumUnits": ["m/s"]},
            "maxlen": 0,
            "option": 0,
            "valueType": "float",
        },
        "WIND_SPEED_FLAG": {
            "comment": "Flag for wind speed",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "boolean",
        },
        "_define_hypercolumn_": {},
        "_keywords_": {},
        "_private_keywords_": {},
    }
