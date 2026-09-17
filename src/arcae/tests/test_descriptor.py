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


def test_phased_array_subtable_creation(tmp_path_factory):
    ms = tmp_path_factory.mktemp("test") / "test.ms"
    with Table.ms_from_descriptor(str(ms)):
        pass

    phased_desc = ms_descriptor("PHASED_ARRAY", complete=False)
    with Table.ms_from_descriptor(str(ms), "PHASED_ARRAY", phased_desc) as phased:
        phased.addrows(1)
        assert phased.nrow() == 1

    with Table.from_filename(str(ms)) as main:
        assert "PHASED_ARRAY" in main.tabledesc()["_keywords_"]

    with Table.from_filename(f"{ms}::PHASED_ARRAY") as phased:
        assert phased.nrow() == 1


def test_phased_array_subtable_descriptor():
    assert ms_descriptor("PHASED_ARRAY", complete=True) == {
        "ANTENNA_ID": {
            "comment": "Antenna ID",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "int",
        },
        "POSITION": {
            "comment": "Position of antenna field",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {
                "MEASINFO": {"Ref": "ITRF", "type": "position"},
                "QuantumUnits": ["m", "m", "m"],
            },
            "maxlen": 0,
            "ndim": 1,
            "shape": [3],
            "_c_order": True,
            "option": 4,
            "valueType": "double",
        },
        "COORDINATE_AXES": {
            "_c_order": True,
            "comment": "Local coordinate system",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "ndim": 2,
            "shape": [3, 3],
            "option": 4,
            "valueType": "double",
        },
        "ELEMENT_FLAG": {
            "_c_order": True,
            "comment": "Flag of elements in array",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "ndim": 2,
            "option": 0,
            "valueType": "boolean",
        },
        "ELEMENT_OFFSET": {
            "_c_order": True,
            "comment": "Offset per element",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {
                "MEASINFO": {"Ref": "ITRF", "type": "position"},
                "QuantumUnits": ["m", "m", "m"],
            },
            "maxlen": 0,
            "ndim": 2,
            "option": 0,
            "valueType": "double",
        },
        "BEAM_ID": {
            "comment": "Beam ID",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "option": 0,
            "valueType": "int",
        },
        "_define_hypercolumn_": {},
        "_keywords_": {},
        "_private_keywords_": {},
    }


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
