import os
from contextlib import ExitStack
from tempfile import TemporaryDirectory

import numpy as np
from numpy.testing import assert_array_equal

from arcae.lib.arrow_tables import Table

NANT = 8
NCHAN = 16
NCORR = 4


TABLE_DESC = {
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
        "valueType": "DCOMPLEX",
    },
    "FLAG": {
        "_c_order": True,
        "comment": "FLAG column",
        "dataManagerGroup": "DATA_GROUP",
        "dataManagerType": "TiledColumnStMan",
        "keywords": {},
        "maxlen": 0,
        "ndim": 2,
        "option": 0,
        "shape": [NCHAN, NCORR],
        "valueType": "BOOLEAN",
    },
}

DMINFO = {
    "*1": {
        "NAME": "DATA_GROUP",
        "TYPE": "TiledColumnStMan",
        "SPEC": {"DEFAULTTILESHAPE": [NCORR, NCHAN, 2]},
        "COLUMNS": ["DATA", "FLAG"],
    }
}


def sanity():
    """Sanity check an arcae install"""
    # Generate test data
    ant1, ant2 = np.triu_indices(NANT, 1)
    time = np.arange(ant1.size, dtype=np.float64)
    data = np.arange(time.size * NCHAN * NCORR)
    data = data.reshape(time.size, NCHAN, NCORR)
    data = data + data * 1j
    flag = np.random.randint(0, 2, size=(time.size, NCHAN, NCORR))
    names = np.array([f"name-{i}" for i in range(time.size)])

    with ExitStack() as stack:
        # Write test data
        dir = stack.enter_context(TemporaryDirectory(prefix="arcae-sanity"))
        ms = os.path.join(dir, "sanity.ms")
        T = stack.enter_context(
            Table.ms_from_descriptor(ms, "MAIN", table_desc=TABLE_DESC, dminfo=DMINFO)
        )
        T.addrows(time.size)
        T.putcol("TIME", time)
        T.putcol("ANTENNA1", ant1)
        T.putcol("ANTENNA2", ant2)
        T.putcol("DATA", data)
        T.putcol("FLAG", flag)
        T.putcol("NAMES", names)

        # Read and check equality
        assert_array_equal(T.getcol("TIME"), time)
        assert_array_equal(T.getcol("ANTENNA1"), ant1)
        assert_array_equal(T.getcol("ANTENNA2"), ant2)
        assert_array_equal(T.getcol("DATA"), data)
        assert_array_equal(T.getcol("FLAG"), flag)
        assert_array_equal(T.getcol("NAMES"), names)
