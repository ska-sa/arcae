import gc
import weakref

import numpy as np
import pytest
from numpy.testing import assert_equal

from arcae.lib.arrow_tables import merge_np_partitions


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("n, chunk", [(100, 33), (125, 42)])
def test_merge_np_partitions(seed, n, chunk):
    rng = np.random.default_rng(seed=seed)
    ddid = rng.integers(0, 10, n)
    field_id = rng.integers(0, 10, n)
    time = rng.random(n)
    interval = rng.random(n)
    ant1 = rng.integers(0, 10, n)
    ant2 = rng.integers(0, 10, n)
    row = rng.integers(0, 10, n)

    partitions = [
        {
            "DATA_DESC_ID": ddid[start : start + chunk],
            "FIELD_ID": field_id[start : start + chunk],
            "TIME": time[start : start + chunk],
            "ANTENNA1": ant1[start : start + chunk],
            "ANTENNA2": ant2[start : start + chunk],
            "INTERVAL": interval[start : start + chunk],
            "ROW": row[start : start + chunk],
        }
        for start in range(0, n, chunk)
    ]

    sorts = [np.lexsort(tuple(reversed(p.values()))) for p in partitions]
    partitions = [{k: v[s] for k, v in p.items()} for p, s in zip(partitions, sorts)]

    expected = {
        "DATA_DESC_ID": ddid,
        "FIELD_ID": field_id,
        "TIME": time,
        "ANTENNA1": ant1,
        "ANTENNA2": ant2,
        "INTERVAL": interval,
        "ROW": row,
    }

    sort = np.lexsort(tuple(reversed(expected.values())))
    expected = {k: v[sort] for k, v in expected.items()}
    merged = merge_np_partitions(partitions)
    assert_equal(merged, expected)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("n, chunk", [(100, 33)])
def test_merge_fail_1d(seed, n, chunk):
    rng = np.random.default_rng(seed=seed)
    ddid = rng.integers(0, 10, 100)
    field_id = rng.integers(0, 10, (100, 4))

    partitions = [
        {
            "DATA_DESC_ID": ddid[start : start + chunk],
            "FIELD_ID": field_id[start : start + chunk],
        }
        for start in range(0, n, chunk)
    ]

    with pytest.raises(ValueError, match="Array must be 1-dimensional"):
        merge_np_partitions(partitions)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("n, chunk", [(100, 33)])
def test_merge_fail_array_length(seed, n, chunk):
    rng = np.random.default_rng(seed=seed)
    ddid = rng.integers(0, 10, 100)
    field_id = rng.integers(0, 10, 50)

    partitions = [
        {
            "DATA_DESC_ID": ddid[start : start + chunk],
            "FIELD_ID": field_id[start : start + chunk],
        }
        for start in range(0, n, chunk)
    ]

    with pytest.raises(ValueError, match="Array lengths do not match"):
        merge_np_partitions(partitions)


def test_merge_np_partitions_no_leak():
    """Merged arrays should be freed once no longer referenced"""
    a = np.sort(np.random.default_rng(42).random(100))
    partitions = [
        {"A": a[s : s + 25].copy(), "B": a[s : s + 25].copy()}
        for s in range(0, 100, 25)
    ]
    merged = merge_np_partitions(partitions)
    assert_equal(merged["A"], a)
    refs = [weakref.ref(v) for v in merged.values()]
    del merged
    gc.collect()
    assert all(r() is None for r in refs)


def test_merge_np_partitions_non_contiguous():
    """Strided inputs are merged using their logical values"""
    a = np.arange(8.0)
    b = a + 0.5
    merged = merge_np_partitions([{"A": a[::2]}, {"A": b[::2]}])
    assert_equal(merged["A"], [0.0, 0.5, 2.0, 2.5, 4.0, 4.5, 6.0, 6.5])


def test_merge_np_partitions_non_native_byteorder():
    """Byte-swapped inputs are merged using their logical values"""
    swapped = np.dtype(np.int64).newbyteorder("S")
    p1 = {"A": np.array([1, 3, 5], dtype=swapped)}
    p2 = {"A": np.array([2, 4, 6], dtype=swapped)}
    merged = merge_np_partitions([p1, p2])
    assert_equal(merged["A"], [1, 2, 3, 4, 5, 6])


def test_merge_np_partitions_empty_partition():
    p1 = {"A": np.array([1, 3], dtype=np.int32)}
    p2 = {"A": np.array([], dtype=np.int32)}
    p3 = {"A": np.array([2], dtype=np.int32)}
    merged = merge_np_partitions([p1, p2, p3])
    assert_equal(merged["A"], [1, 2, 3])


def test_merge_np_partitions_key_order():
    """Arrays are matched across partitions by key, not position"""
    p1 = {"A": np.array([0, 1]), "B": np.array([0.0, 1.0])}
    p2 = {"B": np.array([0.5, 2.0]), "A": np.array([0, 1])}
    merged = merge_np_partitions([p1, p2])
    assert_equal(merged["A"], [0, 0, 1, 1])
    assert_equal(merged["B"], [0.0, 0.5, 1.0, 2.0])


def test_merge_fail_mismatched_keys():
    p1 = {"A": np.array([0, 1])}
    p2 = {"B": np.array([0, 1])}
    with pytest.raises(ValueError, match="Partitions must have the same keys"):
        merge_np_partitions([p1, p2])


def test_merge_fail_dtype_mismatch():
    p1 = {"A": np.array([0, 1], dtype=np.int32)}
    p2 = {"A": np.array([0, 1], dtype=np.int64)}
    with pytest.raises(ValueError, match="Array dtypes must match"):
        merge_np_partitions([p1, p2])


def test_merge_fail_unsupported_dtype():
    """Unsupported dtypes are rejected even for a single partition"""
    with pytest.raises(ValueError, match="Unsupported array type"):
        merge_np_partitions([{"A": np.array([0, 1], dtype=np.int16)}])
