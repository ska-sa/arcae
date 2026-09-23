from concurrent.futures import ThreadPoolExecutor

from numpy.testing import assert_array_equal

import arcae


def test_safe_multithreaded_writes():
    """Assert that this version of arcae does not
    support multithreaded writes"""
    assert arcae.safe_multithreaded_writes()


def test_writes_succeeds_ninstances_1(column_case_table):
    """Test that writing when ninstances==1 succeeds"""
    with arcae.table(column_case_table, ninstances=1, readonly=False) as T:
        data = T.getcol("FIXED")
        T.putcol("FIXED", data + 1)
        assert_array_equal(T.getcol("FIXED"), data + 1)


def test_writes_fail_ninstances_2(column_case_table):
    """Test that writing when ninstances > 1 succeeds"""
    with arcae.table(column_case_table, ninstances=2, readonly=False) as T:
        data = T.getcol("FIXED")
        T.putcol("FIXED", data + 1)
        assert_array_equal(T.getcol("FIXED"), data + 1)


def test_concurrent_keyword_writes(column_case_table):
    """Keyword writes are serialised through the single writer instance,
    so concurrent writers from multiple threads all land"""
    with arcae.table(column_case_table, ninstances=4, readonly=False) as T:
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda i: T.putkeywords({f"kw{i}": i}), range(8)))

        keywords = T.getkeywords()
        for i in range(8):
            assert keywords[f"kw{i}"] == i


def test_concurrent_column_keyword_writes(column_case_table):
    """Column keyword writes are likewise serialised"""
    with arcae.table(column_case_table, ninstances=4, readonly=False) as T:
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda i: T.putcolkeywords("FIXED", {f"kw{i}": i}), range(8)))

        keywords = T.getcolkeywords("FIXED")
        for i in range(8):
            assert keywords[f"kw{i}"] == i
