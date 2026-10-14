"""Single-threaded write overhead of arcae against python-casacore.

The write counterpart of ``read_overhead.py``

    python benchmarks/write_overhead.py [--nrow N] [--repeat R] [--case CASE]

Tables are created in a temporary directory, unless ``--path`` is given.
Timings are the best of ``repeat`` warm runs. arcae and python-casacore
write in turn, as two writers should not have the table open at once.
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
from casacore import tables as ct

import arcae

SHAPE = (11, 4)
DATA_SHAPE = (16, 4)


def col(name, value, ndim=0, shape=None, dm="StandardStMan", group=None):
    if ndim == 0:
        return ct.makescacoldesc(
            name, value, datamanagertype=dm, datamanagergroup=group
        )
    kw = {"ndim": ndim, "datamanagertype": dm, "datamanagergroup": group}
    if shape is not None:
        kw["shape"] = shape
    return ct.makearrcoldesc(name, value, **kw)


def create(path, nrow):
    desc = ct.maketabdesc(
        [
            col("TIME", 0.0, group="SSM"),
            col("ANTENNA1", 0, group="SSM"),
            col("NAME", "", group="SSM"),
            col("FLAG_FIXED", False, 2, SHAPE, "TiledColumnStMan", "TCM"),
            col("FLAG_TSM", False, 2, None, "TiledShapeStMan", "TSM"),
            col("FLAG_SSM", False, 2, None, "StandardStMan", "SSM"),
            col("DATA_TSM", 0j, 2, None, "TiledShapeStMan", "DTSM"),
            col("DIRECTION", 0.0, 2, None, "StandardStMan", "SSM"),
        ]
    )
    with ct.table(str(path), desc, nrow=nrow, ack=False) as T:
        # Define the cells of the variably shaped columns
        T.putcol("FLAG_TSM", np.zeros((nrow,) + SHAPE, bool))
        T.putcol("FLAG_SSM", np.zeros((nrow,) + SHAPE, bool))
        T.putcol("DATA_TSM", np.zeros((nrow,) + DATA_SHAPE, np.complex64))
        T.putcol("DIRECTION", np.zeros((nrow, 1, 2)))


def best(fn, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


CASES = {
    # name: (column, row selection)
    "scalar_double": ("TIME", None),
    "scalar_int": ("ANTENNA1", None),
    "scalar_string": ("NAME", None),
    "flag_fixed": ("FLAG_FIXED", None),
    "flag_tsm": ("FLAG_TSM", None),
    "flag_ssm": ("FLAG_SSM", None),
    "data_tsm": ("DATA_TSM", None),
    "direction_ssm": ("DIRECTION", None),
    "flag_tsm_strided": ("FLAG_TSM", "strided"),
    "flag_ssm_strided": ("FLAG_SSM", "strided"),
    "scalar_double_strided": ("TIME", "strided"),
    # Every second channel, which is not contiguous on disk
    "flag_fixed_chan_sel": ("FLAG_FIXED", "channels"),
    "data_tsm_chan_sel": ("DATA_TSM", "channels"),
}


def make_data(nrow):
    rng = np.random.default_rng(42)
    return {
        "TIME": np.arange(nrow, dtype=np.float64),
        "ANTENNA1": np.arange(nrow, dtype=np.int32) % 64,
        # python-casacore segfaults on object arrays of strings
        "NAME": np.array([f"source-{i % 100}" for i in range(nrow)]),
        "FLAG_FIXED": rng.integers(0, 2, (nrow,) + SHAPE).astype(bool),
        "FLAG_TSM": rng.integers(0, 2, (nrow,) + SHAPE).astype(bool),
        "FLAG_SSM": rng.integers(0, 2, (nrow,) + SHAPE).astype(bool),
        "DATA_TSM": (
            rng.random((nrow,) + DATA_SHAPE) + 1j * rng.random((nrow,) + DATA_SHAPE)
        ).astype(np.complex64),
        "DIRECTION": rng.random((nrow, 1, 2)),
    }


def case_writers(data, nrow):
    """Returns {case: (nrows, arcae writer, python-casacore writer)}"""
    strided = np.arange(0, nrow, 2)
    writers = {}

    for case, (column, sel) in CASES.items():
        values = data[column]
        if sel == "strided":
            values = values[strided]
            index = (strided,)

            def casa_write(C, column=column, values=values):
                # Writes the rows via a reference table
                with C.selectrows(strided) as S:
                    S.putcol(column, values)
        elif sel == "channels":
            channels = np.arange(0, values.shape[1], 2)
            values = np.ascontiguousarray(values[:, channels])
            index = (None, channels)

            def casa_write(C, column=column, values=values):
                trc = [values.shape[1] * 2 - 2, values.shape[2] - 1]
                C.putcolslice(column, values, [0, 0], trc, [2, 1])
        else:
            index = None

            def casa_write(C, column=column, values=values):
                C.putcol(column, values)

        def arcae_write(A, column=column, values=values, index=index):
            A.putcol(column, values, index=index)

        writers[case] = (len(values), arcae_write, casa_write)

    return writers


def per_call(T, calls, repeat, write):
    """Per-call time of single row writes"""
    return best(lambda: [write(T) for _ in range(calls)], repeat) / calls


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nrow", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--case", action="append", choices=list(CASES))
    parser.add_argument("--calls", type=int, default=2_000)
    parser.add_argument("--only-arcae", action="store_true")
    parser.add_argument(
        "--path",
        help="Table to create if missing, and reuse. Defaults to a temporary table",
    )
    args = parser.parse_args()
    cases = args.case or list(CASES)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(args.path) if args.path else Path(tmp) / "bench.table"
        if not path.exists():
            create(path, args.nrow)

        with ct.table(str(path), ack=False) as C:
            if C.nrows() != args.nrow:
                raise ValueError(f"{path} has {C.nrows()} rows, not {args.nrow}")

        writers = case_writers(make_data(args.nrow), args.nrow)
        one_row = np.zeros(1, np.float64)
        timings = {}

        with arcae.table(str(path), readonly=False) as A:
            timings["arcae", "call"] = per_call(
                A,
                args.calls,
                args.repeat,
                lambda A: A.putcol("TIME", one_row, index=(slice(0, 1),)),
            )
            for case in cases:
                timings["arcae", case] = best(lambda: writers[case][1](A), args.repeat)

        if not args.only_arcae:
            with ct.table(str(path), readonly=False, ack=False) as C:
                timings["casa", "call"] = per_call(
                    C,
                    args.calls,
                    args.repeat,
                    lambda C: C.putcol("TIME", one_row, startrow=0, nrow=1),
                )
                for case in cases:
                    timings["casa", case] = best(
                        lambda: writers[case][2](C), args.repeat
                    )

    print(f"nrow={args.nrow} repeat={args.repeat}")
    line = (
        f"{'1-row putcol per call':<24} arcae {timings['arcae', 'call'] * 1e6:8.2f} us"
    )
    if not args.only_arcae:
        t_a, t_c = timings["arcae", "call"], timings["casa", "call"]
        line += f"  casacore {t_c * 1e6:8.2f} us  ratio {t_a / t_c:6.2f}"
    print(line)

    for case in cases:
        nrows = writers[case][0]
        t_a = timings["arcae", case]
        line = f"{case:<24} arcae {t_a * 1e3:8.2f} ms ({t_a / nrows * 1e9:7.1f} ns/row)"
        if not args.only_arcae:
            t_c = timings["casa", case]
            line += f"  casacore {t_c * 1e3:8.2f} ms  ratio {t_a / t_c:6.2f}"
        print(line)


if __name__ == "__main__":
    main()
