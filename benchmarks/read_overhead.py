"""Single-threaded read overhead of arcae against python-casacore.

Covers the per-call and per-row cases reported in
https://github.com/casangi/xradio/issues/637

    python benchmarks/read_overhead.py [--nrow N] [--repeat R] [--case CASE]

Tables are created in a temporary directory, unless ``--path`` is given.
Timings are the best of ``repeat`` warm runs.
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
from casacore import tables as ct

import arcae
from arcae.lib.arrow_tables import Table

SHAPE = (11, 4)


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
            col("FLAG_FIXED", False, 2, SHAPE, "TiledColumnStMan", "TCM"),
            col("FLAG_TSM", False, 2, None, "TiledShapeStMan", "TSM"),
            col("DIRECTION", 0.0, 2, None, "StandardStMan", "SSM"),
            col("RAGGED_TSM", 0.0, 2, None, "TiledShapeStMan", "RTSM"),
            col("RAGGED_SSM", 0.0, 2, None, "StandardStMan", "SSM"),
            col("MISSING_TSM", 0.0, 2, None, "TiledShapeStMan", "MTSM"),
        ]
    )
    with ct.table(str(path), desc, nrow=nrow, ack=False) as T:
        rng = np.random.default_rng(42)
        T.putcol("TIME", np.arange(nrow, dtype=np.float64))
        T.putcol("ANTENNA1", np.arange(nrow, dtype=np.int32) % 64)
        flags = rng.integers(0, 2, (nrow,) + SHAPE).astype(bool)
        T.putcol("FLAG_FIXED", flags)
        T.putcol("FLAG_TSM", flags)
        T.putcol("DIRECTION", rng.random((nrow, 1, 2)))

        # Two shapes, alternating in runs of 16 rows
        small, large = rng.random((4, 2)), rng.random((8, 2))
        for r in range(nrow):
            T.putcell("RAGGED_TSM", r, small if (r // 16) % 2 == 0 else large)
            T.putcell("RAGGED_SSM", r, small if (r // 16) % 2 == 0 else large)
            # Every 100th row is undefined
            if r % 100 != 0:
                T.putcell("MISSING_TSM", r, small)


def best(fn, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


CASES = {
    # name: (column, row selection, python-casacore reader)
    "scalar_double": ("TIME", None, "getcol"),
    "scalar_int": ("ANTENNA1", None, "getcol"),
    "flag_fixed": ("FLAG_FIXED", None, "getcol"),
    "flag_tsm": ("FLAG_TSM", None, "getcol"),
    "direction_ssm": ("DIRECTION", None, "getcol"),
    "ragged_tsm": ("RAGGED_TSM", None, "getvarcol"),
    "ragged_ssm": ("RAGGED_SSM", None, "getvarcol"),
    "flag_tsm_strided": ("FLAG_TSM", "strided", "getcolnp"),
    "scalar_double_strided": ("TIME", "strided", "getcolnp"),
    # Every second channel, which is not contiguous in memory
    "flag_fixed_chan_sel": ("FLAG_FIXED", "channels", "getcolslice"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nrow", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--case", action="append", choices=list(CASES))
    parser.add_argument("--calls", type=int, default=2_000)
    parser.add_argument("--only-arcae", action="store_true")
    parser.add_argument("--ninstances", type=int, default=1)
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
        strided = np.arange(0, args.nrow, 2)

        with ct.table(str(path), ack=False) as C, arcae.table(
            str(path), ninstances=args.ninstances
        ) as A:
            print(f"nrow={args.nrow} repeat={args.repeat} ninstances={args.ninstances}")

            # Per-call overhead
            calls = args.calls
            t_a = best(lambda: [A.nrow() for _ in range(calls)], args.repeat)
            t_c = best(lambda: [C.nrows() for _ in range(calls)], args.repeat)
            print(
                f"{'nrow() per call':<24} arcae {t_a / calls * 1e6:8.2f} us  "
                f"casacore {t_c / calls * 1e6:8.2f} us  ratio {t_a / t_c:6.2f}"
            )
            t_a = best(
                lambda: [A.getcol("TIME", index=(slice(0, 1),)) for _ in range(calls)],
                args.repeat,
            )
            t_c = best(
                lambda: [C.getcol("TIME", startrow=0, nrow=1) for _ in range(calls)],
                args.repeat,
            )
            print(
                f"{'1-row getcol per call':<24} arcae {t_a / calls * 1e6:8.2f} us  "
                f"casacore {t_c / calls * 1e6:8.2f} us  ratio {t_a / t_c:6.2f}"
            )

            for case in cases:
                column, sel, reader = CASES[case]
                if sel == "strided":
                    index = (strided,)
                    nrows = len(strided)

                    def casa_read(column=column, reader=reader):
                        # Reads the rows via a reference table
                        with C.selectrows(strided) as S:
                            return S.getcol(column)
                elif sel == "channels":
                    index = (None, np.arange(0, SHAPE[0], 2))
                    nrows = args.nrow

                    def casa_read(column=column, reader=reader):
                        trc = [SHAPE[0] - 1, SHAPE[1] - 1]
                        return C.getcolslice(column, [0, 0], trc, [2, 1])
                else:
                    index = None
                    nrows = args.nrow

                    def casa_read(column=column, reader=reader):
                        return getattr(C, reader)(column)

                if reader == "getvarcol":
                    # Ragged columns only convert to arrow
                    def arcae_read(column=column, index=index):
                        return A.to_arrow(index=index, columns=column)
                else:

                    def arcae_read(column=column, index=index):
                        return A.getcol(column, index=index)

                t_a = best(arcae_read, args.repeat)
                line = (
                    f"{case:<24} arcae {t_a * 1e3:8.2f} ms "
                    f"({t_a / nrows * 1e9:7.1f} ns/row)"
                )
                if not args.only_arcae:
                    t_c = best(casa_read, args.repeat)
                    line += f"  casacore {t_c * 1e3:8.2f} ms  ratio {t_a / t_c:6.2f}"
                print(line)


if __name__ == "__main__":
    main()
