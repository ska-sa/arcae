"""How evenly arcae spreads I/O over table instances.

    python benchmarks/instance_balance.py [--ninstances 1 2 4 8] [--case CASE]

Each case is timed ``repeat`` times per number of instances, and the
fastest run is reported along with the per-instance statistics
(``Table._instance_stats()``) accumulated during it:

    wall      wall-clock time of the case
    eff       busy time / (ninstances * wall). 1.0 means every instance
              was busy for the whole case
    busy      max / mean busy time over the instances. 1.0 is a perfect balance
    bytes     max / mean bytes read over the instances
    tasks     tasks run by each instance
    wait      mean time a task waited in its instance's queue

``--snapshot DIR`` imports arcae from a snapshot made by
``benchmarks/snapshot.py``, so that builds can be compared
by ``benchmarks/compare_snapshots.py``. Snapshots that predate
``_instance_stats`` only report wall-clock times.

The table is created once (``--path``) and is then read from the page cache,
so cases measure CPU-bound casacore decoding rather than disk throughput.
``--cold`` instead evicts the table from the page cache, and reopens it,
before every run.
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

NCHAN = 64
NCORR = 4


def import_arcae(snapshot):
    if snapshot:
        # The editable install redirects ``import arcae`` to the source tree
        sys.meta_path[:] = [
            f
            for f in sys.meta_path
            if type(f).__name__ not in {"_Finder", "ScikitBuildRedirectingFinder"}
        ]
        sys.path.insert(0, str(Path(snapshot).resolve()))

    import arcae

    return arcae


def evict(path):
    """Evict a table's files from the page cache. Doesn't need root"""
    os.sync()
    for p in Path(path).rglob("*"):
        if p.is_file():
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def create(path, nrow):
    from casacore import tables as ct

    def scalar(name, value):
        return ct.makescacoldesc(name, value, datamanagergroup="SSM")

    def tiled(name, value, shape):
        return ct.makearrcoldesc(
            name,
            value,
            shape=shape,
            datamanagertype="TiledColumnStMan",
            datamanagergroup=name,
        )

    desc = ct.maketabdesc(
        [
            scalar("TIME", 0.0),
            scalar("ANTENNA1", 0),
            scalar("ANTENNA2", 0),
            tiled("UVW", 0.0, [3]),
            tiled("WEIGHT", 0.0, [NCORR]),
            tiled("DATA", 0j, [NCHAN, NCORR]),
            tiled("FLAG", False, [NCHAN, NCORR]),
            ct.makearrcoldesc(
                "RAGGED",
                0j,
                ndim=2,
                datamanagertype="TiledShapeStMan",
                datamanagergroup="RAGGED",
            ),
        ]
    )
    rng = np.random.default_rng(42)

    with ct.table(str(path), desc, nrow=nrow, ack=False) as T:
        T.putcol("TIME", np.arange(nrow, dtype=np.float64))
        T.putcol("ANTENNA1", np.arange(nrow, dtype=np.int32) % 64)
        T.putcol("ANTENNA2", np.arange(nrow, dtype=np.int32) % 63)
        T.putcol("UVW", rng.random((nrow, 3)))
        T.putcol("WEIGHT", rng.random((nrow, NCORR)).astype(np.float32))

        # Write the large columns in blocks to bound memory use
        for start in range(0, nrow, 10_000):
            n = min(10_000, nrow - start)
            data = rng.random((n, NCHAN, NCORR)) + 1j * rng.random((n, NCHAN, NCORR))
            T.putcol("DATA", data.astype(np.complex64), startrow=start, nrow=n)
            flag = rng.integers(0, 2, (n, NCHAN, NCORR)).astype(bool)
            T.putcol("FLAG", flag, startrow=start, nrow=n)

        # Two shapes, alternating in runs of 16 rows
        small = np.ones((NCHAN // 2, NCORR), np.complex64)
        large = np.ones((NCHAN, NCORR), np.complex64)
        for r in range(nrow):
            T.putcell("RAGGED", r, small if (r // 16) % 2 == 0 else large)


# Each case takes an open table and returns a dict of extra results,
# which are reported alongside the scheduling statistics


def big_column(T, args):
    """A single read of the largest column"""
    T.getcol("DATA")
    return {}


def gapped_rows(T, args):
    """A single read of the largest column, skipping every 1000th row"""
    nrow = T.nrow()
    T.getcol("DATA", index=(np.setdiff1d(np.arange(nrow), np.arange(0, nrow, 1000)),))
    return {}


def ragged(T, args):
    """A single read of a column whose shape varies by row"""
    T.to_arrow(columns="RAGGED")
    return {}


def to_arrow(T, args):
    """All columns of mixed sizes, converted to arrow"""
    T.to_arrow()
    return {}


def threads_rows(T, args):
    """Python threads reading disjoint row ranges of the largest column"""
    nrow = T.nrow()
    step = -(-nrow // args.threads)
    starts = range(0, nrow, step)
    with ThreadPoolExecutor(args.threads) as pool:
        futures = [
            pool.submit(T.getcol, "DATA", index=(slice(s, min(s + step, nrow)),))
            for s in starts
        ]
        for f in futures:
            f.result()
    return {}


def concurrent_requests(T, args):
    """Python threads issuing independent reads of consecutive row ranges,
    covering the largest column"""
    nrow = T.nrow()
    step = args.request_rows
    with ThreadPoolExecutor(args.threads) as pool:
        futures = [
            pool.submit(T.getcol, "DATA", index=(slice(s, min(s + step, nrow)),))
            for s in range(0, nrow, step)
        ]
        for f in futures:
            f.result()
    return {}


def threads_columns(T, args):
    """Python threads each reading a different fixed shape column"""
    columns = [c for c in T.columns() if c != "RAGGED"]
    with ThreadPoolExecutor(len(columns)) as pool:
        for f in [pool.submit(T.getcol, c) for c in columns]:
            f.result()
    return {}


def small_during_big(T, args):
    """Latency of small calls made while a large read is in progress"""
    done = threading.Event()

    def reader():
        try:
            for _ in range(3):
                T.getcol("DATA")
        finally:
            done.set()

    thread = threading.Thread(target=reader)
    thread.start()
    latencies = []
    while not done.is_set():
        t0 = time.perf_counter()
        T.getcol("TIME", index=(slice(0, 1),))
        latencies.append(time.perf_counter() - t0)
    thread.join()
    latencies.sort()
    return {
        "small_calls": len(latencies),
        "small_p50_ms": 1e3 * latencies[len(latencies) // 2],
        "small_max_ms": 1e3 * latencies[-1],
    }


CASES = {
    f.__name__: f
    for f in [
        big_column,
        gapped_rows,
        ragged,
        to_arrow,
        threads_rows,
        concurrent_requests,
        threads_columns,
        small_during_big,
    ]
}

STAT_KEYS = ["submitted", "completed", "wait_ns", "busy_ns", "io_bytes"]


def stats_delta(before, after):
    return [{k: a[k] - b[k] for k in STAT_KEYS} for b, a in zip(before, after)]


def ratio_to_mean(values):
    mean = statistics.fmean(values)
    return max(values) / mean if mean > 0 else float("nan")


def run_case(T, fn, args):
    has_stats = hasattr(T, "_instance_stats")
    before = T._instance_stats() if has_stats else None
    t0 = time.perf_counter()
    extra = fn(T, args)
    wall = time.perf_counter() - t0
    result = {"wall": wall, **extra}

    if has_stats:
        delta = stats_delta(before, T._instance_stats())
        busy = [d["busy_ns"] * 1e-9 for d in delta]
        tasks = [d["completed"] for d in delta]
        result.update(
            eff=sum(busy) / (len(busy) * wall),
            busy=ratio_to_mean(busy),
            bytes=ratio_to_mean([d["io_bytes"] for d in delta]),
            tasks=tasks,
            wait_ms=1e3 * sum(d["wait_ns"] for d in delta) * 1e-9 / max(1, sum(tasks)),
            instances=delta,
        )

    return result


def format_result(case, ninstances, r):
    line = f"{case:<18} n={ninstances:<2} wall {r['wall'] * 1e3:8.1f} ms"
    if "eff" in r:
        line += (
            f"  eff {r['eff']:5.2f}  busy {r['busy']:5.2f}  bytes {r['bytes']:5.2f}"
            f"  wait {r['wait_ms']:7.2f} ms  tasks {r['tasks']}"
        )
    if "small_calls" in r:
        line += (
            f"  small p50 {r['small_p50_ms']:6.2f} ms"
            f" max {r['small_max_ms']:7.2f} ms ({r['small_calls']} calls)"
        )
    return line


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nrow", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--ninstances", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--case", action="append", choices=list(CASES))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--request-rows",
        type=int,
        default=10_000,
        help="Rows read by each request of concurrent_requests",
    )
    parser.add_argument("--snapshot", help="Import arcae from this snapshot")
    parser.add_argument("--json", action="store_true", help="Print results as JSON")
    parser.add_argument(
        "--cold",
        action="store_true",
        help="Evict the table from the page cache and reopen it before each run",
    )
    parser.add_argument(
        "--path",
        help="Table to create if missing, and reuse. Defaults to a temporary table",
    )
    args = parser.parse_args()
    cases = args.case or list(CASES)
    arcae = import_arcae(args.snapshot)
    results = []

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(args.path) if args.path else Path(tmp) / "balance.table"
        if not path.exists():
            create(path, args.nrow)

        for ninstances in args.ninstances:
            with arcae.table(str(path), ninstances=ninstances) as T:
                if T.nrow() != args.nrow:
                    raise ValueError(f"{path} has {T.nrow()} rows, not {args.nrow}")
                if "RAGGED" not in T.columns():
                    raise ValueError(f"{path} predates the RAGGED column: delete it")

                for case in cases:
                    if args.cold:
                        runs = []
                        for _ in range(args.repeat):
                            evict(path)
                            with arcae.table(str(path), ninstances=ninstances) as C:
                                runs.append(run_case(C, CASES[case], args))
                    else:
                        # Warm up, then keep the fastest run
                        CASES[case](T, args)
                        runs = [
                            run_case(T, CASES[case], args) for _ in range(args.repeat)
                        ]
                    best = min(runs, key=lambda r: r["wall"])
                    results.append({"case": case, "ninstances": ninstances, **best})
                    if not args.json:
                        print(format_result(case, ninstances, best), flush=True)

    if args.json:
        json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
