"""Compare snapshots of arcae builds with benchmarks/instance_balance.py.

    python benchmarks/compare_snapshots.py --path T.table A B [-- BENCHMARK ARGS]

Timings on a busy machine drift between runs, so rounds alternate the
snapshots, each in a fresh process and in a rotating order, and the fastest
wall time of each case is kept. Snapshots are made by ``benchmarks/snapshot.py``.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parent / "instance_balance.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("snapshots", nargs="+")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--path", required=True, help="Table shared by all runs")
    # Arguments after "--" are passed to the benchmark
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    extra = argv[split + 1 :]

    # best[(case, ninstances)][snapshot] = fastest result
    best = {}
    for r in range(args.rounds):
        # Rotate the order, as the first run of a round can be consistently slower
        shift = r % len(args.snapshots)
        for snapshot in args.snapshots[shift:] + args.snapshots[:shift]:
            cmd = [sys.executable, str(BENCHMARK), "--json", "--path", args.path]
            cmd += ["--snapshot", snapshot, *extra]
            print(f"round {r + 1}/{args.rounds}: {snapshot}", file=sys.stderr)
            output = subprocess.run(cmd, check=True, capture_output=True, text=True)
            for result in json.loads(output.stdout):
                key = (result["case"], result["ninstances"])
                current = best.setdefault(key, {}).get(snapshot)
                if current is None or result["wall"] < current["wall"]:
                    best[key][snapshot] = result

    base = args.snapshots[0]
    names = [Path(s).name for s in args.snapshots]
    print(f"{'case':<18} {'n':>2}  " + "  ".join(f"{n:>22}" for n in names))
    for (case, ninstances), results in best.items():
        cells = []
        for snapshot in args.snapshots:
            r = results[snapshot]
            cell = f"{r['wall'] * 1e3:8.1f} ms"
            if snapshot != base:
                cell += f" ({r['wall'] / results[base]['wall']:5.2f}x)"
            if "eff" in r:
                cell += f" eff {r['eff']:4.2f}"
            cells.append(f"{cell:>22}")
        print(f"{case:<18} {ninstances:>2}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
