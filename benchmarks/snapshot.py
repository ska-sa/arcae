"""Snapshot the current arcae build, so that builds can be benchmarked side by side.

    python benchmarks/snapshot.py DIR

Copies the ``src/arcae`` package and the built extension into ``DIR/arcae``.
Pass ``--snapshot DIR`` to a benchmark to import arcae from the snapshot.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dir", type=Path)
    args = parser.parse_args()

    extensions = list(
        (ROOT / "build" / "src" / "arcae" / "lib").glob("arrow_tables*.so")
    )
    if len(extensions) != 1:
        raise FileNotFoundError(f"Expected one built extension, found {extensions}")

    package = args.dir / "arcae"
    if package.exists():
        shutil.rmtree(package)
    shutil.copytree(
        ROOT / "src" / "arcae",
        package,
        ignore=shutil.ignore_patterns("__pycache__", "tests", "*.so"),
    )
    shutil.copy2(extensions[0], package / "lib")
    print(f"Snapshot of {extensions[0].name} in {package}")


if __name__ == "__main__":
    main()
