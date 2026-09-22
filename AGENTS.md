# arcae: Arrow C++ and Python Bindings for casacore

[ratt-ru/arcae](https://github.com/ratt-ru/arcae) provides Apache Arrow bindings for [casacore](https://github.com/casacore/casacore) tables (the CASA Table Data System, CTDS). It implements a focused subset of [python-casacore](https://github.com/casacore/python-casacore) that **drops the GIL** and offers **safe multi-threaded access** to CASA tables, so MeasurementSet data
 can be read concurrently and exported to cloud-native formats (Arrow, Parquet, Zarr).
 Concurrency is the defining concern of the project: most of the design exists to
 work around the CTDS being neither thread-safe nor GIL-friendly.

### Core Capabilities

- **Read** CASA tables (and TAQL query results / reference tables) into
  `pyarrow.Table`s via `arcae.table(...).to_arrow(...)`, with row/column index
  selection.
- **Column I/O** through `getcol`/`putcol`, including the buffer reinterpretation
  needed to round-trip Arrow's nested/complex layouts back to NumPy.
- **Write** paths: `putcol`, `addrows`, `addcols`, and creating MeasurementSets
  from a descriptor.
- **Multi-threaded fan-out**: a single table can be opened as `ninstances`
  independent casacore instances; reads are multiplexed across them while the GIL
  is released in the C++ layer.
- **Coexistence** with an independent `python-casacore` in the same process,
  because casacore is statically linked into arcae's extension module.

### Architecture Notes

- **Two layers.** A thin Python/Cython surface in `src/arcae/` (entry point
  `arcae.table(...)` → `Table`; the Cython bridge is
  `src/arcae/lib/arrow_tables.pyx`) over a C++ core in `cpp/arcae/` that opens
  tables, partitions and multiplexes I/O, and marshals CASA data into
  `arrow::Table`s.
- **Patched, vendored casacore.** casacore is built from a vcpkg overlay port
  (`vcpkg/overlay-ports/casacore/`) carrying arcae-specific patches — most
  importantly a `thread_local` table cache.
- **Write support** The `write-support` branch contains a
  multi-reader/single-writer `FileLocker` — that make per-thread
  table instances viable. This will be merged into `main` after further testing.
- **Static monolith.** casacore is statically linked into a single self-contained
  `.so`, which is what lets arcae import alongside `python-casacore` without
  symbol clashes.
- **Tests.** C++ via gtest in `cpp/tests/` (run with `ctest`); Python via pytest
  in `src/arcae/tests/`.
- Before changing anything in the locking / threading / instance machinery, read
  the design doc:
  [`design/design.md`](design/design.md).

## Tooling

Create a virtual environment containing the build system dependencies in `pyproject.toml`.
The following commands are available in the venv.

- Set up cmake for the C++ code: `cmake -S . -B build`.
- Build the C++ code and tests: `cmake --build build`.
- Run the C++ test cases: `ctest --test-dir build --verbose`.
- Build the Python extension: `pip install --no-build-isolation --config-settings=editable.rebuild=true -Cbuild-dir=build -ve.`.
