# arcae Design

How arcae provides GIL-free, multi-threaded access to CASA tables and exposes
their contents as Apache Arrow tables. Paths below are relative to the
[ratt-ru/arcae](https://github.com/ratt-ru/arcae) repository root.
This document only applies to the `0.4.0-dev` branch which contains
true multi-threaded write support via the `FileLocker` changes. The `main` branch does not modify `FileLocker` and is only safe for
multi-threaded reads. The exception is §3, which describes how `main` spreads
I/O over instances as of [#260](https://github.com/ska-sa/arcae/pull/260),
pending its port to `0.4.0-dev`.

## Why arcae exists

casacore and its `python-casacore` bindings provide access to the CASA Table
Data System (CTDS) and the MeasurementSets built on it. Two CTDS limitations
motivate arcae:

- **Table access from multiple threads is unsafe**
  ([casacore#1038](https://github.com/casacore/casacore/issues/1038),
  [casacore#1163](https://github.com/casacore/casacore/issues/1163)).
- **`python-casacore` does not release the GIL**
  ([python-casacore#209](https://github.com/casacore/python-casacore/pull/209)).

Rather than rewrite the CTDS, arcae translates CASA table data into Apache Arrow
— a language-portable, columnar in-memory format — entirely within the C++ layer,
where the GIL is not held. Once data is in Arrow it converts cheaply to Parquet,
Zarr and other cloud-native formats. arcae deliberately implements only a subset
of `python-casacore`.

The remaining problem is that the CTDS itself is still single-threaded and
unsafe. arcae's design is the set of mechanisms that make concurrent access to it
*safe* and *parallel* anyway. There are five load-bearing pieces, described below.

---

## 1. A `thread_local` table cache (patched casacore)

casacore keeps a process-global cache of open tables, `PlainTable::theirTableCache`,
so that opening the same table twice returns shared in-memory state. That sharing
is exactly what makes concurrent access unsafe: two threads touching "the same"
table touch the same non-thread-safe storage managers.

arcae patches this cache to be **thread-local**
(`vcpkg/overlay-ports/casacore/001-casacore-cmake.patch`):

```cpp
-TableCache PlainTable::theirTableCache;
+thread_local TableCache PlainTable::theirTableCache;
```

Consequence: the *same* table file can be opened independently from multiple
threads, each thread getting its own `Table` / storage-manager state instead of
aliasing a shared cache entry. This is the foundation that lets arcae run several
genuinely independent instances of one table at once.

## 2. `IsolatedTableProxy`: N instances, one thread each

`IsolatedTableProxy` (ITP) — `cpp/arcae/isolated_table_proxy.{h,cc}` — is the
core concurrency primitive. An ITP holds `N` **instances**, where each instance is:

- one `casacore::TableProxy` (the table opened by a user-supplied functor), plus
- one dedicated single-thread Arrow `ThreadPool`.

Because casacore tables are not thread-safe, **each `TableProxy` is confined to
its own pool thread for its entire lifetime** — every operation on that proxy is
submitted to that one thread, so the CTDS only ever sees single-threaded access.
Combined with the `thread_local` cache (§1), the `N` instances are mutually
independent even though they refer to the same table on disk.

`IsolatedTableProxy::Make(functor, ninstances)` builds the ITP by submitting the
open functor to each of the `N` pools and collecting the resulting proxies into a
`ProxyAndPool` vector. `ninstances` is surfaced all the way up to the Python entry
point `arcae.table(filename, ninstances=...)`.

Operations are dispatched onto an instance through three templated entry points:

- `RunAsync(fn, lock_type)` → returns an `arrow::Future`.
- `RunSync(fn, lock_type)` → runs and waits for the result.
- `Then(future, fn, lock_type)` → chains a continuation onto a prior future.

Each dispatch:

1. picks an instance (§3), reserving the functor's estimated I/O on it,
2. submits the functor to that instance's single thread,
3. wraps the call in a `MaybeLockAndFinalise` guard (see §4),
4. fails with `Status::Invalid("TableProxy is closed")` if the instance's table
   has been closed (see below),
5. invokes the functor with the instance's `TableProxy&`,
6. converts any `casacore::AipsError` into an `arrow::Status` so exceptions never
   escape a pool thread.

`Close()` closes each instance's table on its isolation thread, and only marks
the ITP closed once every instance has closed. Tasks dispatched in between pass
`CheckClosed()` and are queued behind the close. casacore answers some queries
on a closed table instead of failing them (`nrow()` returns zero), so step 4
checks the table on the isolation thread itself, which cannot race the close.

Dispatch lambdas capture `weak_from_this()` (not raw `this`); the task re-locks
the weak pointer at start and bails with `Status::Invalid("TableProxy is closed")`
if the ITP has been destroyed, avoiding use-after-free on teardown.

### Spawning dependent proxies

`Spawn(functor)` creates a derived ITP whose proxies are built *from* this ITP's
proxies (e.g. a TAQL query producing a reference table). It mirrors the instance
topology, runs the functor on each source instance under a read lock, and records
a `dependencies_` edge so the parent ITP outlives the child.

## 3. Multiplexing I/O across instances (read fan-out)

### Choosing an instance

Each dispatch carries an estimate of its I/O in bytes (`RunAsync(fn, cost)`):
the size of each group of chunks for reads, the size of the input array for
writes, and a nominal `kDefaultTaskCost` of 1 MiB otherwise (metadata, and
partitioning a read, which may read a small column itself). `Dispatch(cost)`
(`isolated_table_proxy.cc`) picks the instance with the **least outstanding
cost** and reserves the cost on it. The task releases the reservation when it
completes, before its future's continuations run. Notes:

- **Bytes, not task counts.** Task counts cannot tell a metadata lookup from a
  50 MB read, and queued pieces of a large read were placed behind other
  columns' reads.
- **The dispatching task is discounted.** The reads following a partition are
  dispatched from the partitioning instance's own thread. That instance is about
  to become free, so the cost of a task running on the calling thread is
  discounted.
- **Ties go to instance 0.** This keeps successive calls on an idle proxy on one
  instance, whose casacore caches are warm. Rotating ties was measured, and
  slowed sequential 100-row reads by about a quarter with no gain for
  concurrent reads.
- **Statistics.** Reservations live in each instance's `InstanceStats`, together
  with cumulative counts of tasks, queue wait, busy time and bytes read or
  written. Proxies spawned from a parent share these, as they share its pools.
  `Table._instance_stats()` exposes them for benchmarking
  (`benchmarks/instance_balance.py`).

### Splitting reads

A single high-level read is decomposed into independent chunk reads, which are
grouped and spread over the instances: the **multi-reader** parallelism. With one
instance, the partitioning task reads the chunks itself, avoiding a hand-off.

The row dimension of a fixed-shape selection is a single `casacore::RefRows`,
gaps included, so it would otherwise always form one chunk.
`DataPartition::Make` therefore takes `RowSplitOptions`, splitting the selected
rows into at most one piece per instance, of equal size and each at least
8 MiB. Smaller requests are read whole: callers issuing many concurrent requests
already keep the instances busy, and splitting their requests only adds
hand-offs between threads.

Pieces are cut where the column's storage manager starts a new unit of rows:
a band of tiles of `TiledColumnStMan`, `TiledShapeStMan` and `TiledDataStMan`,
or a timestep block of `DyscoStMan`. A band holds the rows in one tile along
the last hypercube axis. It spans any other row axes of a `TiledDataStMan`
hypercube, which steps rows through, say, baselines before times. Instances read and decode whole units, so a cut through a unit
would read it on two instances: with tiles of 1024 rows, pieces of 1 MiB cut
mid-tile read twice the bytes of the column, and 8 MiB pieces read up to half
as much again with 4096-row tiles. `MakeRowUnitStart` finds a row's unit
start, on the instance's thread in the partitioning task, and each even
boundary moves to the nearer start, merging pieces smaller than a unit.
Dysco's block size comes from the overlay port's
`003-dysco-rows-per-block.patch`, as `DyscoStMan` doesn't otherwise publish it.
Other storage managers, and reference tables, keep the even boundaries:
`TiledCellStMan` stores each row in its own hypercube, `StandardStMan`
buckets are small, and `StManAipsIO` holds whole columns in memory. Missing (negative) rows stay in chunks of their own. Variably-shaped columns already
form one chunk per row. `GroupChunks` then groups the chunks into one group per
instance, with similar numbers of elements.

The decomposition machinery lives alongside the ITP:

- `selection.{h}` — the row/column index selection requested by the caller.
- `data_partition.{h,cc}` — splits a selection into per-chunk work items.
- `result_shape.{h,cc}` — computes the shape of the Arrow result (fixed vs.
  variable rank).
- `read_impl.{h,cc}` — orchestrates the chunked reads and assembles the result.

Because the heavy work happens in C++ pool threads and the C++ waits do not hold
the GIL, multiple Python threads can keep these instances busy concurrently.

## 4. Locking: multi-reader / single-writer over `fcntl`

Every dispatched operation is guarded by `MaybeLockAndFinalise`
(`isolated_table_proxy.h`), an RAII object that acquires the table lock on
construction and finalises on destruction:

- On construction it calls `proxy->lock(write?, ...)` and **honours the result** —
  if the lock is not acquired it throws (caught by the dispatcher and turned into
  `Status::Invalid`), so a functor never runs unlocked.
- On destruction, for write locks it flushes (`proxy->flush(false)`) before
  unlocking, and it **never lets an exception escape** the destructor (which could
  otherwise run during stack unwinding and call `std::terminate`).

Underneath, casacore's `FileLocker` is patched
(`vcpkg/overlay-ports/casacore/001-casacore-cmake.patch`,
`casa/IO/FileLocker.{cc,h}` and `casa/IO/LockFile.cc`) to provide
**multi-reader / single-writer (MRSW)** semantics over the underlying `fcntl`
advisory lock. The vanilla `FileLocker` cannot express this safely within one
process because POSIX `fcntl` advisory locks are owned per *(process, inode)*: a
second lock request on an overlapping range from the same process replaces or
merges the existing lock instead of blocking, and closing any fd to the file drops
*all* the process's locks. The patch turns `FileLocker` into a thread-aware MRSW
state machine layered on top of `fcntl`:

- a per-thread map of held lock types, guarded by a state mutex and condition
  variable, with refcounting so the real `fcntl` lock is taken by the *first*
  acquirer and released only by the *last* releaser;
- a shared, refcounted file descriptor per lock-file path (`LockFile`), so the
  "closing any fd drops all locks" footgun is neutralised;
- request IDs keyed on a hashed thread id rather than the pid.

The intent is that, within one process, many readers and a single writer can hold
coordinated locks on the same table files.

## 5. Writes spawn on the first instance (single-writer)

Concurrent writes issued from multiple threads race in the casacore layer, so
arcae funnels **all writes through a single instance and thread**. Write-side
entry points on `NewTableProxy` — `PutColumn`, `AddRows`, `AddColumns`
(`cpp/arcae/new_table_proxy.cc`) — route through `itp_->SpawnWriter()`:

```cpp
Result<bool> NewTableProxy::AddRows(std::size_t nrows) {
  return itp_->SpawnWriter()...->...;
}
```

`SpawnWriter()` (`isolated_table_proxy.cc`) returns a lightweight ITP that
**borrows instance 0** of the parent (never a least-loaded pick):

```cpp
auto instance = 0;  // GetInstance();
itp->proxy_pools_.push_back(proxy_pools_[instance]);
```

Instance 0 is chosen deliberately and unconditionally: it means writes keep
working after non-syncable operations such as `AddColumns`, which other instances
would not observe. The spawned writer dispatches with `lock_type = Write`, giving
the single-writer half of the MRSW model. It uses a custom deleter that clears the
borrowed `proxy_pools_`/`dependencies_` and marks itself closed *before* deletion,
so tearing down the borrowed writer never double-flushes or double-closes the
shared instance-0 proxy. The actual write logic lives in `write_impl.{h,cc}`.

This yields the overall concurrency model:

| Workload | Routing | Lock |
|----------|---------|------|
| Reads | least outstanding I/O of `N` instances (fan-out) | `Read` (shared) |
| Writes | instance 0 only (serialised) | `Write` (exclusive) |

## 6. Marshalling CASA data into Arrow, exposed via Cython

The C++ core reads CASA column data and assembles it into an `arrow::Table`
(`NewTableProxy::ToArrow`, `read_impl.cc`). The mapping from CTDS types to Arrow
is:

- **Fixed-shape multidimensional** data (e.g. visibilities) → nested
  `FixedSizeListArray`s.
- **Variably-shaped** data (e.g. some subtable columns) → nested `ListArray`s.
- **Complex** values → an extra two-element `FixedSizeListArray` of floats (Arrow
  has no native complex type yet).

These Arrow buffers are bit-compatible with NumPy's memory layout, so `getcol` /
`putcol` reinterpret the underlying buffers rather than copying element-by-element,
transparently bridging Arrow's nested representation and NumPy arrays.

The Python boundary is a Cython extension, `src/arcae/lib/arrow_tables.pyx`, which
wraps `NewTableProxy` as the Python `Table` class. The C++-built `arrow::Table` is
handed to Python as a `pyarrow.Table` with no further copy. Key surface:

- `Table.from_filename` / `from_taql` / `ms_from_descriptor` — construction;
- `to_arrow(index=...)` — read (a slice of) the table as a `pyarrow.Table`;
- `getcol` / `putcol`, `addrows` / `addcols`, `nrow`, `columns`, descriptor
  accessors.

The public Python entry point is `arcae.table(filename, ninstances=1,
readonly=True, lockoptions="auto")` (`src/arcae/__init__.py`), which constructs a
`Table`. The Cython module import is deferred until first use to avoid clashes
between arcae's statically-linked casacore and any separately-imported
`python-casacore`.

---

## File map

| Concern | Files |
|---------|-------|
| Python entry point / API | `src/arcae/__init__.py`, `src/arcae/lib/arrow_tables.pyx` |
| Per-instance threading & locking | `cpp/arcae/isolated_table_proxy.{h,cc}` |
| Table proxy & public C++ API | `cpp/arcae/new_table_proxy.{h,cc}`, `cpp/arcae/table_factory.{h,cc}` |
| Read path | `cpp/arcae/read_impl.{h,cc}`, `cpp/arcae/data_partition.{h,cc}`, `cpp/arcae/result_shape.{h,cc}`, `cpp/arcae/selection.h` |
| Write path | `cpp/arcae/write_impl.{h,cc}` |
| Type / descriptor handling | `cpp/arcae/type_traits.{h,cc}`, `cpp/arcae/descriptor.{h,cc}` |
| Patched casacore (cache + locking) | `vcpkg/overlay-ports/casacore/001-casacore-cmake.patch` |
| C++ tests | `cpp/tests/` (gtest) |
| Python tests | `src/arcae/tests/` (pytest) |
