# arcae Design

How arcae provides GIL-free, multi-threaded access to CASA tables and exposes
their contents as Apache Arrow tables. Paths below are relative to the
[ratt-ru/arcae](https://github.com/ratt-ru/arcae) repository root.
This document only applies to the `0.4.0-dev` branch which contains
true multi-threaded write support via the `FileLocker` changes. The `main` branch does not modify `FileLocker` and is only safe for
multi-threaded reads.

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
*safe* and *parallel* anyway. The load-bearing pieces are described below.

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

The price is that instances no longer share anything, including their view of
the table's structure. Keeping them consistent when that structure changes is
the subject of §6.

A second patch (`vcpkg/overlay-ports/casacore/002-json-parser-mutex.patch`)
serialises casacore's JSON parser. It is a flex scanner and bison parser whose
state is process wide, and arcae parses JSON (lock options, column and table
descriptors, keywords) on the instance threads of many tables at once.

## 2. `IsolatedTableProxy`: N instances, one thread each

`IsolatedTableProxy` (ITP) — `cpp/arcae/isolated_table_proxy.{h,cc}` — is the
core concurrency primitive. An ITP holds `N` **instances**, where each instance is:

- one `casacore::TableProxy` (the table opened by a user-supplied functor), held
  in a `ProxySlot` so that it can be replaced (§6), plus
- one dedicated single-thread Arrow `ThreadPool`.

Because casacore tables are not thread-safe, **each `TableProxy` is confined to
its own pool thread for its entire lifetime** — every operation on that proxy is
submitted to that one thread, so the CTDS only ever sees single-threaded access.
Combined with the `thread_local` cache (§1), the `N` instances are mutually
independent even though they refer to the same table on disk.

`IsolatedTableProxy::Make(functor, ninstances, reopen)` builds the ITP by
submitting the open functor to each of the `N` pools, one at a time, and
collecting the resulting proxies into a `ProxyAndPool` vector. `reopen` is the
recipe for opening an instance again later (§6). `ninstances` is surfaced all the
way up to the Python entry point `arcae.table(filename, ninstances=...)`.

Operations are dispatched onto an instance through three templated entry points:

- `RunAsync(fn, lock_type)` → returns an `arrow::Future`.
- `RunSync(fn, lock_type)` → runs and waits for the result.
- `Then(future, fn, lock_type)` → chains a continuation onto a prior future.

Each dispatch:

1. picks an instance,
2. submits the functor to that instance's single thread,
3. locks the instance through `LockInstance`, which wraps a
   `MaybeLockAndFinalise` guard (see §4) and replaces an instance that has
   fallen out of date (see §6),
4. invokes the functor with the instance's `TableProxy&`,
5. converts any `casacore::AipsError` into an `arrow::Status` so exceptions never
   escape a pool thread.

Dispatch lambdas capture `weak_from_this()` (not raw `this`); the task re-locks
the weak pointer at start and bails with `Status::Invalid("TableProxy is closed")`
if the ITP has been destroyed, avoiding use-after-free on teardown.

### Spawning dependent proxies

`Spawn(functor)` creates a derived ITP whose proxies are built *from* this ITP's
proxies (e.g. a TAQL query producing a reference table). It mirrors the instance
topology, runs the functor on each source instance under a read lock, and records
a `dependencies_` edge so the parent ITP outlives the child. It also records the
*root* ITP, the one made by `Make` that the chain of spawns started from, which
is what a derived instance is rebuilt over (§6).

## 3. Multiplexing I/O across instances (read fan-out)

Reads are spread across all `N` instances. `GetInstance()`
(`isolated_table_proxy.cc`) selects the **least-loaded** instance by querying each
pool's pending task count and returning the one with the fewest:

```cpp
for (std::size_t i = 0; i < proxy_pools_.size(); ++i) {
  const auto pool_tasks = proxy_pools_[i].io_pool_->GetNumTasks();
  if (pool_tasks < num_tasks) { instance = i; num_tasks = pool_tasks; }
}
```

A single high-level read is decomposed into many independent chunk reads, each
dispatched via `RunAsync`/`Then` with `lock_type = Read`, and those chunks fan out
across the instances — this is the **multi-reader** parallelism. The decomposition
machinery lives alongside the ITP:

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
- After taking a read lock it forces a `resync()`. The resync casacore performs
  on lock acquisition only revisits data managers whose change counter moved,
  which misses array column storage; the forced resync does not. Because the
  instances are independent tables (§1), this matters within one process just as
  it does across processes. A resync that finds the column set out of date marks
  the guard `stale` rather than failing, for `LockInstance` to act on (§6).
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

Instance 0 is chosen deliberately and unconditionally. Having exactly one
instance that ever holds a write lock is what makes the others safe to discard
and reopen when the table's structure changes (§6). It is also the instance that
makes a structural change, and so the one that is already current afterwards. The spawned writer dispatches with `lock_type = Write`, giving
the single-writer half of the MRSW model. It uses a custom deleter that clears the
borrowed `proxy_pools_`/`dependencies_` and marks itself closed *before* deletion,
so tearing down the borrowed writer never double-flushes or double-closes the
shared instance-0 proxy. It shares instance 0's `ProxySlot` rather than copying
the proxy, and carries the parent's refresh recipe, so it both sees and can make
a replacement of that instance. The actual write logic lives in
`write_impl.{h,cc}`.

This yields the overall concurrency model:

| Workload | Routing | Lock |
|----------|---------|------|
| Reads | least-loaded of `N` instances (fan-out) | `Read` (shared) |
| Writes | instance 0 only (serialised) | `Write` (exclusive) |
| Structural changes (`AddColumns`) | instance 0, then every other instance is reopened (§6) | `Write`, released before reopening |

## 6. One table across many instances: structural consistency

An arcae table must behave as one table, even though it is `N` independent
casacore tables, and even though other arcae handles and other processes may have
the same table open. For data this falls out of locking (§4): a lock or resync
brings an instance up to date with rows and values written elsewhere. The table's
*structure* is different.

### Why instances must be reopened

casacore fixes an open table's column set for the lifetime of the table object.
The lock file, which is shared, records the table's column count, and
`PlainTable::lock` and `PlainTable::resync` compare it with the instance's own
`TableDesc`, throwing *"another process changed the number of columns"* when they
differ. An instance that has not seen a new column therefore can no longer lock,
read, or even close (closing flushes, and the flush takes a lock).

Stock casacore rarely meets this within a process, because its table cache
shares one `PlainTable` per path: a column added through any handle is added for
all of them. arcae's `thread_local` cache (§1) removes that sharing on purpose, so
a column added through instance 0 is invisible to instances 1..N-1, to other
arcae handles, and to other processes, exactly as if each were a separate
process.

No sync can absorb the change. `syncTable()` defers to `ColumnSet::syncColumns`,
which refuses a column count mismatch by design and otherwise only refreshes the
keywords and attributes of columns that already exist. `reopenRW()` reopens the
existing data managers for writing and never touches the `TableDesc`. The only
way for an instance to see a new column is to become a new table object, so
arcae **replaces the instance**.

### How an instance is replaced

`IsolatedTableProxy::RefreshInstance(i)` replaces instance `i`, on instance
`i`'s own thread, in one of two ways.

- **Instances of a table made by `Make` are reopened** with the ITP's
  `ReopenFn(name, writable)`. The recipe is supplied by the table factories
  (`cpp/arcae/table_factory.cc`) and is separate from the functor that first
  opened the table: the `DefaultMS` and `CreateTable` functors create their table
  with `Table::New`, and running them again would recreate it. `OpenTable`
  reopens with its original lock options and cache sizes; created tables reopen
  with user locking and their cache sizes.
- **Instances of a table made by `Spawn` are rebuilt**, not queried again. The
  stale instance's row numbers in the root table (`Table::rowNumbers()`) and its
  column names are taken, and the selection is rebuilt as
  `root(rows).project(columns)` over the root ITP's instance `i`, which is itself
  refreshed first if needed. A casacore `RefTable` always refers to the root
  table directly, even when made from another reference table, so nothing is
  needed from intermediate ITPs. Re-running the TAQL query would be simpler but
  wrong: the table may have changed since the query ran, and a re-evaluated
  selection could hold rows that the other instances, built earlier, do not.

The mechanics of the swap rely on invariants set elsewhere in the design.

- **The stale table is dropped, not closed.** `TableProxy::close()` flushes, and
  the flush takes a lock, which a stale instance cannot do. The table is instead
  unlocked and released without flushing. That is safe only because a stale
  instance holds no write lock: releasing a write lock writes the table files out,
  and an out-of-date `TableDesc` written over the current one would drop the new
  column from disk. Only instance 0 ever takes a write lock (§5), and it flushes
  as it releases each one.
- **It is evicted from the thread's table cache.** A derived table on the same
  thread (e.g. a TAQL selection) holds the stale root table alive, and opening the
  path again would return it from the cache. The entry is removed before
  reopening. When the stale table is eventually destroyed it removes its cache
  entry *by name*, evicting the fresh table's entry instead; the only effect is
  that a later open of that path on that thread builds a separate table object,
  which the design already handles across threads.
- **The slot is replaced on its own thread.** A `ProxySlot` is only ever read or
  written on its instance's thread, so a replacement needs no synchronisation
  beyond the thread confinement of §2, and every ITP sharing the slot (the
  `SpawnWriter` ITP, §5) sees it.
- **Replacement never runs under a write lock**, since reopening takes locks of
  its own.

### When instances are replaced

Instances are replaced both eagerly and lazily.

- **Eagerly, after a structural change.** `NewTableProxy::AddColumns` adds the
  columns through the writer (instance 0), waits for the write, whose lock is
  released as it completes, and then calls `RefreshInstances(0)` to replace every
  other instance, one at a time. The handle that made the change is consistent
  before `addcols` returns.
- **Lazily, on next use.** Every task an ITP dispatches (and the source lock
  `Spawn` takes) locks its instance through `LockInstance`. If the lock throws the
  column count mismatch, or the forced resync reports it (§4), the instance is
  replaced and locked again, once. Only the lock is retried, never the task's
  functor, so no write is repeated. This is what keeps every *other* view of the
  table consistent: other arcae handles on the same path, derived TAQL tables,
  the writer instance itself, and tables open when another process adds a column,
  which no bookkeeping within this process could reach. Callers never need to
  invalidate or reopen tables themselves.

`Close()` treats the column count mismatch as the expected state of an instance
that nobody used after a column was added, since it has nothing to flush, and
does not report it. Any other failure to close still releases the table and is
reported.

### Limits

- A derived table that is not a selection over its source (a TAQL result computed
  into a table of its own) cannot be rebuilt, and returns `NotImplemented`. It
  cannot go stale on account of its source either.
- A derived table that renames columns cannot be expressed as a projection of the
  root, and returns `NotImplemented`.
- An ITP made without a `ReopenFn` (`Taql` with no input tables) cannot refresh.
- The column count mismatch is recognised by its message
  (`IsColumnCountMismatch`), which is all casacore provides.
- arcae does not expose column removal, which would change the column set the
  same way and would be handled by the same machinery.

## 7. Marshalling CASA data into Arrow, exposed via Cython

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
| Table proxy & public C++ API | `cpp/arcae/new_table_proxy.{h,cc}`, `cpp/arcae/table_factory.{h,cc}` (including the reopen recipes) |
| Read path | `cpp/arcae/read_impl.{h,cc}`, `cpp/arcae/data_partition.{h,cc}`, `cpp/arcae/result_shape.{h,cc}`, `cpp/arcae/selection.h` |
| Write path | `cpp/arcae/write_impl.{h,cc}` |
| Type / descriptor handling | `cpp/arcae/type_traits.{h,cc}`, `cpp/arcae/descriptor.{h,cc}` |
| Patched casacore (cache + locking) | `vcpkg/overlay-ports/casacore/001-casacore-cmake.patch` |
| Patched casacore (JSON parser mutex) | `vcpkg/overlay-ports/casacore/002-json-parser-mutex.patch` |
| C++ tests | `cpp/tests/` (gtest) |
| Python tests | `src/arcae/tests/` (pytest) |
