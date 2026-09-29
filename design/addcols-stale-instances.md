# Stale sibling instances after `AddColumns`

Status: fixes 1 and 2 implemented; fix 3 proposed. Target branch: `0.4.0-dev`
(MRSW write support).
Found while migrating dask-ms off python-casacore (ratt-ru/dask-ms#384).
Related: ska-sa/arcae#241.

## Summary

An `IsolatedTableProxy` is `ninstances` independent `casacore::TableProxy`
objects. `AddColumns` is applied to exactly one of them, so the other
`ninstances - 1` are left describing a table that no longer exists. casacore
has no way to bring them back into line, so they can no longer be locked,
read, **or closed** — the table cannot even be torn down cleanly.

This is not specific to dask-ms. It reproduces in a dozen lines of arcae.

## Reproduction

```python
import arcae
from arcae.lib.arrow_tables import Table

DESC = {"NEWCOL": {"valueType": "double", "ndim": 0, "option": 0, "maxlen": 0,
                   "comment": "x", "keywords": {},
                   "dataManagerType": "StandardStMan", "dataManagerGroup": "SSM"}}

for n in (1, 2, 4):
    Table.ms_from_descriptor(path, subtable="MAIN").close()
    t = arcae.table(path, readonly=False, ninstances=n)
    t.addcols(DESC, {})
    t.close()
```

```
ninstances=1: close OK
ninstances=2: close FAILED -> Error closing table: Table::lock cannot sync table ...;
                              another process changed the number of columns
ninstances=4: close FAILED -> (same)
```

`ninstances=1` passes only because there are no siblings to go stale.

Reads fail the same way, but **only once concurrency reaches the stale
instances**, which makes the bug latent and easy to miss:

```python
t = arcae.table(path, readonly=False, ninstances=4)
t.addrows(200); t.addcols(DESC, {})
# 20 sequential getcol calls      -> ok=20  fail=0
# 200 getcol calls over 16 threads -> ok=86  fail=114
```

Sequential reads all land on instance 0, which is the one that performed the
add and is therefore current. `GetInstance()` only hands out a sibling when
instance 0 is busy. So a single-threaded test of `addcols`-then-read passes,
and the same code fails under load — or, as in dask-ms, as soon as a dask
graph reads the table from several threads.

A separate `arcae.table` handle on the same path, or a reader in another
process, fails on its first locked operation regardless of concurrency.

## Root cause

`AddColumns` ([`cpp/arcae/new_table_proxy.cc:213`](../cpp/arcae/new_table_proxy.cc))
routes through `SpawnWriter()`
([`cpp/arcae/isolated_table_proxy.cc:57`](../cpp/arcae/isolated_table_proxy.cc)),
which borrows instance 0. Only that instance's `TableDesc` learns about the
new column; the lock file, which is shared, now advertises the new column
count to everybody.

Every subsequent lock acquisition compares the two and throws
(casacore v3.8.1, `tables/Tables/PlainTable.cc:453`):

```cpp
if (ncolumn != tableDesc().ncolumn()) {
    throw (TableError ("Table::lock cannot sync table "
                       + tableName() + "; another process "
                       "changed the number of columns"));
}
```

`PlainTable::resync` carries the identical check at `PlainTable.cc:535`.

Teardown is caught by this too, which is what makes it more than a read bug.
`Close()` does `flush()` then `close()` per instance
([`cpp/arcae/isolated_table_proxy.cc:107`](../cpp/arcae/isolated_table_proxy.cc)),
`TableProxy::close()` flushes, and the flush takes a table lock of its own
(`keywordSet()` → `ColumnSet::userLock`). So a stale instance throws on the
way out and is never actually closed.

`SpawnWriter` already acknowledges half of this:

```cpp
// Using the first instance means that writes can still work after
// non-syncable operations like AddColumns
auto instance = 0;  // GetInstance();
```

Pinning the writer to instance 0 keeps *writes* working. The siblings were
left for later; this note is that later.

## Why there is no sync-based fix

Worth stating explicitly, because both obvious candidates were tried and
neither can work.

**`resync()` / `syncTable()` cannot absorb the change.** `syncTable()`
(`PlainTable.cc:467`) re-reads the table into a separate `PlainTable` and then
delegates to `ColumnSet::syncColumns`, which throws on a count mismatch **by
design** (`ColumnSet.cc:957`) and otherwise only refreshes keywords and the
attributes of columns that already exist. It never adds one.

**`TableProxy::reopenRW` is the wrong primitive.** `PlainTable::reopenRW`
(`PlainTable.cc:348`) early-returns when the table is already writable, sets
`option_p = Table::Update`, and calls `colSetPtr_p->reopenRW()` /
`keywordSet().reopenRW()`. `ColumnSet::reopenRW` (`ColumnSet.cc:723`) walks
`blockDataMan_p` and `colMap_p` *as they already are*, reopening existing data
managers for write. Nothing on that path touches `tableDesc()`, so the very
next `lock()` throws identically. It would also make a handle opened readonly
writable, which is not what a reader asked for.

The underlying rule is that **an open casacore table's column set is fixed for
the lifetime of the object.** The only way to see a new column is to construct
a new table object.

Note this is a consequence of the patched casacore's `thread_local`
`PlainTable::theirTableCache`, and is the price of the MRSW design rather than
a defect in it. Stock casacore shares one `PlainTable` per path per process,
so an `addcols` is visible to every handle at once and this cannot arise —
which is why python-casacore never hit it.

## Impact

- Any `ninstances > 1` table that gains a column becomes uncloseable, and
  unreadable as soon as reads reach a sibling instance. Failing to close leaks
  the casacore table and its lock registration, and risks throwing from a
  destructor during teardown.
- The read failure is load-dependent, so it will pass a sequential test and
  fail in production. `Close()` is the reliable symptom: it touches every
  instance unconditionally and so fails every time.
- A second process adding a column breaks readers in the first, and no amount
  of intra-process bookkeeping can fix that — only a reopen can.
- dask-ms currently works around it in `CasaTable.invalidate`, which walks its
  handle registry after `addcols` and drops every cached table for that path
  plus anything derived from it. Fix 3 below would let that be deleted.

## Proposed changes

### 1. Refresh sibling instances after a structural change

Implemented. Restores the contract that an arcae table behaves as one table.

`IsolatedTableProxy::RefreshInstances(except)` replaces every instance other
than `except` with a freshly opened one, and `AddColumns` calls it with
`except = 0` once its write has completed.

**It reopens; it does not re-run the factory.** An earlier draft of this note
proposed retaining `Make`'s factory functor and calling it again. That is only
correct for `OpenTable`: the `DefaultMS` and `CreateTable` factories build a
`SetupNewTable` with `Table::New`, so running them again recreates the table.
`Make` instead takes a separate `ReopenFn(name, writable)`, which the
factories in `table_factory.cc` supply: `OpenTable` reopens with its own lock
options and cache sizes, and created tables reopen with user locking and their
cache sizes. An ITP without one (`Taql` with no input tables, or a `Spawn`ed
one) returns `NotImplemented` from `RefreshInstances` when it has more than one
instance.

How each instance is replaced, on its own isolation thread:

- **Drop the stale table, don't close it.** `TableProxy::close()` flushes, and
  the flush takes a lock, which throws for a stale instance. The table is
  instead unlocked and released (`tp.table() = Table()`). That is safe only
  because the instance holds no write lock: releasing a write lock writes the
  table files out (`TableLockData::release` → `PlainTable::putFile`), and a
  stale instance would write its out-of-date `TableDesc` over the current one,
  dropping the new column from disk. Siblings never take a write lock because
  `SpawnWriter` pins writes to instance 0; `RefreshInstances(0)` depends on
  that and the code says so.
- **Evict it from the thread's table cache.** A TAQL table spawned from this
  one shares its isolation threads and holds the root `PlainTable` alive
  (`RefTable::baseTabPtr_p`). Opening the path again would then return that
  same stale table from `theirTableCache`, so the entry is removed first
  (a no-op when the drop already destroyed it). When the stale table does
  eventually die, `closeObject` removes the cache entry *by name*, evicting
  the fresh table's entry instead. The consequence is only that a later open
  of the path on that thread builds a separate table object, which arcae
  already handles across threads. A casacore patch making `TableCache::remove`
  pointer-aware would remove this wart.
- **One instance at a time**, as `Make` creates them. This was first forced
  by casacore's flex/bison JSON parser, which the reopen recipes use and which
  was not thread-safe: refreshing concurrently corrupted the parse. The
  casacore overlay port now serialises parsing (`002-json-parser-mutex.patch`,
  #246), so a concurrent refresh would be safe, but it gains little for an
  operation that only follows a structural change.
- **After the write lock is released**: `AddColumns` waits on the write, whose
  `MaybeLockAndFinalise` has unlocked by then, before refreshing.

Instances live in a shared `ProxySlot`, so the ITP returned by `SpawnWriter`
sees an instance that has been replaced rather than keeping its own copy of
the old proxy.

`removeColumn` is the same class of bug, but arcae does not expose column
removal. Keyword writes do not change the column count and are unaffected.

A `Spawn`ed (TAQL) table's own instances are not refreshed: its reference
tables are over the stale root and cannot lock or close once reads reach its
siblings. Fix 3 is what handles them.

### 2. Do not leak an instance whose close fails

Implemented. `TableProxy::close()` flushes internally, so splitting `flush()`
and `close()` into separate `try` blocks (the earlier proposal) would still
throw from `close()` and release nothing. `Close()` now drops the table, as in
fix 1, whenever flushing or closing throws, and still reports the error.

### 3. Lazy self-heal in `MaybeLockAndFinalise`

The strategic fix, and the only one that covers every case.

In `MaybeLockAndFinalise`
([`cpp/arcae/isolated_table_proxy.h:46`](../cpp/arcae/isolated_table_proxy.h)),
catch the column-count `TableError` out of `proxy->lock()` / `proxy->resync()`,
rebuild that one instance from `factory_`, and retry once.

This subsumes fix 1 — each instance repairs itself on next use, so
`AddColumns` needs no special casing — and additionally handles:

- another **process** adding a column, which nothing else can address;
- other arcae tables in this process, i.e. the readers that dask-ms currently
  invalidates by hand.

Costs and cares:

- It sits in the hot path, inside a `try`/`catch` that already exists.
- The retry must re-fetch the proxy from the pool slot rather than reuse its
  local `shared_ptr`. The old proxy stays alive through that `shared_ptr`, so
  the swap is safe, but a stale local reference would defeat the point.
- Distinguish the column-count error specifically. Retrying every `TableError`
  would turn genuine failures into infinite loops.

## Recommended order

1. **Fix 2**: done.
2. **Fix 1**: done. Unblocks `addcols` on multi-instance tables.
3. **Fix 3**: when the goal is to delete dask-ms's `CasaTable.invalidate`,
   and to repair TAQL tables derived from a table that gained a column. The
   retry should reuse `RefreshProxy` from `isolated_table_proxy.cc`; for a
   spawned ITP it must refresh the parent's slot on the same thread and then
   re-run the spawn functor.

## Downstream

dask-ms `replace-python-casacore-with-arcae` currently carries
`CasaTable.invalidate`, called from `_updated_table` immediately after
`addcols`. It walks the live-handle registry, closes and evicts every cached
table for that path plus anything derived from it (TAQL queries over it), so
each handle reopens on next access. It deliberately includes the handle that
performed the add, precisely because of the sibling-instance problem described
here.

That workaround is only sound because `_updated_table` runs while the write
graph is being built, with no tasks in flight. It cannot address the
cross-process case. Fix 3 removes the need for it entirely.
