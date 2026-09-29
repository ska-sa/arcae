#include "arcae/isolated_table_proxy.h"

#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <string_view>

#include <arrow/status.h>
#include <arrow/util/future.h>
#include <arrow/util/logging.h>
#include <arrow/util/thread_pool.h>

#include <casacore/casa/Containers/Block.h>
#include <casacore/casa/Exceptions/Error.h>
#include <casacore/tables/Tables/PlainTable.h>
#include <casacore/tables/Tables/RowNumbers.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/TableCache.h>
#include <casacore/tables/Tables/TableDesc.h>
#include <casacore/tables/Tables/TableProxy.h>

using ::arrow::Future;
using ::arrow::Result;
using ::arrow::Status;
using ::arrow::internal::GetCpuThreadPool;
using ::arrow::internal::ThreadPool;
using ::casacore::TableProxy;

namespace arcae {
namespace detail {
namespace {

using RowNumbersPtr = std::shared_ptr<casacore::RowNumbers>;

// Release the table held by tp without flushing it. Must run on the
// isolation thread that owns tp: destroying a PlainTable removes it from
// the thread_local table cache it was registered in.
//
// This is only safe because tp does not hold a write lock once unlocked.
// Releasing a write lock writes the table files out, and for an instance
// whose column set is out of date that would write a stale table
// description over the current one. Siblings never take a write lock
// (SpawnWriter pins writes to instance 0), which is what makes a stale
// instance safe to discard.
void DropTable(TableProxy& tp) noexcept {
  try {
    tp.unlock();
  } catch (const std::exception& e) {
    ARROW_LOG(DEBUG) << "Error unlocking table: " << e.what();
  }
  try {
    tp.table() = casacore::Table();
  } catch (const std::exception& e) {
    ARROW_LOG(WARNING) << "Error releasing table: " << e.what();
  }
}

// Flush and close tp on its isolation thread.
//
// TableProxy::close() flushes, and the flush takes a table lock of its
// own, so an instance that can no longer lock -- one whose column count
// another instance changed -- throws before anything is closed. Drop the
// table regardless, so that a failed close does not keep it and its lock
// registration alive until the TableProxy happens to be destroyed.
Status CloseProxy(TableProxy& tp) {
  try {
    tp.flush(false);
    tp.close();
    return Status::OK();
  } catch (const std::exception& e) {
    DropTable(tp);
    // An out of date instance has nothing to lose: writes go through an
    // instance that flushes as it releases each write lock. So this is the
    // expected state of an instance that was never used after another
    // handle added a column, not a failure.
    if (IsColumnCountMismatch(e)) return Status::OK();
    return Status::Invalid("Error closing table: ", e.what());
  }
}

// Replace the TableProxy in proxy with a freshly opened one.
// Must run on the isolation thread that owns the instance.
Status RefreshProxy(std::shared_ptr<TableProxy>& proxy, const ReopenFn& reopen) {
  std::string name;
  bool writable;

  try {
    const auto& table = proxy->table();
    if (table.isNull()) return Status::Invalid("Table instance is closed");
    name = table.tableName();
    writable = table.isWritable();
  } catch (const std::exception& e) {
    return Status::Invalid("Error inspecting table instance: ", e.what());
  }

  DropTable(*proxy);

  // Opening a table first consults this thread's table cache. If something
  // else on this thread -- e.g. a TAQL query over this table -- still
  // references the stale table it is alive and cached, and opening it again
  // would simply return it. Evict it so the open constructs a new one.
  // When nothing else did reference it, dropping it already removed the
  // entry and this does nothing.
  //
  // The stale table removes its cache entry by name when it is eventually
  // destroyed, which then evicts the new table instead. That only means a
  // later open of this path on this thread constructs a separate table
  // object, the situation arcae is already designed around across threads.
  casacore::PlainTable::tableCache().remove(name);

  ARROW_ASSIGN_OR_RAISE(auto fresh, reopen(name, writable));
  proxy = std::move(fresh);
  return Status::OK();
}

// Rebuild a derived (reference table) instance over the root table's
// instance on the same isolation thread, keeping the rows and columns it
// selected. The caller holds a read lock on root, a current root instance.
Status RebuildDerived(std::shared_ptr<TableProxy>& proxy, const RowNumbersPtr& rows,
                      const casacore::Vector<casacore::String>& columns,
                      const TableProxy& root) {
  try {
    const auto& root_table = root.table();
    const auto& root_desc = root_table.tableDesc();
    casacore::Block<casacore::String> names(columns.size());
    for (std::size_t c = 0; c < columns.size(); ++c) {
      // TAQL can rename columns, which a projection of the root cannot express
      if (!root_desc.isColumn(columns[c])) {
        return Status::NotImplemented("Refreshing a derived table with column ",
                                      columns[c], ", which is not in its source table");
      }
      names[c] = columns[c];
    }
    auto derived = root_table(*rows).project(names);
    proxy = std::make_shared<TableProxy>(derived);
    return Status::OK();
  } catch (const std::exception& e) {
    return Status::Invalid("Error rebuilding derived table: ", e.what());
  }
}

}  // namespace

bool IsColumnCountMismatch(const std::exception& e) {
  // Raised by PlainTable::lock, PlainTable::resync and ColumnSet::syncColumns.
  // casacore gives no more specific way to tell it apart.
  return std::string_view(e.what()).find("changed the number of columns") !=
         std::string_view::npos;
}

bool IsolatedTableProxy::IsClosed() const { return is_closed_; }

Result<std::unique_ptr<IsolatedTableProxy::MaybeLockAndFinalise>>
IsolatedTableProxy::LockInstance(std::size_t instance, CasaLockType lock_type) const {
  for (int attempt = 0;; ++attempt) {
    bool retry = attempt == 0 && CanRefresh();
    std::unique_ptr<MaybeLockAndFinalise> lock;
    try {
      lock = std::make_unique<MaybeLockAndFinalise>(GetProxy(instance), lock_type);
    } catch (const casacore::AipsError& e) {
      // The constructor has already released anything it acquired
      if (!retry || !IsColumnCountMismatch(e)) throw;
    }
    // A stale resync is not fatal: the operation can proceed against the
    // columns the instance knows about, as it does when it cannot refresh
    if (lock && (!lock->stale || !retry)) return lock;
    // Release the lock before reopening, which takes locks of its own
    lock.reset();
    ARROW_LOG(DEBUG) << "Refreshing out of date table instance " << instance;
    ARROW_RETURN_NOT_OK(RefreshInstance(instance));
  }
}

Status IsolatedTableProxy::RefreshInstance(std::size_t instance) const {
  auto& proxy = proxy_pools_[instance].slot_->proxy;

  if (!root_) {
    if (!reopen_) {
      return Status::NotImplemented(
          "Refreshing the instances of this table after a structural change");
    }
    return RefreshProxy(proxy, reopen_);
  }

  // A reference table always refers to the root table directly, even when
  // made from another reference table, so only the root instance needs to
  // be current. Take what the stale instance selected before dropping it.
  auto rows = std::make_shared<casacore::RowNumbers>();
  casacore::Vector<casacore::String> columns;
  try {
    const auto& table = proxy->table();
    if (table.isNull()) return Status::Invalid("Table instance is closed");
    if (table.isRootTable()) {
      // Not a selection over the root, e.g. a TAQL result computed into a
      // table of its own. It cannot be stale on account of the root.
      return Status::NotImplemented("Refreshing a derived table that is not a ",
                                    "selection over its source table");
    }
    *rows = table.rowNumbers();
    columns = table.tableDesc().columnNames();
  } catch (const std::exception& e) {
    return Status::Invalid("Error inspecting derived table instance: ", e.what());
  }

  // Dropping first releases this instance's hold on the stale root
  DropTable(*proxy);
  // Refreshes the root instance, if it is itself out of date
  ARROW_ASSIGN_OR_RAISE(auto root_lock,
                        root_->LockInstance(instance, CasaLockType::Read));
  return RebuildDerived(proxy, rows, columns, *root_lock->proxy);
}

std::size_t IsolatedTableProxy::GetInstance() const {
  using NumTasksType = decltype(ProxyAndPool::io_pool_->GetNumTasks());
  std::size_t instance = 0;
  NumTasksType num_tasks = std::numeric_limits<NumTasksType>::max();
  assert(proxy_pools_.size() > 0);

  for (std::size_t i = 0; i < proxy_pools_.size(); ++i) {
    const auto& pool = proxy_pools_[i].io_pool_;
    const auto pool_tasks = pool->GetNumTasks();
    if (pool_tasks < num_tasks) {
      instance = i;
      num_tasks = pool_tasks;
    }
  }

  return instance;
}

const std::shared_ptr<TableProxy>& IsolatedTableProxy::GetProxy(
    std::size_t instance) const {
  assert(instance < proxy_pools_.size());
  return proxy_pools_[instance].slot_->proxy;
}

const std::shared_ptr<ThreadPool>& IsolatedTableProxy::GetPool(
    std::size_t instance) const {
  assert(instance < proxy_pools_.size());
  return proxy_pools_[instance].io_pool_;
}

std::shared_ptr<IsolatedTableProxy> IsolatedTableProxy::SpawnWriter() {
  // Create an IsolatedTableProxy that serialises writes to a single
  // table instance (and thread).
  // A custom deleter that releases resources (proxies and pools)
  // that are actually managed by the parent ITP
  std::shared_ptr<IsolatedTableProxy> itp(new IsolatedTableProxy(), [](auto* p) {
    p->proxy_pools_.clear();
    p->dependencies_.clear();
    p->is_closed_ = true;
    delete p;
  });
  itp->dependencies_.emplace_back(shared_from_this());
  // Writes are pinned to the first instance. This keeps writes working
  // after non-syncable operations like AddColumns, and RefreshInstances(0)
  // relies on it: only instance 0 ever takes a write lock, so the siblings
  // it replaces never have table files to write out.
  auto instance = 0;  // GetInstance();
  itp->proxy_pools_.push_back(proxy_pools_[instance]);
  // The writer's instance is this ITP's instance 0, so it refreshes it the
  // same way. Instance 0 goes stale when another handle or process adds a
  // column; refreshing it is still safe, because lock() throws on the
  // mismatch before the writer has changed anything to flush.
  itp->reopen_ = reopen_;
  itp->root_ = root_;
  itp->is_closed_ = false;
  return itp;
}

Status IsolatedTableProxy::RefreshInstances(std::size_t except) {
  ARROW_RETURN_NOT_OK(CheckClosed());
  if (proxy_pools_.size() <= 1) return Status::OK();
  if (!CanRefresh()) {
    return Status::NotImplemented(
        "Refreshing the instances of this table after a structural change");
  }

  // Refresh one instance at a time, as Make() creates them. This only
  // follows a structural change, and the recipes' JSON parsing is
  // serialised in casacore anyway (002-json-parser-mutex.patch)
  for (std::size_t i = 0; i < proxy_pools_.size(); ++i) {
    if (i == except) continue;
    const auto& pp = proxy_pools_[i];
    // As in Close(): waiting on a task submitted to the pool running the
    // calling thread would deadlock, and running inline is what the
    // submission is there to guarantee
    if (pp.io_pool_->OwnsThisThread()) {
      ARROW_RETURN_NOT_OK(RefreshInstance(i));
      continue;
    }
    // Waited on below, so this outlives the task
    auto future = arrow::DeferNotOk(pp.io_pool_->Submit([this, i]() -> Result<bool> {
      ARROW_RETURN_NOT_OK(RefreshInstance(i));
      return true;
    }));
    ARROW_RETURN_NOT_OK(future.status());
  }
  return Status::OK();
}

bool IsolatedTableProxy::OwnsThisThread() const {
  for (const auto& pp : proxy_pools_) {
    if (pp.io_pool_->OwnsThisThread()) return true;
  }
  return false;
}

void IsolatedTableProxy::ReleasePools() {
  if (!OwnsThisThread()) return;

  // ThreadPool::Shutdown() waits for the pool's workers to exit, so destroying
  // a pool from one of its own workers parks that worker forever. Hand the
  // pools to the CPU pool for destruction instead. Ownership is transferred
  // only once the task is queued: a failed Spawn() destroys its argument on
  // the calling thread, which is what must be avoided here.
  auto* pools = new std::vector<ProxyAndPool>(std::move(proxy_pools_));
  auto status = GetCpuThreadPool()->Spawn([pools]() { delete pools; });

  if (!status.ok()) {
    // Only reachable once the CPU pool has shut down, i.e. during process
    // teardown, where leaking the pools beats hanging the calling thread.
    ARROW_LOG(WARNING) << "Unable to defer I/O pool destruction: " << status;
  }
}

Status IsolatedTableProxy::CheckClosed() const {
  if (!is_closed_) return Status::OK();
  return Status::Invalid("TableProxy is closed");
}

Result<bool> IsolatedTableProxy::Close() {
  if (is_closed_) return false;
  // Mark closed on scope exit, regardless of how the close tasks fare.
  std::shared_ptr<void> defer_close(nullptr, [this](...) { this->is_closed_ = true; });
  std::vector<Future<bool>> results;
  results.reserve(proxy_pools_.size());
  Status inline_status = Status::OK();
  // Let each instance finish what it is doing before closing it. Closing a
  // table is not a passive teardown: casacore's TableProxy::close() flushes,
  // and the flush takes a table lock of its own (keywordSet() ->
  // ColumnSet::userLock). Closing while this instance still has work in
  // flight therefore puts the close into a lock wait behind an operation it
  // should simply have waited for.
  for (auto& pp : proxy_pools_) {
    if (!pp.io_pool_->OwnsThisThread()) pp.io_pool_->WaitForIdle();
  }
  for (auto& pp : proxy_pools_) {
    const auto& pool = pp.io_pool_;
    // Submitting to a pool that owns the calling thread and then waiting on the
    // result deadlocks: the only worker able to run the close task is the
    // thread doing the waiting. This happens whenever the last reference to
    // this proxy is dropped on an isolation thread, as it is when the read and
    // write callbacks holding that reference are torn down once their pipeline
    // completes. Closing inline is correct there: being on the isolation
    // thread is all the submission is there to guarantee.
    if (pool->OwnsThisThread()) {
      auto status = CloseProxy(*pp.slot_->proxy);
      if (inline_status.ok()) inline_status = std::move(status);
      continue;
    }
    results.push_back(arrow::DeferNotOk(pool->Submit([slot = pp.slot_]() -> Result<bool> {
      ARROW_RETURN_NOT_OK(CloseProxy(*slot->proxy));
      return true;
    })));
  }
  auto all_done = arrow::All(results);
  ARROW_ASSIGN_OR_RAISE(auto outcomes, all_done.MoveResult());
  // Drain any late-scheduled continuations (e.g. Then callbacks) so that
  // members are not destroyed while a pool task may still reference them.
  // A pool running the calling thread is busy by definition and can only go
  // idle once this call returns.
  for (auto& pp : proxy_pools_) {
    if (!pp.io_pool_->OwnsThisThread()) pp.io_pool_->WaitForIdle();
  }
  // Surface the first close failure, if any (still leaving the proxy closed).
  ARROW_RETURN_NOT_OK(inline_status);
  for (const auto& outcome : outcomes) {
    ARROW_RETURN_NOT_OK(outcome.status());
  }
  return true;
}

IsolatedTableProxy::~IsolatedTableProxy() {
  auto result = Close();
  if (!result.ok()) {
    ARROW_LOG(WARNING) << "Error closing file " << result.status();
  }
  ReleasePools();
}

}  // namespace detail
}  // namespace arcae
