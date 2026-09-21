#include "arcae/isolated_table_proxy.h"

#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>

#include <arrow/status.h>
#include <arrow/util/future.h>
#include <arrow/util/logging.h>
#include <arrow/util/thread_pool.h>

#include <casacore/tables/Tables/TableProxy.h>

using ::arrow::Future;
using ::arrow::Result;
using ::arrow::Status;
using ::arrow::internal::GetCpuThreadPool;
using ::arrow::internal::ThreadPool;
using ::casacore::TableProxy;

namespace arcae {
namespace detail {

bool IsolatedTableProxy::IsClosed() const { return is_closed_; }

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
  return proxy_pools_[instance].table_proxy_;
}

const std::shared_ptr<ThreadPool>& IsolatedTableProxy::GetPool(
    std::size_t instance) const {
  assert(instance < proxy_pools_.size());
  return proxy_pools_[instance].io_pool_;
}

bool IsolatedTableProxy::OwnsThisThread() const {
  for (const auto& [proxy, pool] : proxy_pools_) {
    if (pool->OwnsThisThread()) return true;
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
  if (!is_closed_) {
    std::shared_ptr<void> defer_close(nullptr, [this](...) { this->is_closed_ = true; });
    std::vector<Future<bool>> results;
    results.reserve(proxy_pools_.size());
    Status inline_status = Status::OK();
    for (auto& [proxy, pool] : proxy_pools_) {
      // Submitting to a pool that owns the calling thread and then waiting on
      // the result deadlocks: the only worker able to run the close task is the
      // thread doing the waiting. This happens whenever the last reference to
      // this proxy is dropped on an isolation thread, as it is when the
      // callbacks holding that reference are torn down once their pipeline
      // completes. Closing inline is correct there: being on the isolation
      // thread is all the submission is there to guarantee.
      if (pool->OwnsThisThread()) {
        // close() may throw casacore::AipsError, and Close() is called from the
        // destructor, so the exception must not be allowed to escape.
        try {
          proxy->close();
        } catch (const std::exception& e) {
          inline_status = Status::Invalid("Error closing table: ", e.what());
        }
        continue;
      }
      results.push_back(arrow::DeferNotOk(pool->Submit([tp = proxy]() {
        tp->close();
        return true;
      })));
    }
    auto all_done = arrow::All(results);
    all_done.Wait();
    ARROW_RETURN_NOT_OK(inline_status);
    return true;
  }
  return false;
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
