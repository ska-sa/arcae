#include "arcae/isolated_table_proxy.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
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

namespace {

using Clock = std::chrono::steady_clock;

// The task running on this thread, if any
thread_local RunningTask current_task;

std::int64_t ElapsedNs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

}  // namespace

void RecordIoBytes(std::int64_t nbytes) {
  if (auto* stats = current_task.stats) {
    stats->io_bytes.fetch_add(nbytes, std::memory_order_relaxed);
  }
}

IsolatedTableProxy::Submission IsolatedTableProxy::Dispatch(std::int64_t cost) const {
  assert(proxy_pools_.size() > 0);
  constexpr auto relaxed = std::memory_order_relaxed;
  std::size_t instance = 0;
  auto least = std::numeric_limits<std::int64_t>::max();

  for (std::size_t i = 0; i < proxy_pools_.size(); ++i) {
    auto* stats = proxy_pools_[i].stats_.get();
    auto load = stats->outstanding_cost.load(relaxed);
    // A task dispatching work from its isolation thread is about to complete
    if (stats == current_task.stats) load -= current_task.cost;
    if (load < least) {
      instance = i;
      least = load;
    }
  }

  // Concurrent dispatches may choose the same instance before either
  // reserves its cost. This only costs balance, and is rare
  auto* stats = proxy_pools_[instance].stats_.get();
  stats->outstanding_cost.fetch_add(cost, relaxed);
  stats->submitted.fetch_add(1, relaxed);
  return Submission{instance, stats, Clock::now(), cost};
}

IsolatedTableProxy::TaskScope::TaskScope(const Submission& submission)
    : stats_(submission.stats), cost_(submission.cost), previous_(current_task) {
  if (!stats_) return;
  start_ = Clock::now();
  stats_->wait_ns.fetch_add(ElapsedNs(submission.time, start_),
                            std::memory_order_relaxed);
  current_task = {stats_, cost_};
}

IsolatedTableProxy::TaskScope::~TaskScope() {
  if (!stats_) return;
  current_task = previous_;
  // Release the reservation before the task's future completes, so that
  // continuations dispatching further work see this instance as free
  stats_->outstanding_cost.fetch_sub(cost_, std::memory_order_relaxed);
  stats_->busy_ns.fetch_add(ElapsedNs(start_, Clock::now()), std::memory_order_relaxed);
  stats_->completed.fetch_add(1, std::memory_order_relaxed);
}

std::vector<InstanceStatsSnapshot> IsolatedTableProxy::GetInstanceStats() const {
  std::vector<InstanceStatsSnapshot> result;
  result.reserve(proxy_pools_.size());
  for (const auto& [proxy, pool, stats] : proxy_pools_) {
    constexpr auto relaxed = std::memory_order_relaxed;
    result.push_back(InstanceStatsSnapshot{
        stats->submitted.load(relaxed), stats->completed.load(relaxed),
        stats->wait_ns.load(relaxed), stats->busy_ns.load(relaxed),
        stats->io_bytes.load(relaxed)});
  }
  return result;
}

bool IsolatedTableProxy::IsClosed() const { return is_closed_; }

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
  for (const auto& [proxy, pool, stats] : proxy_pools_) {
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
    for (auto& [proxy, pool, stats] : proxy_pools_) {
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
      results.push_back(arrow::DeferNotOk(pool->Submit([tp = proxy]() -> Result<bool> {
        // An exception escaping a pool thread terminates the process
        try {
          tp->close();
        } catch (const std::exception& e) {
          return Status::Invalid("Error closing table: ", e.what());
        }
        return true;
      })));
    }
    auto all_done = arrow::All(results);
    all_done.Wait();
    ARROW_RETURN_NOT_OK(inline_status);
    for (const auto& result : all_done.result().ValueOrDie()) {
      ARROW_RETURN_NOT_OK(result.status());
    }
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
