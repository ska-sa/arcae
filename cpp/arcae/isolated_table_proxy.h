#ifndef ARCAE_ISOLATED_TABLE_PROXY_H
#define ARCAE_ISOLATED_TABLE_PROXY_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <type_traits>
#include <vector>

#include <casacore/casa/Exceptions/Error.h>
#include <casacore/tables/Tables.h>
#include <casacore/tables/Tables/TableProxy.h>

#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/util/future.h>
#include <arrow/util/thread_pool.h>

#include "arcae/type_traits.h"

namespace arcae {
namespace detail {

// Cumulative scheduling statistics for one instance.
// Shared by every IsolatedTableProxy dispatching to the instance's pool,
// so that proxies spawned from a parent count against the same pool
// Submitting threads and the isolation thread update separate
// cache lines, so that recording statistics doesn't bounce lines between them
struct InstanceStats {
  // Updated by submitting threads
  alignas(64) std::atomic<std::int64_t> submitted{0};
  // Estimated bytes of I/O of submitted, but incomplete, tasks.
  // Also released by the isolation thread as tasks complete
  std::atomic<std::int64_t> outstanding_cost{0};
  // Updated by the isolation thread
  alignas(64) std::atomic<std::int64_t> completed{0};
  // Time from submission until a task starts running
  std::atomic<std::int64_t> wait_ns{0};
  // Time spent running tasks
  std::atomic<std::int64_t> busy_ns{0};
  // Bytes read from or written to the table
  std::atomic<std::int64_t> io_bytes{0};
};

// Point in time copy of InstanceStats
struct InstanceStatsSnapshot {
  std::int64_t submitted;
  std::int64_t completed;
  std::int64_t wait_ns;
  std::int64_t busy_ns;
  std::int64_t io_bytes;
};

// A task running on an isolation thread
struct RunningTask {
  InstanceStats* stats = nullptr;
  // Estimated bytes of I/O reserved by the task
  std::int64_t cost = 0;
};

// Estimated cost of a task whose I/O is unknown, such as reading metadata,
// or partitioning a read, which may read small columns itself
inline constexpr std::int64_t kDefaultTaskCost = 1024 * 1024;

// Attribute nbytes of I/O to the instance running the calling task.
// Does nothing outside of a task dispatched by an IsolatedTableProxy
void RecordIoBytes(std::int64_t nbytes);

// Isolates access to a CASA Table to a single thread
class IsolatedTableProxy : public std::enable_shared_from_this<IsolatedTableProxy> {
 public:
  // Close the IsolatedTableProxy
  arrow::Result<bool> Close();
  // Is the IsolatedTableProxy closed?
  bool IsClosed() const;
  // Is the calling thread one of the isolation threads owned by this proxy?
  bool OwnsThisThread() const;
  // Return a failed status code if the table is closed
  arrow::Status CheckClosed() const;
  // Destroy the IsolatedTableProxy, attempting
  // to close the encapsulated TableProxy in the process
  virtual ~IsolatedTableProxy();

  // Runs function with signature
  // ReturnType Function(const TableProxy &) on the isolation thread
  // returning an arrow::Future<ReturnType>.
  // cost estimates the bytes of I/O performed by the function
  template <typename Fn, typename = std::enable_if_t<
                             std::is_invocable_v<Fn, const casacore::TableProxy&>>>
  ArrowFutureType<Fn, const casacore::TableProxy&> RunAsync(
      Fn&& functor, std::int64_t cost = kDefaultTaskCost) const {
    using ResultType = ArrowResultType<Fn, const casacore::TableProxy&>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto submission = Dispatch(cost);
    return RunInPool(submission.instance,
                     [this, submission = submission,
                      functor = std::forward<Fn>(functor)]() mutable -> ResultType {
                       return InvokeOnInstance<ResultType>(submission, functor);
                     });
  }

  // Runs functions with signature
  // ReturnType Function(TableProxy &) on the isolation thread
  // returning an arrow::Future<ReturnType>.
  // cost estimates the bytes of I/O performed by the function
  template <typename Fn,
            typename = std::enable_if_t<std::is_invocable_v<Fn, casacore::TableProxy&>>>
  ArrowFutureType<Fn, casacore::TableProxy&> RunAsync(
      Fn&& functor, std::int64_t cost = kDefaultTaskCost) {
    using ResultType = ArrowFutureType<Fn, casacore::TableProxy&>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto submission = Dispatch(cost);
    return RunInPool(submission.instance,
                     [this, submission = submission,
                      functor = std::forward<Fn>(functor)]() mutable -> ResultType {
                       return InvokeOnInstance<ResultType>(submission, functor);
                     });
  }

  // Runs functions with signature
  // ReturnType Function(const TableProxy &) on the isolation thread
  // If ReturnType is not an arrow::Result, it will be converted
  // to an arrow::Result<ReturnType>.
  // cost estimates the bytes of I/O performed by the function
  template <typename Fn, typename = std::enable_if_t<
                             std::is_invocable_v<Fn, const casacore::TableProxy&>>>
  ArrowResultType<Fn, const casacore::TableProxy&> RunSync(
      Fn&& functor, std::int64_t cost = kDefaultTaskCost) const {
    using ResultType = ArrowResultType<Fn, const casacore::TableProxy&>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto submission = Dispatch(cost);
    return RunInPoolSync(submission.instance,
                         [this, submission = submission,
                          functor = std::forward<Fn>(functor)]() mutable -> ResultType {
                           return InvokeOnInstance<ResultType>(submission, functor);
                         });
  }

  // Runs functions with signature
  // ReturnType Function(TableProxy &) on the isolation thread
  // If ReturnType is not an arrow::Result, it will be converted
  // to an arrow::Result<ReturnType>.
  // cost estimates the bytes of I/O performed by the function
  template <typename Fn,
            typename = std::enable_if_t<std::is_invocable_v<Fn, casacore::TableProxy&>>>
  ArrowResultType<Fn, casacore::TableProxy&> RunSync(
      Fn&& functor, std::int64_t cost = kDefaultTaskCost) {
    using ResultType = ArrowResultType<Fn, casacore::TableProxy&>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto submission = Dispatch(cost);
    return RunInPoolSync(submission.instance,
                         [this, submission = submission,
                          functor = std::forward<Fn>(functor)]() mutable -> ResultType {
                           return InvokeOnInstance<ResultType>(submission, functor);
                         });
  }

  // Construct an IsolatedTableProxy with the supplied function
  template <
      typename Fn,
      typename = std::enable_if<std::is_same_v<
          ArrowResultType<Fn>, arrow::Result<std::shared_ptr<casacore::TableProxy>>>>>
  static arrow::Result<std::shared_ptr<IsolatedTableProxy>> Make(
      Fn&& functor, std::size_t ninstances = 1) {
    if (ninstances < 1) {
      return arrow::Status::Invalid("Number of instances must at least be 1");
    }

    struct enable_make_shared_itp : public IsolatedTableProxy {};
    std::shared_ptr<IsolatedTableProxy> proxy =
        std::make_shared<enable_make_shared_itp>();
    proxy->proxy_pools_.reserve(ninstances);
    using ResultType = arrow::Result<std::shared_ptr<casacore::TableProxy>>;
    auto fwd_functor = std::forward<Fn>(functor);

    // Mark as closed so that if construction fails, we don't try to close it
    proxy->is_closed_ = true;

    // Create ninstances I/O pools
    for (std::size_t i = 0; i < ninstances; ++i) {
      ARROW_ASSIGN_OR_RAISE(auto io_pool, ::arrow::internal::ThreadPool::Make(1));
      auto table_fut =
          arrow::DeferNotOk(io_pool->Submit([fn = fwd_functor]() mutable -> ResultType {
            return Invoke<ResultType>(Submission{}, fn);
          }));
      ARROW_ASSIGN_OR_RAISE(auto table_proxy, table_fut.MoveResult());
      proxy->proxy_pools_.push_back(ProxyAndPool{
          std::move(table_proxy), std::move(io_pool), std::make_shared<InstanceStats>()});
    }

    proxy->is_closed_ = false;
    return proxy;
  }

  // Construct a new IsolatedTableProxy with the supplied function
  // which is dependent on this IsolatedTableProxy.
  // This generally exists to create Reference Tables through
  // for e.g. Taql queries
  template <typename Fn,
            typename = std::enable_if<
                std::is_invocable_v<Fn, const casacore::TableProxy&> &&
                std::is_same_v<ArrowResultType<Fn, const casacore::TableProxy&>,
                               arrow::Result<std::shared_ptr<casacore::TableProxy>>>>>
  arrow::Result<std::shared_ptr<IsolatedTableProxy>> Spawn(Fn&& functor) {
    struct enable_make_shared_itp : public IsolatedTableProxy {};
    std::shared_ptr<IsolatedTableProxy> itp = std::make_shared<enable_make_shared_itp>();
    using ResultType = arrow::Result<std::shared_ptr<casacore::TableProxy>>;

    // Mark as closed so that if construction fails, we don't try to close it
    itp->is_closed_ = true;
    auto fwd_functor = std::forward<Fn>(functor);

    for (std::size_t i = 0; i < proxy_pools_.size(); ++i) {
      auto future = arrow::DeferNotOk(
          GetPool(i)->Submit([this, i = i, fn = fwd_functor]() -> ResultType {
            return Invoke<ResultType>(Submission{}, fn, *GetProxy(i));
          }));

      ARROW_ASSIGN_OR_RAISE(auto table_proxy, future.MoveResult());
      // Share the parent's statistics, as the spawned proxy runs on the same pool
      itp->proxy_pools_.emplace_back(
          ProxyAndPool{std::move(table_proxy), GetPool(i), proxy_pools_[i].stats_});
    }

    itp->is_closed_ = false;
    // Add an explicit dependency on the ITP
    itp->dependencies_.push_back(shared_from_this());
    return itp;
  }

  std::shared_ptr<casacore::TableProxy> Proxy() const { return nullptr; }

  std::size_t nInstances() const { return proxy_pools_.size(); }

  // Return a snapshot of each instance's scheduling statistics
  std::vector<InstanceStatsSnapshot> GetInstanceStats() const;

 protected:
  IsolatedTableProxy() = default;
  IsolatedTableProxy(const IsolatedTableProxy& rhs) = delete;
  IsolatedTableProxy(IsolatedTableProxy&& rhs) = delete;
  IsolatedTableProxy& operator=(const IsolatedTableProxy& rhs) = delete;
  IsolatedTableProxy& operator=(IsolatedTableProxy&& rhs) = delete;

  // A task submitted to an instance.
  // stats is null for tasks that are not measured
  struct Submission {
    std::size_t instance = 0;
    InstanceStats* stats = nullptr;
    std::chrono::steady_clock::time_point time;
    // Estimated bytes of I/O, reserved on the instance until the task completes
    std::int64_t cost = 0;
  };

  // Choose the instance with the least outstanding I/O for a task of the given
  // cost, and reserve the cost on it. Ties go to the lowest numbered instance,
  // which keeps successive calls on an idle proxy on one instance, whose caches
  // are warm. Spreading ties over the instances slowed small sequential reads.
  // The reservation is released when the task runs, so a task dropped by a
  // shut down pool leaves it in place: the proxy is closing in that case
  Submission Dispatch(std::int64_t cost) const;

  // Measures a task while it runs on an isolation thread,
  // attributing any I/O it records to the task's instance
  class TaskScope {
   public:
    explicit TaskScope(const Submission& submission);
    ~TaskScope();
    TaskScope(const TaskScope&) = delete;
    TaskScope& operator=(const TaskScope&) = delete;

   private:
    InstanceStats* stats_;
    std::int64_t cost_;
    // The task this one is nested in, if any
    RunningTask previous_;
    std::chrono::steady_clock::time_point start_;
  };

  // Invoke the functor on the submitted instance's table within a TaskScope.
  // Tasks dispatched while Close() waits are queued behind the close, so fail
  // them if the table has been closed: casacore answers some queries on a
  // closed table instead of failing them, e.g. reporting zero rows.
  // This runs on the isolation thread that closes the table, so can't race it
  template <typename ResultType, typename Fn>
  ResultType InvokeOnInstance(const Submission& submission, Fn& functor) const {
    auto checked = [&functor](casacore::TableProxy& tp) -> ResultType {
      if (tp.table().isNull()) return arrow::Status::Invalid("TableProxy is closed");
      return std::invoke(functor, tp);
    };
    return Invoke<ResultType>(submission, checked, *GetProxy(submission.instance));
  }

  // Invoke the functor within a TaskScope,
  // converting exceptions into a failed status
  template <typename ResultType, typename Fn, typename... Args>
  static ResultType Invoke(const Submission& submission, Fn& functor, Args&&... args) {
    TaskScope scope(submission);
    try {
      return std::invoke(functor, std::forward<Args>(args)...);
    } catch (casacore::AipsError& e) {
      return arrow::Status::Invalid("Unhandled casacore exception: ", e.what());
    } catch (std::exception& e) {
      return arrow::Status::Invalid("Unhandled exception: ", e.what());
    }
  }

  // Get the Table Proxy for the given instance
  const std::shared_ptr<casacore::TableProxy>& GetProxy(std::size_t instance) const;

  // Get the I/O pool for the given instance
  const std::shared_ptr<arrow::internal::ThreadPool>& GetPool(std::size_t instance) const;

  // Relinquish the I/O pools, deferring their destruction to another thread
  // if the calling thread belongs to one of them
  void ReleasePools();

  // Run the given functor in the I/O pool
  // and wait for the future's result
  template <typename Fn>
  ArrowResultType<Fn> RunInPoolSync(std::size_t instance, Fn&& functor) const {
    return RunInPool(instance, std::forward<Fn>(functor)).MoveResult();
  }

  // Run the given functor in the I/O pool
  // returning an future to the result
  template <typename Fn>
  ArrowFutureType<Fn> RunInPool(std::size_t instance, Fn&& functor) const {
    const auto& pool = proxy_pools_[instance].io_pool_;
    return arrow::DeferNotOk(pool->Submit(std::forward<Fn>(functor)));
  }

  // Run the given functor in the I/O pool
  // returning a future to the result
  template <typename Fn>
  ArrowFutureType<Fn> RunInPool(std::size_t instance, Fn&& functor) {
    const auto& pool = proxy_pools_[instance].io_pool_;
    return arrow::DeferNotOk(pool->Submit(std::forward<Fn>(functor)));
  }

 private:
  struct ProxyAndPool {
    std::shared_ptr<casacore::TableProxy> table_proxy_;
    std::shared_ptr<arrow::internal::ThreadPool> io_pool_;
    std::shared_ptr<InstanceStats> stats_;
  };

  std::vector<ProxyAndPool> proxy_pools_;
  // Read by dispatching threads while Close() runs
  std::atomic<bool> is_closed_;
  std::vector<std::shared_ptr<IsolatedTableProxy>> dependencies_;
};

}  // namespace detail
}  // namespace arcae

#endif  // ARCAE_ISOLATED_TABLE_PROXY_H
