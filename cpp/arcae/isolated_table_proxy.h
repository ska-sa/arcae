#ifndef ARCAE_ISOLATED_TABLE_PROXY_H
#define ARCAE_ISOLATED_TABLE_PROXY_H

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
  // returning an arrow::Future<ReturnType>
  template <typename Fn, typename = std::enable_if_t<
                             std::is_invocable_v<Fn, const casacore::TableProxy&>>>
  ArrowFutureType<Fn, const casacore::TableProxy&> RunAsync(Fn&& functor) const {
    using ResultType = ArrowResultType<Fn, const casacore::TableProxy&>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto instance = GetInstance();
    return RunInPool([this, instance = instance,
                      functor = std::forward<Fn>(functor)]() mutable -> ResultType {
      try {
        return std::invoke(functor, *this->GetProxy(instance));
      } catch (casacore::AipsError& e) {
        return arrow::Status::Invalid("Unhandled casacore exception: ", e.what());
      }
    });
  }

  // Runs functions with signature
  // ReturnType Function(TableProxy &) on the isolation thread
  // returning an arrow::Future<ReturnType>
  template <typename Fn,
            typename = std::enable_if_t<std::is_invocable_v<Fn, casacore::TableProxy&>>>
  ArrowFutureType<Fn, casacore::TableProxy&> RunAsync(Fn&& functor) {
    using ResultType = ArrowFutureType<Fn, casacore::TableProxy&>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto instance = GetInstance();
    return RunInPool(instance,
                     [this, instance = instance,
                      functor = std::forward<Fn>(functor)]() mutable -> ResultType {
                       try {
                         return std::invoke(functor, *this->GetProxy(instance));
                       } catch (casacore::AipsError& e) {
                         return arrow::Status::Invalid("Unhandled casacore exception: ",
                                                       e.what());
                       }
                     });
  }

  template <typename Fn, typename R,
            typename = std::enable_if_t<
                std::is_invocable_v<Fn, const R&, const casacore::TableProxy&>>>
  ArrowFutureType<Fn, const R&, const casacore::TableProxy&> Then(
      arrow::Future<R>& future, Fn&& functor) const {
    using ResultType = ArrowFutureType<Fn, const R&, const casacore::TableProxy&>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto instance = GetInstance();
    return future.Then(
        [this, instance = instance,
         fn = std::forward<Fn>(functor)](const R& result) mutable -> ResultType {
          try {
            return std::invoke(fn, result, *this->GetProxy(instance));
          } catch (casacore::AipsError& e) {
            return arrow::Status::Invalid("Unhandled casacore exception: ", e.what());
          }
        },
        {},
        arrow::CallbackOptions{arrow::ShouldSchedule::Always,
                               this->GetPool(instance).get()});
  }

  template <typename Fn, typename R,
            typename = std::enable_if_t<
                std::is_invocable_v<Fn, const R&, casacore::TableProxy&>>>
  ArrowFutureType<Fn, const R&, casacore::TableProxy&> Then(arrow::Future<R>& future,
                                                            Fn&& functor) {
    using ResultType = ArrowFutureType<Fn, const R&, casacore::TableProxy&>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto instance = GetInstance();
    return future.Then(
        [this, instance = instance,
         fn = std::forward<Fn>(functor)](const R& result) mutable -> ResultType {
          try {
            return std::invoke(fn, result, *this->GetProxy(instance));
          } catch (casacore::AipsError& e) {
            return arrow::Status::Invalid("Unhandled casacore exception: ", e.what());
          }
        },
        {},
        arrow::CallbackOptions{arrow::ShouldSchedule::Always,
                               this->GetPool(instance).get()});
  }

  // Runs functions with signature
  // ReturnType Function(const TableProxy &) on the isolation thread
  // If ReturnType is not an arrow::Result, it will be converted
  // to an arrow::Result<ReturnType>
  template <typename Fn, typename = std::enable_if_t<
                             std::is_invocable_v<Fn, const casacore::TableProxy&>>>
  ArrowResultType<Fn, const casacore::TableProxy&> RunSync(Fn&& functor) const {
    using ResultType = ArrowFutureType<Fn, const casacore::TableProxy&>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto instance = GetInstance();
    return RunInPoolSync([this, instance = instance,
                          functor = std::forward<Fn>(functor)]() mutable -> ResultType {
      try {
        return std::invoke(functor, *this->GetProxy(instance));
      } catch (casacore::AipsError& e) {
        return arrow::Status::Invalid("Unhandled casacore exception: ", e.what());
      }
    });
  }

  // Runs functions with signature
  // ReturnType Function(TableProxy &) on the isolation thread
  // If ReturnType is not an arrow::Result, it will be converted
  // to an arrow::Result<ReturnType>
  template <typename Fn,
            typename = std::enable_if_t<std::is_invocable_v<Fn, casacore::TableProxy&>>>
  ArrowResultType<Fn, casacore::TableProxy&> RunSync(Fn&& functor) {
    using ResultType = ArrowResultType<Fn, casacore::TableProxy&>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto instance = GetInstance();
    return RunInPoolSync(instance,
                         [this, instance = instance,
                          functor = std::forward<Fn>(functor)]() mutable -> ResultType {
                           try {
                             return std::invoke(functor, *this->GetProxy(instance));
                           } catch (casacore::AipsError& e) {
                             return arrow::Status::Invalid(
                                 "Unhandled casacore exception: ", e.what());
                           }
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
    auto fwd_functor = std::forward<Fn>(functor);

    // Mark as closed so that if construction fails, we don't try to close it
    proxy->is_closed_ = true;

    // Create ninstances I/O pools
    for (std::size_t i = 0; i < ninstances; ++i) {
      ARROW_ASSIGN_OR_RAISE(auto io_pool, ::arrow::internal::ThreadPool::Make(1));
      auto table_fut = arrow::DeferNotOk(io_pool->Submit(fwd_functor));
      ARROW_ASSIGN_OR_RAISE(auto table_proxy, table_fut.MoveResult());
      proxy->proxy_pools_.push_back(
          ProxyAndPool{std::move(table_proxy), std::move(io_pool)});
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
            return std::invoke(fn, *GetProxy(i));
          }));

      ARROW_ASSIGN_OR_RAISE(auto table_proxy, future.MoveResult());
      itp->proxy_pools_.emplace_back(ProxyAndPool{std::move(table_proxy), GetPool(i)});
    }

    itp->is_closed_ = false;
    // Add an explicit dependency on the ITP
    itp->dependencies_.push_back(shared_from_this());
    return itp;
  }

  std::shared_ptr<casacore::TableProxy> Proxy() const { return nullptr; }

  std::size_t nInstances() const { return proxy_pools_.size(); }

 protected:
  IsolatedTableProxy() = default;
  IsolatedTableProxy(const IsolatedTableProxy& rhs) = delete;
  IsolatedTableProxy(IsolatedTableProxy&& rhs) = delete;
  IsolatedTableProxy& operator=(const IsolatedTableProxy& rhs) = delete;
  IsolatedTableProxy& operator=(IsolatedTableProxy&& rhs) = delete;

  // Gets the least active instance
  std::size_t GetInstance() const;

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
  // and wait for the future's result
  template <typename Fn>
  ArrowResultType<Fn> RunInPoolWait(std::size_t instance, Fn&& functor) {
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
  };

  std::vector<ProxyAndPool> proxy_pools_;
  bool is_closed_;
  std::vector<std::shared_ptr<IsolatedTableProxy>> dependencies_;
};

}  // namespace detail
}  // namespace arcae

#endif  // ARCAE_ISOLATED_TABLE_PROXY_H
