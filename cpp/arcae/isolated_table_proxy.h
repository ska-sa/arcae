#ifndef ARCAE_ISOLATED_TABLE_PROXY_H
#define ARCAE_ISOLATED_TABLE_PROXY_H

#include <atomic>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <casacore/casa/Exceptions/Error.h>
#include <casacore/casa/IO/FileLocker.h>
#include <casacore/tables/Tables.h>
#include <casacore/tables/Tables/TableProxy.h>

#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/util/future.h>
#include <arrow/util/logging.h>
#include <arrow/util/thread_pool.h>

#include "arcae/type_traits.h"

namespace arcae {
namespace detail {

using CasaLockType = casacore::FileLocker::LockType;
using CasaTableProxy = casacore::TableProxy;
using ConstTableProxyRef = const casacore::TableProxy&;
using TableProxyRef = casacore::TableProxy&;

// Opens an existing table afresh on the calling isolation thread,
// given its name and whether it should be writable. Used to replace an
// instance whose column set is out of date (see RefreshInstances)
using ReopenFn = std::function<arrow::Result<std::shared_ptr<CasaTableProxy>>(
    const std::string& name, bool writable)>;

// Is this the error casacore raises when a table instance's column set
// no longer matches the table on disk? casacore fixes an open table's
// columns for the lifetime of the object, so the instance must be reopened.
bool IsColumnCountMismatch(const std::exception& e);

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

  struct MaybeLockAndFinalise {
    std::shared_ptr<CasaTableProxy> proxy;
    CasaLockType lock_type;
    bool locked = false;
    // Set when the resync found the instance's column set out of date
    bool stale = false;

    MaybeLockAndFinalise(std::shared_ptr<CasaTableProxy> proxy_, CasaLockType lock_type_)
        : proxy(std::move(proxy_)), lock_type(lock_type_) {
      if (lock_type != CasaLockType::None) {
        // Honour the acquisition result: TableProxy::lock returns whether
        // the lock was taken. Throwing here is caught by the AipsError
        // handler in the dispatching task and converted to Status::Invalid,
        // so we never run the wrapped functor without a lock.
        try {
          locked = proxy->lock(lock_type == CasaLockType::Write, 0);
        } catch (...) {
          // PlainTable::lock can throw with the lock already taken: it syncs
          // once it has acquired, and a column count mismatch throws from
          // there -- which is what a reader sees after another handle added
          // a column (ska-sa/arcae#241). This object is never constructed,
          // so its destructor will never run and release that lock. Left
          // held it is held for the life of the process, and since the lock
          // state is shared process wide, every later writer waits on it.
          Unlock();
          throw;
        }
        if (!locked) throw casacore::AipsError("Failed to acquire table lock");

        if (lock_type == CasaLockType::Read) {
          // Acquiring a read lock is not enough to see another handle's
          // writes. casacore resyncs on lock acquisition by way of
          // ColumnSet::resync(nrow, forceSync=false), which only revisits
          // data managers whose change counter moved, and that misses
          // array column storage: a reader observes writes to scalar
          // columns but keeps returning stale array data indefinitely.
          // Table::resync() is the forceSync=true path and does see them.
          //
          // This matters here in a way it does not for single threaded
          // casacore users, because arcae's table cache is thread_local:
          // two handles in one process are genuinely independent tables,
          // so they take the same code path as two processes.
          try {
            proxy->resync();
          } catch (const casacore::AipsError& e) {
            // A resync is an optimisation over what the lock already did, so
            // failing it is not a reason to fail the operation. In
            // particular, Table::resync always compares the column count and
            // throws when it differs, while Table::lock only compares it on
            // the occasions it decides to sync. No resync can take in a
            // column added to an open table (ska-sa/arcae#241): the instance
            // must be reopened. Flag it as stale so that LockInstance can do
            // so; where it cannot, the read proceeds against the columns
            // this instance already knows about.
            //
            // The lock is deliberately kept: we hold it, and the destructor
            // that releases it only runs because we do not throw here.
            ARROW_LOG(DEBUG) << "Unable to resync table: " << e.what();
            stale = IsColumnCountMismatch(e);
          }
        }
      }
    }
    ~MaybeLockAndFinalise() {
      // Never let an exception escape the destructor: it can run during
      // stack unwinding of a functor that already threw, and a second
      // in-flight exception would call std::terminate.
      if (!locked) return;
      if (lock_type == CasaLockType::Write) {
        try {
          proxy->flush(false);
        } catch (const std::exception& e) {
          ARROW_LOG(WARNING) << "Error flushing table: " << e.what();
        }
      }
      // Unlock in its own try: a throwing flush must not cost us the
      // release, for the same reason as above.
      Unlock();
    }

   private:
    // Release the lock, swallowing any error. Used from both the
    // constructor's failure path and the destructor.
    void Unlock() noexcept {
      try {
        proxy->unlock();
      } catch (const std::exception& e) {
        ARROW_LOG(WARNING) << "Error unlocking table: " << e.what();
      } catch (...) {
        ARROW_LOG(WARNING) << "Error unlocking table";
      }
    }
  };

  // Runs function with signature
  // ReturnType Function(const TableProxy &) on the isolation thread
  // returning an arrow::Future<ReturnType>
  template <typename Fn,
            typename = std::enable_if_t<std::is_invocable_v<Fn, ConstTableProxyRef>>>
  ArrowFutureType<Fn, ConstTableProxyRef> RunAsync(
      Fn&& functor, CasaLockType lock_type = CasaLockType::Read) const {
    using ResultType = ArrowResultType<Fn, ConstTableProxyRef>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto instance = GetInstance();
    return RunInPool(
        instance,
        [weak_self = weak_from_this(), instance = instance, lock_type = lock_type,
         functor = std::forward<Fn>(functor)]() mutable -> ResultType {
          try {
            auto self = weak_self.lock();
            if (!self) return arrow::Status::Invalid("TableProxy is closed");
            ARROW_ASSIGN_OR_RAISE(auto lock, self->LockInstance(instance, lock_type));
            return std::invoke(functor, *lock->proxy);
          } catch (casacore::AipsError& e) {
            return arrow::Status::Invalid("Unhandled casacore exception: ", e.what());
          } catch (std::runtime_error& e) {
            return arrow::Status::Invalid("Unhandled exception: ", e.what());
          }
        });
  }

  // Runs functions with signature
  // ReturnType Function(TableProxy &) on the isolation thread
  // returning an arrow::Future<ReturnType>
  template <typename Fn,
            typename = std::enable_if_t<std::is_invocable_v<Fn, TableProxyRef>>>
  ArrowFutureType<Fn, TableProxyRef> RunAsync(
      Fn&& functor, CasaLockType lock_type = CasaLockType::Read) {
    using ResultType = ArrowFutureType<Fn, TableProxyRef>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto instance = GetInstance();
    return RunInPool(
        instance,
        [weak_self = weak_from_this(), instance = instance, lock_type = lock_type,
         functor = std::forward<Fn>(functor)]() mutable -> ResultType {
          try {
            auto self = weak_self.lock();
            if (!self) return arrow::Status::Invalid("TableProxy is closed");
            ARROW_ASSIGN_OR_RAISE(auto lock, self->LockInstance(instance, lock_type));
            return std::invoke(functor, *lock->proxy);
          } catch (casacore::AipsError& e) {
            return arrow::Status::Invalid("Unhandled casacore exception: ", e.what());
          } catch (std::runtime_error& e) {
            return arrow::Status::Invalid("Unhandled exception: ", e.what());
          }
        });
  }

  template <
      typename Fn, typename R,
      typename = std::enable_if_t<std::is_invocable_v<Fn, const R&, ConstTableProxyRef>>>
  ArrowFutureType<Fn, const R&, ConstTableProxyRef> Then(
      arrow::Future<R>& future, Fn&& functor,
      CasaLockType lock_type = CasaLockType::Read) const {
    using ResultType = ArrowFutureType<Fn, const R&, ConstTableProxyRef>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto instance = GetInstance();
    return future.Then(
        [weak_self = weak_from_this(), instance = instance, lock_type = lock_type,
         fn = std::forward<Fn>(functor)](const R& result) mutable -> ResultType {
          try {
            auto self = weak_self.lock();
            if (!self) return arrow::Status::Invalid("TableProxy is closed");
            ARROW_ASSIGN_OR_RAISE(auto lock, self->LockInstance(instance, lock_type));
            return std::invoke(fn, result, *lock->proxy);
          } catch (casacore::AipsError& e) {
            return arrow::Status::Invalid("Unhandled casacore exception: ", e.what());
          } catch (std::runtime_error& e) {
            return arrow::Status::Invalid("Unhandled exception: ", e.what());
          }
        },
        {},
        arrow::CallbackOptions{arrow::ShouldSchedule::Always,
                               this->GetPool(instance).get()});
  }

  template <typename Fn, typename R,
            typename = std::enable_if_t<std::is_invocable_v<Fn, const R&, TableProxyRef>>>
  ArrowFutureType<Fn, const R&, TableProxyRef> Then(
      arrow::Future<R>& future, Fn&& functor,
      CasaLockType lock_type = CasaLockType::Read) {
    using ResultType = ArrowFutureType<Fn, const R&, TableProxyRef>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto instance = GetInstance();
    return future.Then(
        [weak_self = weak_from_this(), instance = instance, lock_type = lock_type,
         fn = std::forward<Fn>(functor)](const R& result) mutable -> ResultType {
          try {
            auto self = weak_self.lock();
            if (!self) return arrow::Status::Invalid("TableProxy is closed");
            ARROW_ASSIGN_OR_RAISE(auto lock, self->LockInstance(instance, lock_type));
            return std::invoke(fn, result, *lock->proxy);
          } catch (casacore::AipsError& e) {
            return arrow::Status::Invalid("Unhandled casacore exception: ", e.what());
          } catch (std::runtime_error& e) {
            return arrow::Status::Invalid("Unhandled exception: ", e.what());
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
  template <typename Fn,
            typename = std::enable_if_t<std::is_invocable_v<Fn, ConstTableProxyRef>>>
  ArrowResultType<Fn, ConstTableProxyRef> RunSync(
      Fn&& functor, CasaLockType lock_type = CasaLockType::Read) const {
    using ResultType = ArrowFutureType<Fn, ConstTableProxyRef>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto instance = GetInstance();
    return RunInPoolSync(
        instance,
        [weak_self = weak_from_this(), instance = instance, lock_type = lock_type,
         functor = std::forward<Fn>(functor)]() mutable -> ResultType {
          try {
            auto self = weak_self.lock();
            if (!self) return arrow::Status::Invalid("TableProxy is closed");
            ARROW_ASSIGN_OR_RAISE(auto lock, self->LockInstance(instance, lock_type));
            return std::invoke(functor, *lock->proxy);
          } catch (casacore::AipsError& e) {
            return arrow::Status::Invalid("Unhandled casacore exception: ", e.what());
          } catch (std::runtime_error& e) {
            return arrow::Status::Invalid("Unhandled exception: ", e.what());
          }
        });
  }

  // Runs functions with signature
  // ReturnType Function(TableProxy &) on the isolation thread
  // If ReturnType is not an arrow::Result, it will be converted
  // to an arrow::Result<ReturnType>
  template <typename Fn,
            typename = std::enable_if_t<std::is_invocable_v<Fn, TableProxyRef>>>
  ArrowResultType<Fn, TableProxyRef> RunSync(
      Fn&& functor, CasaLockType lock_type = CasaLockType::Read) {
    using ResultType = ArrowResultType<Fn, TableProxyRef>;
    ARROW_RETURN_NOT_OK(CheckClosed());
    auto instance = GetInstance();
    return RunInPoolSync(
        instance,
        [weak_self = weak_from_this(), instance = instance, lock_type = lock_type,
         functor = std::forward<Fn>(functor)]() mutable -> ResultType {
          try {
            auto self = weak_self.lock();
            if (!self) return arrow::Status::Invalid("TableProxy is closed");
            ARROW_ASSIGN_OR_RAISE(auto lock, self->LockInstance(instance, lock_type));
            return std::invoke(functor, *lock->proxy);
          } catch (casacore::AipsError& e) {
            return arrow::Status::Invalid("Unhandled casacore exception: ", e.what());
          } catch (std::runtime_error& e) {
            return arrow::Status::Invalid("Unhandled exception: ", e.what());
          }
        });
  }

  // Construct an IsolatedTableProxy with the supplied function
  template <typename Fn,
            typename = std::enable_if<std::is_same_v<
                ArrowResultType<Fn>, arrow::Result<std::shared_ptr<CasaTableProxy>>>>>
  // reopen, if supplied, allows instances to be refreshed after a
  // structural change such as AddColumns. It cannot be functor itself:
  // for a table that functor creates, calling it again recreates the table.
  static arrow::Result<std::shared_ptr<IsolatedTableProxy>> Make(
      Fn&& functor, std::size_t ninstances = 1, ReopenFn reopen = nullptr) {
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
    proxy->reopen_ = std::move(reopen);

    // Create ninstances I/O pools
    for (std::size_t i = 0; i < ninstances; ++i) {
      ARROW_ASSIGN_OR_RAISE(auto io_pool, ::arrow::internal::ThreadPool::Make(1));
      auto table_fut = arrow::DeferNotOk(io_pool->Submit(fwd_functor));
      ARROW_ASSIGN_OR_RAISE(auto table_proxy, table_fut.MoveResult());
      proxy->proxy_pools_.push_back(
          ProxyAndPool{std::make_shared<ProxySlot>(ProxySlot{std::move(table_proxy)}),
                       std::move(io_pool)});
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
                std::is_invocable_v<Fn, ConstTableProxyRef> &&
                std::is_same_v<ArrowResultType<Fn, ConstTableProxyRef>,
                               arrow::Result<std::shared_ptr<CasaTableProxy>>>>>
  arrow::Result<std::shared_ptr<IsolatedTableProxy>> Spawn(Fn&& functor) {
    struct enable_make_shared_itp : public IsolatedTableProxy {};
    std::shared_ptr<IsolatedTableProxy> itp = std::make_shared<enable_make_shared_itp>();
    using ResultType = arrow::Result<std::shared_ptr<CasaTableProxy>>;

    // Mark as closed so that if construction fails, we don't try to close it
    itp->is_closed_ = true;
    auto fwd_functor = std::forward<Fn>(functor);

    for (std::size_t i = 0; i < proxy_pools_.size(); ++i) {
      auto future = arrow::DeferNotOk(
          GetPool(i)->Submit([this, i = i, fn = fwd_functor]() -> ResultType {
            // Hold a read lock on the source proxy while the functor runs.
            // Under user locking, casacore requires the source table to be
            // locked while e.g. a TAQL command builds a reference table from it.
            ARROW_ASSIGN_OR_RAISE(auto lock, this->LockInstance(i, CasaLockType::Read));
            return std::invoke(fn, *lock->proxy);
          }));

      ARROW_ASSIGN_OR_RAISE(auto table_proxy, future.MoveResult());
      itp->proxy_pools_.emplace_back(ProxyAndPool{
          std::make_shared<ProxySlot>(ProxySlot{std::move(table_proxy)}), GetPool(i)});
    }

    itp->is_closed_ = false;
    // Add an explicit dependency on the ITP
    itp->dependencies_.push_back(shared_from_this());
    // Derived instances are rebuilt over the root's, see RefreshInstance
    itp->root_ = root_ ? root_ : shared_from_this();
    return itp;
  }

  // Spawns an IsolatedTableProxy encapsulating a single instance
  // from this ITP. Suitable for constraining writes to a single
  // thread and instance as concurrent writes issued from multiple
  // threads will produce race conditions in the underlying casacore layer
  std::shared_ptr<IsolatedTableProxy> SpawnWriter();

  // Replace every instance except `except` with a freshly opened one.
  //
  // casacore fixes an open table's column set for the lifetime of the
  // object, so once one instance adds a column the others describe a
  // table that no longer exists: every later lock, read or close on them
  // throws. Only the instance that made the change, `except`, is current.
  //
  // Must be called once the write lock taken for the change has been
  // released, as reopening a table acquires locks of its own.
  arrow::Status RefreshInstances(std::size_t except);

  // Replace the given instance with one that is current. Must run on
  // the instance's isolation thread.
  //
  // An instance of a table made by Make() is reopened. An instance of one
  // made by Spawn() -- a reference table, e.g. a TAQL selection -- is rebuilt
  // over the root table's instance on the same thread, from the same rows
  // and columns. Its query is deliberately not run again: the table may
  // have changed since, and a re-evaluated selection could differ from the
  // one the other instances hold.
  //
  // Only the instance's TableProxy is replaced, so this is logically const.
  arrow::Status RefreshInstance(std::size_t instance) const;

  std::size_t nInstances() const { return proxy_pools_.size(); }

 protected:
  IsolatedTableProxy() = default;
  IsolatedTableProxy(const IsolatedTableProxy& rhs) = delete;
  IsolatedTableProxy(IsolatedTableProxy&& rhs) = delete;
  IsolatedTableProxy& operator=(const IsolatedTableProxy& rhs) = delete;
  IsolatedTableProxy& operator=(IsolatedTableProxy&& rhs) = delete;

  // Gets the least active instance
  std::size_t GetInstance() const;

  // Can instances of this ITP be replaced by RefreshInstance?
  bool CanRefresh() const { return reopen_ || root_; }

  // Lock the given instance on its isolation thread. An instance found to
  // be out of date -- another handle or process added a column -- is
  // refreshed and locked again, once. Other failures throw, as before.
  arrow::Result<std::unique_ptr<MaybeLockAndFinalise>> LockInstance(
      std::size_t instance, CasaLockType lock_type) const;

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
  // Holds an instance's TableProxy. Shared with the ITPs returned by
  // SpawnWriter, so that a refreshed instance is seen by all of them.
  // Only ever read or replaced on the instance's own isolation thread.
  struct ProxySlot {
    std::shared_ptr<casacore::TableProxy> proxy;
  };

  struct ProxyAndPool {
    std::shared_ptr<ProxySlot> slot_;
    std::shared_ptr<arrow::internal::ThreadPool> io_pool_;
  };

  std::vector<ProxyAndPool> proxy_pools_;
  // How a table made by Make() reopens its instances
  ReopenFn reopen_;
  // The ITP made by Make() that a Spawn()ed ITP ultimately derives from
  std::shared_ptr<IsolatedTableProxy> root_;
  // Default to closed so a partially-constructed or default-constructed
  // proxy is never treated as open. Atomic because it is written/read
  // across the isolation pool threads.
  std::atomic<bool> is_closed_{true};
  std::vector<std::shared_ptr<IsolatedTableProxy>> dependencies_;
};

}  // namespace detail
}  // namespace arcae

#endif  // ARCAE_ISOLATED_TABLE_PROXY_H
