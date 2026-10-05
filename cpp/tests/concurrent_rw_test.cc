// With ninstances > 1, a writer (instance 0, via SpawnWriter) must exclude
// readers running on the *other* instances. The patched FileLocker only
// coordinates multi-reader/single-writer access among threads sharing a
// single FileLocker object; because arcae gives each instance its own
// TableProxy -> Table -> LockFile -> FileLocker, the
// coordination historically did not span instances and a reader could acquire a
// read lock on the same byte range while a writer held the write lock (silently
// downgrading it at the OS level).
//
// WriterExcludesReadersAcrossInstances is deterministic: it fails on the
// pre-fix build (readers observe the writer in its critical section) and passes
// once the lock state is shared across instances.
//
// The *AcrossProcesses tests cover the orthogonal per-process axis. Across
// distinct PIDs, fcntl(F_SETLK) exclusion is casacore's genuine, original
// mechanism (it worked even pre-patch), so these are NOT a regression gate for
// Issue 1. Their job is to guard that the patch's rewrites -- HashThreadId
// reqIds replacing pid, the refcounted shared-fd LockFile cache, the new
// LockState -- did not BREAK the inter-process exclusion casacore always had,
// and that a reader in another process observes a writer's committed data via
// casacore's getInfo/putInfo lock-file handshake (the real cross-process
// freshness path; the thread_local TableCache concern is in-process only).
//
// They fork() before any IsolatedTableProxy (hence any Arrow/casacore thread)
// is created, so the forking process is still effectively single-threaded and
// the threaded-fork lock-inheritance hazard is avoided. Children coordinate via
// an anonymous MAP_SHARED region and communicate failure via exit code; all
// gtest assertions run in the parent after reaping, because ASSERT/EXPECT in a
// forked child do not propagate to the parent's test result.

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdlib>
#endif

#include "gtest/gtest.h"

#include <arrow/status.h>
#include <arrow/testing/gtest_util.h>
#include <arrow/util/future.h>

#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/ArrayMath.h>
#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/tables/Tables.h>
#include <casacore/tables/Tables/ArrColDesc.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/SetupNewTab.h>
#include <casacore/tables/Tables/TableProxy.h>

#include <tests/test_utils.h>

#include "arcae/isolated_table_proxy.h"

using ::arcae::detail::IsolatedTableProxy;
using ::arrow::Result;

using LockType = ::casacore::FileLocker::LockType;
using ::casacore::Array;
using ::casacore::ArrayColumn;
using ::casacore::ArrayColumnDesc;
using ::casacore::ColumnDesc;
using CasaInt = ::casacore::Int;
using MS = ::casacore::MeasurementSet;
using ::casacore::Record;
using ::casacore::SetupNewTable;
using ::casacore::Table;
using ::casacore::TableColumn;
using ::casacore::TableDesc;
using ::casacore::TableProxy;
using IPos = ::casacore::IPosition;

using namespace std::string_literals;
using namespace std::chrono_literals;

namespace {

static constexpr std::size_t knrow = 512;
static constexpr std::size_t knchan = 16;
static constexpr std::size_t kncorr = 4;

class ConcurrentReadWriteTest : public ::testing::Test {
 protected:
  std::string table_name_;

  void SetUp() override {
    auto* ti = ::testing::UnitTest::GetInstance()->current_test_info();
    table_name_ = std::string(ti->name() + "-"s + arcae::hexuuid(4) + ".table"s);

    auto table_desc = TableDesc(MS::requiredTableDesc());
    auto data_shape = IPos({kncorr, knchan});
    table_desc.addColumn(
        ArrayColumnDesc<CasaInt>("DATA", data_shape, ColumnDesc::FixedShape));
    auto setup = SetupNewTable(table_name_, table_desc, Table::New);
    auto ms = MS(setup, knrow);
    auto col = ArrayColumn<CasaInt>(TableColumn(ms, "DATA"));
    col.putColumn(Array<CasaInt>(IPos({kncorr, knchan, knrow}), CasaInt(0)));
  }

  Result<std::shared_ptr<IsolatedTableProxy>> OpenTable(std::size_t ninstances) {
    return IsolatedTableProxy::Make(
        [name = table_name_]() {
          auto lockoptions = Record();
          lockoptions.define("option", "user");
          auto tp = std::make_shared<TableProxy>(name, lockoptions, Table::Old);
          tp->reopenRW();
          return tp;
        },
        ninstances);
  }
};

// Deterministic gate: a writer on instance 0 must exclude readers on other
// instances for the duration that it holds the write lock.
TEST_F(ConcurrentReadWriteTest, WriterExcludesReadersAcrossInstances) {
  ASSERT_OK_AND_ASSIGN(auto itp, OpenTable(/*ninstances=*/2));
  auto writer = itp->SpawnWriter();

  std::atomic<bool> writer_in_critical{false};
  std::atomic<bool> reader_saw_writer{false};
  std::atomic<int> reads_completed{0};

  // The writer holds the write lock for a fixed window. The flag is only ever
  // true while the write lock is held (MaybeLockAndFinalise acquires before the
  // functor runs and releases after it returns).
  auto wfut = writer->RunAsync(
      [&](TableProxy&) -> Result<bool> {
        writer_in_critical = true;
        std::this_thread::sleep_for(400ms);
        writer_in_critical = false;
        return true;
      },
      LockType::Write);

  // Let the writer enter its critical section.
  std::this_thread::sleep_for(80ms);
  ASSERT_TRUE(writer_in_critical.load()) << "writer did not enter its critical section";

  // Fan readers out across the remaining instances while the writer holds the
  // lock. A reader that runs its functor (i.e. holds a read lock) while the
  // writer is still in its critical section proves the exclusion is broken.
  std::vector<arrow::Future<bool>> rfuts;
  while (writer_in_critical.load()) {
    rfuts.push_back(itp->RunAsync(
        [&](const TableProxy&) -> Result<bool> {
          if (writer_in_critical.load()) reader_saw_writer = true;
          return true;
        },
        LockType::Read));
    std::this_thread::sleep_for(5ms);
  }

  for (auto& f : rfuts) {
    auto r = f.MoveResult();
    if (r.ok()) ++reads_completed;
  }
  ASSERT_OK(wfut.status());

  EXPECT_FALSE(reader_saw_writer.load())
      << "A reader acquired a read lock while a writer held the write lock on "
         "another instance (cross-instance MRSW exclusion failed). "
      << reads_completed.load() << " reads completed.";

  // The writer borrows instance 0 from itp; its custom deleter releases the
  // borrowed proxy without closing it, so we drop it (rather than Close it,
  // which would double-close the shared proxy) before closing the parent.
  writer.reset();
  ASSERT_OK(itp->Close());
}

// Stress gate: a writer repeatedly writes a uniform value to every row while
// readers fan across the other instances. A correctly excluded read always
// observes an internally uniform column; a torn read (mixed old/new values)
// indicates the writer was not excluded or stale buffers were served.
TEST_F(ConcurrentReadWriteTest, NoTornReadsAcrossInstances) {
  static constexpr int knwrites = 200;
  ASSERT_OK_AND_ASSIGN(auto itp, OpenTable(/*ninstances=*/4));
  auto writer = itp->SpawnWriter();

  std::atomic<bool> writing{true};
  std::atomic<int> torn_reads{0};
  std::atomic<int> reads_completed{0};

  std::thread writer_thread([&]() {
    for (int w = 1; w <= knwrites; ++w) {
      auto status = writer->RunSync(
          [w](TableProxy& tp) -> Result<bool> {
            auto col = ArrayColumn<CasaInt>(TableColumn(tp.table(), "DATA"));
            col.putColumn(Array<CasaInt>(IPos({kncorr, knchan, knrow}), CasaInt(w)));
            return true;
          },
          LockType::Write);
      if (!status.ok()) {
        ADD_FAILURE() << "write failed: " << status.status();
        break;
      }
    }
    writing = false;
  });

  while (writing.load()) {
    std::vector<arrow::Future<bool>> rfuts;
    for (int i = 0; i < 16; ++i) {
      rfuts.push_back(itp->RunAsync(
          [](const TableProxy& tp) -> Result<bool> {
            auto col = ArrayColumn<CasaInt>(TableColumn(tp.table(), "DATA"));
            auto data = col.getColumn();
            // Uniform write => min == max for a non-torn read.
            return casacore::min(data) == casacore::max(data);
          },
          LockType::Read));
    }
    for (auto& f : rfuts) {
      auto r = f.MoveResult();
      if (r.ok()) {
        ++reads_completed;
        if (!*r) ++torn_reads;
      }
    }
  }

  writer_thread.join();
  EXPECT_EQ(torn_reads.load(), 0)
      << "observed torn reads (" << reads_completed.load() << " reads total)";

  writer.reset();
  ASSERT_OK(itp->Close());
}

#ifndef _WIN32

// Coordination region shared between the forked children and the parent. All
// fields are lock-free atomics living in anonymous MAP_SHARED memory mapped
// before fork(), so every process sees the same bytes.
struct SharedState {
  std::atomic<int> ready;                // barrier: children opened their tables
  std::atomic<bool> writer_in_critical;  // writer holds the write lock right now
  std::atomic<bool> reader_saw_writer;   // a reader ran while writer_in_critical
  std::atomic<bool> writing;             // writer loop is still running (torn-read test)
  std::atomic<int> torn_reads;           // reads observing min != max
  std::atomic<int> reads_completed;      // successful reads (diagnostic)
};

namespace {

SharedState* MapSharedState() {
  void* p = ::mmap(nullptr, sizeof(SharedState), PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return nullptr;
  auto* s = new (p) SharedState();
  s->ready.store(0);
  s->writer_in_critical.store(false);
  s->reader_saw_writer.store(false);
  s->writing.store(true);
  s->torn_reads.store(0);
  s->reads_completed.store(0);
  return s;
}

void UnmapSharedState(SharedState* s) {
  if (s) ::munmap(static_cast<void*>(s), sizeof(SharedState));
}

// Spin until every child has reached the barrier (or a generous deadline), so
// the writer's critical window genuinely overlaps reader activity.
void WaitForBarrier(SharedState* s, int total) {
  auto deadline = std::chrono::steady_clock::now() + 5s;
  while (s->ready.load() < total && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
}

// Reap all children, killing any that outlive the deadline so a cross-process
// deadlock fails the test instead of hanging CI. Returns true iff every child
// exited 0 within the deadline.
bool ReapAll(const std::vector<pid_t>& pids, std::chrono::milliseconds timeout) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  std::vector<bool> reaped(pids.size(), false);
  std::vector<int> codes(pids.size(), -1);
  std::size_t outstanding = pids.size();

  while (outstanding > 0 && std::chrono::steady_clock::now() < deadline) {
    for (std::size_t i = 0; i < pids.size(); ++i) {
      if (reaped[i]) continue;
      int status = 0;
      pid_t r = ::waitpid(pids[i], &status, WNOHANG);
      if (r == pids[i]) {
        reaped[i] = true;
        --outstanding;
        codes[i] = (WIFEXITED(status) ? WEXITSTATUS(status) : 128);
      }
    }
    if (outstanding > 0) std::this_thread::sleep_for(10ms);
  }

  bool all_ok = true;
  for (std::size_t i = 0; i < pids.size(); ++i) {
    if (!reaped[i]) {
      ::kill(pids[i], SIGKILL);
      ::waitpid(pids[i], nullptr, 0);
      ADD_FAILURE() << "child " << pids[i]
                    << " did not exit within the deadline "
                       "(possible cross-process deadlock)";
      all_ok = false;
    } else if (codes[i] != 0) {
      ADD_FAILURE() << "child " << pids[i] << " exited with code " << codes[i];
      all_ok = false;
    }
  }
  return all_ok;
}

// Child exit codes (parent reports these on failure).
enum ChildExit { kChildOk = 0, kChildOpenFailed = 2, kChildWriteFailed = 3 };

}  // namespace

// A writer process holding the write lock must exclude reader processes for the
// duration it holds it. This is the per-process analogue of
// WriterExcludesReadersAcrossInstances; it guards casacore's inter-process
// fcntl exclusion against regressions in the patch's reqId/shared-fd rewrites.
TEST_F(ConcurrentReadWriteTest, WriterExcludesReadersAcrossProcesses) {
  static constexpr int knreaders = 3;
  static constexpr int ntotal = knreaders + 1;

  SharedState* s = MapSharedState();
  ASSERT_NE(s, nullptr) << "mmap of shared state failed";

  std::vector<pid_t> pids;
  bool is_child = false;
  bool child_is_writer = false;

  for (int i = 0; i < ntotal && !is_child; ++i) {
    pid_t pid = ::fork();
    ASSERT_NE(pid, -1) << "fork failed";
    if (pid == 0) {
      is_child = true;
      child_is_writer = (i == 0);
    } else {
      pids.push_back(pid);
    }
  }

  if (is_child) {
    // ---- writer child ----
    if (child_is_writer) {
      auto itp_res = OpenTable(/*ninstances=*/1);
      if (!itp_res.ok()) ::_exit(kChildOpenFailed);
      auto itp = *itp_res;
      auto writer = itp->SpawnWriter();
      s->ready.fetch_add(1);
      WaitForBarrier(s, ntotal);

      // The flag is only ever set while the write lock is held: the lock is
      // acquired by MaybeLockAndFinalise before the functor runs and released
      // after it returns.
      auto fut = writer->RunAsync(
          [&](TableProxy&) -> Result<bool> {
            s->writer_in_critical.store(true);
            std::this_thread::sleep_for(400ms);
            s->writer_in_critical.store(false);
            return true;
          },
          LockType::Write);
      auto r = fut.MoveResult();
      ::_exit(r.ok() ? kChildOk : kChildWriteFailed);
    }

    // ---- reader child ----
    auto itp_res = OpenTable(/*ninstances=*/2);
    if (!itp_res.ok()) ::_exit(kChildOpenFailed);
    auto itp = *itp_res;
    s->ready.fetch_add(1);
    WaitForBarrier(s, ntotal);

    // Wait (bounded) for the writer to enter its critical section, then hammer
    // it with reads for as long as it stays there. A read that runs its functor
    // while the writer flag is up proves the inter-process exclusion is broken.
    // A read that blocks until release, or fails fast under contention, is fine.
    auto wstart = std::chrono::steady_clock::now();
    while (!s->writer_in_critical.load() &&
           std::chrono::steady_clock::now() - wstart < 2s) {
      std::this_thread::sleep_for(1ms);
    }
    while (s->writer_in_critical.load()) {
      auto fut = itp->RunAsync(
          [&](const TableProxy&) -> Result<bool> {
            if (s->writer_in_critical.load()) s->reader_saw_writer.store(true);
            return true;
          },
          LockType::Read);
      auto r = fut.MoveResult();
      if (r.ok()) s->reads_completed.fetch_add(1);
      std::this_thread::sleep_for(2ms);
    }
    ::_exit(kChildOk);
  }

  // ---- parent ----
  bool all_ok = ReapAll(pids, /*timeout=*/30s);
  EXPECT_TRUE(all_ok);
  EXPECT_FALSE(s->reader_saw_writer.load())
      << "a reader process acquired a read lock while a writer process held the "
         "write lock (inter-process MRSW exclusion failed). "
      << s->reads_completed.load() << " reads completed.";
  UnmapSharedState(s);
}

// A writer process repeatedly writes a uniform value while reader processes read
// the same column. A correctly synchronised read always observes an internally
// uniform column; a torn read (mixed old/new values) means the reader process
// was not excluded or served stale storage-manager buffers -- i.e. casacore's
// cross-process getInfo/putInfo freshness handshake did not fire.
TEST_F(ConcurrentReadWriteTest, NoTornReadsAcrossProcesses) {
  static constexpr int knreaders = 2;
  static constexpr int ntotal = knreaders + 1;
  static constexpr int knwrites = 50;

  SharedState* s = MapSharedState();
  ASSERT_NE(s, nullptr) << "mmap of shared state failed";

  std::vector<pid_t> pids;
  bool is_child = false;
  bool child_is_writer = false;

  for (int i = 0; i < ntotal && !is_child; ++i) {
    pid_t pid = ::fork();
    ASSERT_NE(pid, -1) << "fork failed";
    if (pid == 0) {
      is_child = true;
      child_is_writer = (i == 0);
    } else {
      pids.push_back(pid);
    }
  }

  if (is_child) {
    // ---- writer child ----
    if (child_is_writer) {
      auto itp_res = OpenTable(/*ninstances=*/1);
      if (!itp_res.ok()) ::_exit(kChildOpenFailed);
      auto itp = *itp_res;
      auto writer = itp->SpawnWriter();
      s->ready.fetch_add(1);
      WaitForBarrier(s, ntotal);

      int code = kChildOk;
      for (int w = 1; w <= knwrites; ++w) {
        auto status = writer->RunSync(
            [w](TableProxy& tp) -> Result<bool> {
              auto col = ArrayColumn<CasaInt>(TableColumn(tp.table(), "DATA"));
              col.putColumn(Array<CasaInt>(IPos({kncorr, knchan, knrow}), CasaInt(w)));
              return true;
            },
            LockType::Write);
        if (!status.ok()) {
          code = kChildWriteFailed;
          break;
        }
      }
      s->writing.store(false);
      ::_exit(code);
    }

    // ---- reader child ----
    auto itp_res = OpenTable(/*ninstances=*/2);
    if (!itp_res.ok()) ::_exit(kChildOpenFailed);
    auto itp = *itp_res;
    s->ready.fetch_add(1);
    WaitForBarrier(s, ntotal);

    // Safety deadline guards against a wedged writer; the parent watchdog backs
    // it up by killing survivors.
    auto deadline = std::chrono::steady_clock::now() + 60s;
    while (s->writing.load() && std::chrono::steady_clock::now() < deadline) {
      auto fut = itp->RunAsync(
          [](const TableProxy& tp) -> Result<bool> {
            auto col = ArrayColumn<CasaInt>(TableColumn(tp.table(), "DATA"));
            auto data = col.getColumn();
            // Uniform write => min == max for a non-torn read.
            return casacore::min(data) == casacore::max(data);
          },
          LockType::Read);
      auto r = fut.MoveResult();
      if (r.ok()) {
        s->reads_completed.fetch_add(1);
        if (!*r) s->torn_reads.fetch_add(1);
      }
      // Back off between reads: continuously re-acquired read locks starve the
      // cross-process writer (fcntl read/write contention), which both slows the
      // test to a crawl and erodes overlap. A short pause keeps ample read/write
      // interleaving while letting the writer make progress.
      std::this_thread::sleep_for(2ms);
    }
    ::_exit(kChildOk);
  }

  // ---- parent ----
  bool all_ok = ReapAll(pids, /*timeout=*/120s);
  EXPECT_TRUE(all_ok);
  EXPECT_EQ(s->torn_reads.load(), 0) << "observed torn reads across processes ("
                                     << s->reads_completed.load() << " reads total)";
  UnmapSharedState(s);
}

#endif  // _WIN32

}  // namespace
