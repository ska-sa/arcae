#include <chrono>
#include <memory>
#include <thread>
#include "gtest/gtest.h"

#include <arrow/status.h>
#include <arrow/testing/gtest_util.h>

#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/BasicSL/Complexfwd.h>
#include <casacore/casa/Utilities/DataType.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/tables/Tables.h>
#include <casacore/tables/Tables/SetupNewTab.h>
#include <casacore/tables/Tables/TableProxy.h>

#include <tests/test_utils.h>

#include <gtest/gtest.h>

#include "arcae/isolated_table_proxy.h"

using ::arcae::GetArrayColumn;
using ::arcae::detail::IsolatedTableProxy;

using casacore::Array;
using casacore::ArrayColumnDesc;
using casacore::ColumnDesc;
using casacore::Complex;
using MS = casacore::MeasurementSet;
using casacore::Record;
using casacore::SetupNewTable;
using casacore::Table;
using casacore::TableDesc;
using casacore::TableLock;
using casacore::TableProxy;
using IPos = casacore::IPosition;

using namespace std::string_literals;

static constexpr std::size_t knrow = 10;
static constexpr std::size_t knchan = 4;
static constexpr std::size_t kncorr = 2;

namespace {

class IsolatedTableProxyTest : public ::testing::Test {
 protected:
  std::string table_name_;

  void SetUp() override {
    auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    table_name_ = std::string(test_info->name() + "-"s + arcae::hexuuid(4) + ".table"s);

    auto table_desc = TableDesc(MS::requiredTableDesc());
    auto data_shape = IPos({kncorr, knchan});
    auto data_column_desc =
        ArrayColumnDesc<Complex>("MODEL_DATA", data_shape, ColumnDesc::FixedShape);
    table_desc.addColumn(data_column_desc);
    auto setup_new_table = SetupNewTable(table_name_, table_desc, Table::New);
    auto ms = MS(setup_new_table, knrow);
    auto data = GetArrayColumn<Complex>(ms, MS::MODEL_DATA);
    data.putColumn(Array<Complex>(IPos({kncorr, knchan, knrow}), {1, 2}));
  }

  // Open the table. If given, weak_proxy is set to the underlying TableProxy,
  // whose expiry marks the point at which the table has actually been released
  arrow::Result<std::shared_ptr<IsolatedTableProxy>> OpenTable(
      std::weak_ptr<TableProxy>* weak_proxy = nullptr) {
    return IsolatedTableProxy::Make([name = table_name_, weak_proxy]() {
      auto lock = TableLock(TableLock::LockOption::AutoLocking);
      auto lockoptions = Record();
      lockoptions.define("option", "nolock");
      lockoptions.define("internal", lock.interval());
      lockoptions.define("maxwait", casacore::Int(lock.maxWait()));
      auto proxy = std::make_shared<TableProxy>(name, lockoptions, Table::Old);
      if (weak_proxy) *weak_proxy = proxy;
      return proxy;
    });
  }
};

// Generous upper bound on operations that should complete immediately.
// Exceeding it means a deadlock, so the tests fail instead of hanging
static constexpr double kTeardownTimeout = 30.0;

// Poll until the proxy expires. Teardown of an IsolatedTableProxy hands its
// I/O pools to another thread, so it does not complete synchronously
bool WaitForExpiry(const std::weak_ptr<TableProxy>& proxy,
                   double seconds = kTeardownTimeout) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);

  while (!proxy.expired()) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  return true;
}

TEST_F(IsolatedTableProxyTest, MakeTable) {
  ASSERT_OK_AND_ASSIGN(
      auto itp, IsolatedTableProxy::Make([]() {
        auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        auto name = std::string(test_info->name() + "-"s + arcae::hexuuid(4) + ".table"s);
        auto table_desc = TableDesc(MS::requiredTableDesc());
        auto setup_new_table = SetupNewTable(name, table_desc, Table::New);
        return std::make_shared<TableProxy>(MS(setup_new_table, knrow));
      }));

  {
    // Test in sync mode
    ASSERT_OK_AND_ASSIGN(
        auto nrow, itp->RunSync([](const TableProxy& tp) { return tp.table().nrow(); }));
    EXPECT_EQ(nrow, knrow);
  }
  {
    // Test in async mode
    auto fut = itp->RunAsync([](const TableProxy& tp) { return tp.table().nrow(); });
    ASSERT_OK_AND_ASSIGN(auto nrow, fut.MoveResult());
    EXPECT_EQ(nrow, knrow);
  }
}

TEST_F(IsolatedTableProxyTest, RunAsyncConstAndNonConst) {
  ASSERT_OK_AND_ASSIGN(auto itp, OpenTable());
  {
    // Test in sync mode
    ASSERT_OK_AND_ASSIGN(
        auto cnrow, itp->RunSync([](const TableProxy& tp) { return tp.table().nrow(); }));
    EXPECT_EQ(cnrow, knrow);
    ASSERT_OK_AND_ASSIGN(auto nrow,
                         itp->RunSync([](TableProxy& tp) { return tp.table().nrow(); }));
    EXPECT_EQ(nrow, knrow);
  }

  {
    // Test in async mode
    auto fut = itp->RunAsync([](const TableProxy& tp) { return tp.table().nrow(); });
    ASSERT_OK_AND_ASSIGN(auto cnrow, fut.MoveResult());
    EXPECT_EQ(cnrow, knrow);
    fut = itp->RunSync([](TableProxy& tp) { return tp.table().nrow(); });
    ASSERT_OK_AND_ASSIGN(auto nrow, fut.MoveResult());
  }
}

TEST_F(IsolatedTableProxyTest, GetColumn) {
  ASSERT_OK_AND_ASSIGN(auto itp, OpenTable());
  {
    // Test in sync mode
    ASSERT_OK_AND_ASSIGN(auto data,
                         itp->RunSync([column_name = "MODEL_DATA"](const TableProxy& tp) {
                           auto column = GetArrayColumn<Complex>(tp.table(), column_name);
                           return column.getColumn();
                         }));
    EXPECT_EQ(data.shape(), IPos({kncorr, knchan, knrow}));
  }

  {
    // Test in async mode
    auto fut = itp->RunAsync([column_name = "MODEL_DATA"](const TableProxy& tp) {
      auto column = GetArrayColumn<Complex>(tp.table(), column_name);
      return column.getColumn();
    });
    ASSERT_OK_AND_ASSIGN(auto data, fut.MoveResult());
    EXPECT_EQ(data.shape(), IPos({kncorr, knchan, knrow}));
  }
}

TEST_F(IsolatedTableProxyTest, FailIfClosed) {
  ASSERT_OK_AND_ASSIGN(auto itp, OpenTable());
  ASSERT_OK_AND_ASSIGN(auto close_result, itp->Close());
  EXPECT_EQ(close_result, true);
  ASSERT_NOT_OK(itp->RunSync([](const TableProxy& tp) { return true; }));
  ASSERT_OK_AND_ASSIGN(close_result, itp->Close());
  EXPECT_EQ(close_result, false);
}

// Close() submits the close to the isolation pool and waits for it. On the
// isolation thread itself, that waits for a task that only the waiting thread
// could ever run
TEST_F(IsolatedTableProxyTest, CloseFromIsolationThread) {
  ASSERT_OK_AND_ASSIGN(auto itp, OpenTable());
  auto fut = itp->RunAsync([itp](const TableProxy&) { return itp->Close(); });
  ASSERT_TRUE(fut.Wait(kTeardownTimeout))
      << "Close() deadlocked on its own isolation thread";
  ASSERT_OK_AND_ASSIGN(auto closed, fut.MoveResult());
  EXPECT_TRUE(closed);
  EXPECT_TRUE(itp->IsClosed());
}

// The read and write callbacks hold a reference to the proxy and are destroyed
// on an isolation thread once their pipeline completes, so the last reference
// is routinely released there. Destruction must still close the table, which
// means neither closing it nor destroying the I/O pools may block that thread
TEST_F(IsolatedTableProxyTest, DestroyOnIsolationThread) {
  std::weak_ptr<TableProxy> weak_proxy;

  {
    ASSERT_OK_AND_ASSIGN(auto itp, OpenTable(&weak_proxy));
    ASSERT_FALSE(weak_proxy.expired());

    // The functor owns the only remaining reference and is destroyed on the
    // isolation thread when the task completes
    auto fut = itp->RunAsync([itp](const TableProxy& tp) { return tp.table().nrow(); });
    itp.reset();
    ASSERT_OK_AND_ASSIGN(auto nrow, fut.MoveResult());
    EXPECT_EQ(nrow, knrow);
  }

  EXPECT_TRUE(WaitForExpiry(weak_proxy))
      << "Table was never closed: teardown deadlocked on its isolation thread";
}

}  // namespace
