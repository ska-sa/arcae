// This test case writes to a Tiled Storage Manager at tile boundaries from
// multiple threads, each through its own IsolatedTableProxy over the same
// table. (In practice, arcae only writes from a single thread at a time.)
//
// Each write takes an explicit user-locking write lock (LockType::Write), so
// the patched FileLocker's shared LockState serialises the writers in-process
// (multi-reader/single-writer). This both prevents the Tiled Storage Manager
// header corruption that an unsynchronised version exhibits ("FilebufIO::
// readBlock - incorrect number of bytes" against "table.f0") and exercises the
// cross-instance write-lock coordination. See:
// 1. CAS-13609 in https://casadocs.readthedocs.io/en/v6.4.3/changelog.html
// 2. https://keflavich.github.io/blog/casa-reading-incorrect-number-of-bytes-wmpi.html

#include <memory>

#include <arcae/isolated_table_proxy.h>
#include <arcae/new_table_proxy.h>
#include <arcae/table_factory.h>

#include <arrow/testing/gtest_util.h>
#include <arrow/util/thread_pool.h>

#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/tables/Tables.h>

#include <casacore/casa/IO/FileLocker.h>
#include <casacore/tables/Tables/TableLock.h>
#include <casacore/tables/Tables/TableProxy.h>
#include <gtest/gtest.h>

#include <tests/test_utils.h>

using ::arrow::Result;

using ::arcae::detail::IsolatedTableProxy;

using LockType = ::casacore::FileLocker::LockType;
using ::casacore::Array;
using ::casacore::ArrayColumn;
using ::casacore::ArrayColumnDesc;
using ::casacore::ColumnDesc;
using ::casacore::TableLock;
using CasaComplex = ::casacore::Complex;
using MS = ::casacore::MeasurementSet;
using MSColumns = ::casacore::MSMainEnums::PredefinedColumns;
using ::casacore::Record;
using ::casacore::SetupNewTable;
using ::casacore::Slice;
using ::casacore::Table;
using ::casacore::TableColumn;
using ::casacore::TableDesc;
using ::casacore::TableProxy;
using ::casacore::TiledColumnStMan;
using IPos = ::casacore::IPosition;

using namespace std::string_literals;

static constexpr std::size_t knrow = 20;
static constexpr std::size_t knchan = 4;
static constexpr std::size_t kncorr = 1;
static constexpr std::size_t knthreads = 16;

static constexpr std::size_t kinc = 2;

class WriteTests : public ::testing::Test {
 protected:
  std::string table_name_;

  void SetUp() override {
    auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    auto table_desc = TableDesc(MS::requiredTableDesc());
    table_name_ = std::string(test_info->name() + "-"s + arcae::hexuuid(4) + ".table"s);

    auto data_shape = IPos({kncorr, knchan});
    auto tile_shape = IPos({kncorr, knchan, 1});
    auto data_column_desc =
        ArrayColumnDesc<CasaComplex>("MODEL_DATA", data_shape, ColumnDesc::FixedShape);
    table_desc.addColumn(data_column_desc);
    auto storage_manager = TiledColumnStMan("TiledModelData", tile_shape);
    auto setup_new_table = SetupNewTable(table_name_, table_desc, Table::New);
    setup_new_table.bindColumn("MODEL_DATA", storage_manager);
    auto ms = MS(setup_new_table, knrow);

    // Create a ramp of data values and insert it into the data column
    auto data = Array<CasaComplex>(IPos{kncorr, knchan, knrow});

    for (auto [i, it] = std::tuple{0, data.begin()}; it != data.end(); ++it, ++i) {
      *it = i;
    }

    auto data_column = ArrayColumn<CasaComplex>(TableColumn(ms, "MODEL_DATA"));
    data_column.putColumn(data);
  }

  arrow::Result<std::shared_ptr<IsolatedTableProxy>> OpenTable() {
    return IsolatedTableProxy::Make([name = table_name_]() {
      auto lock = TableLock(TableLock::LockOption::UserLocking);
      auto lockoptions = Record();
      lockoptions.define("option", "user");
      lockoptions.define("interval", lock.interval());
      lockoptions.define("maxwait", casacore::Int(lock.maxWait()));
      auto tp = std::make_shared<TableProxy>(name, lockoptions, Table::Old);
      tp->reopenRW();
      return tp;
    });
  }
};

// Parallel NewTableProxy writes
TEST_F(WriteTests, Parallel) {
  {
    // Write some data in parallel;
    std::vector<std::shared_ptr<IsolatedTableProxy>> writers;
    std::vector<arrow::Future<bool>> futures;
    ASSERT_OK_AND_ASSIGN(auto pool, arrow::internal::ThreadPool::Make(knthreads));

    // Create knthreads worth of SafeTableProxies
    for (std::size_t i = 0; i < knthreads; ++i) {
      ASSERT_OK_AND_ASSIGN(auto writer, OpenTable());
      writers.emplace_back(std::move(writer));
    }

    // Iterate over the space of knrows in increments of kinc
    // Each iteration will independently write it's part of the column
    for (std::size_t r = 0, c = 0; r < knrow; r += kinc, ++c) {
      std::size_t end = std::min(r + kinc, knrow);
      auto result = arrow::DeferNotOk(pool->Submit(
          [start = r, nrow = end - r, writer = writers[c % knthreads]]() mutable {
            return writer->RunAsync(
                [start = start, nrow = nrow](TableProxy& proxy) mutable -> Result<bool> {
                  auto table = proxy.table();
                  auto table_column = TableColumn(table, "MODEL_DATA");
                  const auto& column_desc = table_column.columnDesc();
                  auto column = ArrayColumn<CasaComplex>(table_column);
                  column.setMaximumCacheSize(1);
                  auto data = Array<CasaComplex>(IPos({kncorr, knchan, int(nrow)}));
                  data.set(start);

                  try {
                    column.putColumnRange(Slice(start, nrow), data);
                  } catch (std::exception& e) {
                    return arrow::Status::Invalid("Write failed ", e.what());
                  }

                  return true;
                },
                LockType::Write);
          }));

      futures.push_back(result);
    }

    // Wait for all futures to complete
    auto all = arrow::All(futures).result().ValueOrDie();
    for (auto res : all) res.ValueOrDie();
    all.clear();
    futures.clear();
  }

  // Reopen the table
  ASSERT_OK_AND_ASSIGN(auto itp, OpenTable());

  // Using the same iteration pattern for writes, read sections of the table
  // and compare them for the expected result
  for (std::size_t r = 0, c = 0; r < knrow; r += kinc, ++c) {
    std::size_t end = std::min(r + kinc, knrow);

    auto future = itp->RunAsync(
        [start = r,
         nrow = end - r](TableProxy& proxy) mutable -> arrow::Result<Array<CasaComplex>> {
          auto table = proxy.table();
          auto table_column = TableColumn(table, "MODEL_DATA");
          const auto& column_desc = table_column.columnDesc();
          auto column = ArrayColumn<CasaComplex>(table_column);
          table.lock(false, 0);
          std::shared_ptr<void> result(nullptr, [&](...) { table.unlock(); });

          try {
            return column.getColumnRange(Slice(start, nrow));
          } catch (std::exception& e) {
            return arrow::Status::Invalid("Write failed ", e.what());
          }
        });

    ASSERT_OK_AND_ASSIGN(auto data, future.MoveResult());

    for (auto it = data.begin(); it != data.end(); ++it) {
      EXPECT_EQ(*it, CasaComplex(r, 0));
    }
  }
}
