#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/Arrays/Slicer.h>
#include <casacore/casa/BasicSL/Complex.h>
#include <casacore/casa/Containers/Record.h>
#include <casacore/casa/Utilities/DataType.h>
#include <casacore/tables/DataMan/DataManager.h>
#include <casacore/tables/DataMan/StandardStMan.h>
#include <casacore/tables/DataMan/TiledCellStMan.h>
#include <casacore/tables/DataMan/TiledColumnStMan.h>
#include <casacore/tables/DataMan/TiledDataStMan.h>
#include <casacore/tables/DataMan/TiledDataStManAccessor.h>
#include <casacore/tables/DataMan/TiledShapeStMan.h>
#include <casacore/tables/Tables/ArrColDesc.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/ScaColDesc.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <casacore/tables/Tables/SetupNewTab.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/TableDesc.h>

#include "arcae/data_partition.h"
#include "arcae/result_shape.h"
#include "arcae/selection.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <arrow/testing/gtest_util.h>

#include <tests/test_utils.h>

using ::arcae::detail::DataPartition;
using ::arcae::detail::IndexType;
using ::arcae::detail::MakeRowUnitStart;
using ::arcae::detail::ResultShapeData;
using ::arcae::detail::RowSplitOptions;
using ::arcae::detail::SelectionBuilder;
using ::arcae::detail::SpanPairs;

using ::casacore::ArrayColumnDesc;
using ::casacore::ColumnDesc;
using ::casacore::Complex;
using ::casacore::DataType;
using ::casacore::IPosition;
using ::casacore::ScalarColumnDesc;
using ::casacore::SetupNewTable;
using ::casacore::Slicer;
using ::casacore::Table;
using ::casacore::TableDesc;

using namespace std::string_literals;

namespace {

TEST(DataPartitionTest, Fixed) {
  auto result_shape =
      ResultShapeData{"DATA", IPosition{2, 3, 4}, 3, DataType::TpComplex, std::nullopt};

  auto selection = SelectionBuilder()
                       .Order('F')                 // 1 range in dim 0
                       .Add({3, 0, 1})             // 2 ranges in dim 1
                       .Add({2, 1, -1, 5, 6, -1})  // 3 ranges in dim 2
                       .Build();                   // 1 x 2 x 3

  ASSERT_OK_AND_ASSIGN(auto partition, DataPartition::Make(selection, result_shape));
  EXPECT_EQ(partition.nChunks(), 4);

  EXPECT_THAT(partition.Chunk(0).Disk(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(0).Mem(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(0).Disk(1), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(0).Mem(1), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(partition.Chunk(0).Disk(2), ::testing::ElementsAre(-1, -1));
  EXPECT_THAT(partition.Chunk(0).Mem(2), ::testing::ElementsAre(2, 5));
  EXPECT_THAT(partition.Chunk(0).ReferenceRows().rowVector(),
              ::testing::ElementsAre(-1, -1));
  EXPECT_EQ(partition.Chunk(0).SectionSlicer(),
            Slicer(IPosition({0, 0}), IPosition({1, 1}), Slicer::endIsLast));

  EXPECT_THAT(partition.Chunk(1).Disk(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(1).Mem(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(1).Disk(1), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(1).Mem(1), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(partition.Chunk(1).Disk(2), ::testing::ElementsAre(1, 2, 5, 6));
  EXPECT_THAT(partition.Chunk(1).Mem(2), ::testing::ElementsAre(1, 0, 3, 4));
  EXPECT_THAT(partition.Chunk(1).ReferenceRows().rowVector(),
              ::testing::ElementsAre(1, 2, 5, 6));
  EXPECT_EQ(partition.data_chunks_[1].SectionSlicer(),
            Slicer(IPosition({0, 0}), IPosition({1, 1}), Slicer::endIsLast));

  EXPECT_THAT(partition.Chunk(2).Disk(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(2).Mem(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(2).Disk(1), ::testing::ElementsAre(3));
  EXPECT_THAT(partition.Chunk(2).Mem(1), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(2).Disk(2), ::testing::ElementsAre(-1, -1));
  EXPECT_THAT(partition.Chunk(2).Mem(2), ::testing::ElementsAre(2, 5));
  EXPECT_THAT(partition.Chunk(2).ReferenceRows().rowVector(),
              ::testing::ElementsAre(-1, -1));
  EXPECT_EQ(partition.Chunk(2).SectionSlicer(),
            Slicer(IPosition({0, 3}), IPosition({1, 3}), Slicer::endIsLast));

  EXPECT_THAT(partition.Chunk(3).Disk(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(3).Mem(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(3).Disk(1), ::testing::ElementsAre(3));
  EXPECT_THAT(partition.Chunk(3).Mem(1), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(3).Disk(2), ::testing::ElementsAre(1, 2, 5, 6));
  EXPECT_THAT(partition.Chunk(3).Mem(2), ::testing::ElementsAre(1, 0, 3, 4));
  EXPECT_THAT(partition.Chunk(3).ReferenceRows().rowVector(),
              ::testing::ElementsAre(1, 2, 5, 6));
  EXPECT_EQ(partition.Chunk(3).SectionSlicer(),
            Slicer(IPosition({0, 3}), IPosition({1, 3}), Slicer::endIsLast));
}

TEST(DataPartitionTest, Variable) {
  auto result_shape =
      ResultShapeData{"DATA", std::nullopt, 3, DataType::TpComplex,
                      std::vector<IPosition>{IPosition({3, 4}), IPosition({1, 4})}};

  auto selection = SelectionBuilder()
                       .Order('F')      // 1 range in dim 0 (varies per row)
                       .Add({3, 0, 1})  // 2 ranges in dim 1
                       .Add({-1, 0})    // 2 rows
                       .Build();

  ASSERT_OK_AND_ASSIGN(auto partition, DataPartition::Make(selection, result_shape));
  EXPECT_EQ(partition.nChunks(), 4);  // 1x1x1 + 1x2x2

  EXPECT_THAT(partition.Chunk(0).Disk(0), ::testing::ElementsAre(0, 1, 2));
  EXPECT_THAT(partition.Chunk(0).Mem(0), ::testing::ElementsAre(0, 1, 2));
  EXPECT_THAT(partition.Chunk(0).Disk(1), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(0).Mem(1), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(partition.Chunk(0).Disk(2), ::testing::ElementsAre(-1));
  EXPECT_THAT(partition.Chunk(0).Mem(2), ::testing::ElementsAre(0));
  EXPECT_EQ(partition.Chunk(0).RowSlicer(),
            Slicer(IPosition({-1}), IPosition({-1}), Slicer::endIsLast));
  EXPECT_EQ(partition.Chunk(0).SectionSlicer(),
            Slicer(IPosition({0, 0}), IPosition({2, 1}), Slicer::endIsLast));

  EXPECT_THAT(partition.Chunk(1).Disk(0), ::testing::ElementsAre(0, 1, 2));
  EXPECT_THAT(partition.Chunk(1).Mem(0), ::testing::ElementsAre(0, 1, 2));
  EXPECT_THAT(partition.Chunk(1).Disk(1), ::testing::ElementsAre(3));
  EXPECT_THAT(partition.Chunk(1).Mem(1), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(1).Disk(2), ::testing::ElementsAre(-1));
  EXPECT_THAT(partition.Chunk(1).Mem(2), ::testing::ElementsAre(0));
  EXPECT_EQ(partition.Chunk(1).RowSlicer(),
            Slicer(IPosition({-1}), IPosition({-1}), Slicer::endIsLast));
  EXPECT_EQ(partition.Chunk(1).SectionSlicer(),
            Slicer(IPosition({0, 3}), IPosition({2, 3}), Slicer::endIsLast));

  EXPECT_THAT(partition.Chunk(2).Disk(0), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(2).Mem(0), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(2).Disk(1), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(2).Mem(1), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(partition.Chunk(2).Disk(2), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(2).Mem(2), ::testing::ElementsAre(1));
  EXPECT_EQ(partition.Chunk(2).RowSlicer(),
            Slicer(IPosition({0}), IPosition({0}), Slicer::endIsLast));
  EXPECT_EQ(partition.Chunk(2).SectionSlicer(),
            Slicer(IPosition({0, 0}), IPosition({0, 1}), Slicer::endIsLast));

  EXPECT_THAT(partition.Chunk(3).Disk(0), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(3).Mem(0), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(3).Disk(1), ::testing::ElementsAre(3));
  EXPECT_THAT(partition.Chunk(3).Mem(1), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(3).Disk(2), ::testing::ElementsAre(0));
  EXPECT_THAT(partition.Chunk(3).Mem(2), ::testing::ElementsAre(1));
  EXPECT_EQ(partition.Chunk(3).RowSlicer(),
            Slicer(IPosition({0}), IPosition({0}), Slicer::endIsLast));
  EXPECT_EQ(partition.Chunk(3).SectionSlicer(),
            Slicer(IPosition({0, 3}), IPosition({0, 3}), Slicer::endIsLast));
}

// Each row of a {2, 3} complex cell is 2 x 3 x 8 = 48 bytes
static constexpr std::size_t kRowBytes = 2 * 3 * 8;

// Rows are split into at most max_splits chunks of consecutive rows,
// whose sizes differ by at most one row
TEST(DataPartitionTest, SplitRows) {
  auto result_shape =
      ResultShapeData{"DATA", IPosition{2, 3, 10}, 3, DataType::TpComplex, std::nullopt};
  auto selection = SelectionBuilder().Order('F').Build();

  ASSERT_OK_AND_ASSIGN(auto partition,
                       DataPartition::Make(selection, result_shape, RowSplitOptions{4}));
  ASSERT_EQ(partition.nChunks(), 4);
  EXPECT_THAT(partition.Chunk(0).Disk(2), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(partition.Chunk(1).Disk(2), ::testing::ElementsAre(2, 3, 4));
  EXPECT_THAT(partition.Chunk(2).Disk(2), ::testing::ElementsAre(5, 6));
  EXPECT_THAT(partition.Chunk(3).Disk(2), ::testing::ElementsAre(7, 8, 9));
  EXPECT_EQ(partition.nElements(), 2 * 3 * 10);

  for (std::size_t c = 0; c < partition.nChunks(); ++c) {
    const auto& chunk = partition.Chunk(c);
    EXPECT_TRUE(chunk.IsContiguous());
    EXPECT_EQ(chunk.FlatOffset(), 2 * 3 * chunk.Mem(2)[0]);
    EXPECT_EQ(chunk.SectionSlicer(),
              Slicer(IPosition({0, 0}), IPosition({1, 2}), Slicer::endIsLast));
  }
}

// Chunks are not split below the minimum number of bytes
TEST(DataPartitionTest, SplitRowsMinimumBytes) {
  auto result_shape =
      ResultShapeData{"DATA", IPosition{2, 3, 10}, 3, DataType::TpComplex, std::nullopt};
  auto selection = SelectionBuilder().Order('F').Build();

  ASSERT_OK_AND_ASSIGN(
      auto partition,
      DataPartition::Make(selection, result_shape, RowSplitOptions{4, 5 * kRowBytes}));
  ASSERT_EQ(partition.nChunks(), 2);
  EXPECT_THAT(partition.Chunk(0).Disk(2), ::testing::ElementsAre(0, 1, 2, 3, 4));
  EXPECT_THAT(partition.Chunk(1).Disk(2), ::testing::ElementsAre(5, 6, 7, 8, 9));

  // A read just over twice the minimum is split into two equal chunks,
  // rather than two of the minimum and a smaller remainder
  ASSERT_OK_AND_ASSIGN(partition, DataPartition::Make(selection, result_shape,
                                                      RowSplitOptions{4, 4 * kRowBytes}));
  ASSERT_EQ(partition.nChunks(), 2);
  EXPECT_THAT(partition.Chunk(0).Disk(2), ::testing::ElementsAre(0, 1, 2, 3, 4));
  EXPECT_THAT(partition.Chunk(1).Disk(2), ::testing::ElementsAre(5, 6, 7, 8, 9));

  // A read less than twice the minimum is not split
  ASSERT_OK_AND_ASSIGN(partition, DataPartition::Make(selection, result_shape,
                                                      RowSplitOptions{4, 6 * kRowBytes}));
  EXPECT_EQ(partition.nChunks(), 1);
  ASSERT_OK_AND_ASSIGN(
      partition,
      DataPartition::Make(selection, result_shape, RowSplitOptions{4, 100 * kRowBytes}));
  EXPECT_EQ(partition.nChunks(), 1);
}

// Selected rows with gaps and in arbitrary order are split in disk order,
// while missing rows remain in a chunk of their own
TEST(DataPartitionTest, SplitSelectedRows) {
  auto result_shape =
      ResultShapeData{"DATA", IPosition{2, 3, 8}, 3, DataType::TpComplex, std::nullopt};
  auto selection = SelectionBuilder()
                       .Order('F')
                       .AddEmpty()
                       .AddEmpty()
                       .Add({8, -1, 5, 6, 7, 4, -1, 0})
                       .Build();

  ASSERT_OK_AND_ASSIGN(auto partition,
                       DataPartition::Make(selection, result_shape, RowSplitOptions{3}));
  ASSERT_EQ(partition.nChunks(), 4);
  EXPECT_THAT(partition.Chunk(0).Disk(2), ::testing::ElementsAre(-1, -1));
  EXPECT_THAT(partition.Chunk(1).Disk(2), ::testing::ElementsAre(0, 4));
  EXPECT_THAT(partition.Chunk(1).Mem(2), ::testing::ElementsAre(7, 5));
  EXPECT_THAT(partition.Chunk(2).Disk(2), ::testing::ElementsAre(5, 6));
  EXPECT_THAT(partition.Chunk(2).Mem(2), ::testing::ElementsAre(2, 3));
  EXPECT_THAT(partition.Chunk(3).Disk(2), ::testing::ElementsAre(7, 8));
  EXPECT_THAT(partition.Chunk(3).Mem(2), ::testing::ElementsAre(4, 0));
  EXPECT_EQ(partition.nElements(), 2 * 3 * 8);
}

// Storage units of four rows
::casacore::rownr_t UnitOfFour(::casacore::rownr_t row) { return row - row % 4; }

// Chunk boundaries move to the nearest start of a storage unit
TEST(DataPartitionTest, SplitRowsAligned) {
  auto result_shape =
      ResultShapeData{"DATA", IPosition{2, 3, 10}, 3, DataType::TpComplex, std::nullopt};
  auto selection = SelectionBuilder().Order('F').Build();

  // Even boundaries at rows 2, 5 and 7 move to 0, 4 and 8.
  // Boundary 0 would leave an empty chunk, so it is dropped
  ASSERT_OK_AND_ASSIGN(
      auto partition,
      DataPartition::Make(selection, result_shape, RowSplitOptions{4, 0, UnitOfFour}));
  ASSERT_EQ(partition.nChunks(), 3);
  EXPECT_THAT(partition.Chunk(0).Disk(2), ::testing::ElementsAre(0, 1, 2, 3));
  EXPECT_THAT(partition.Chunk(1).Disk(2), ::testing::ElementsAre(4, 5, 6, 7));
  EXPECT_THAT(partition.Chunk(2).Disk(2), ::testing::ElementsAre(8, 9));
  EXPECT_EQ(partition.nElements(), 2 * 3 * 10);

  // A unit holding every row is not split
  ASSERT_OK_AND_ASSIGN(
      partition, DataPartition::Make(selection, result_shape,
                                     RowSplitOptions{4, 0, [](auto) { return 0; }}));
  EXPECT_EQ(partition.nChunks(), 1);
}

// Boundaries align by the rows on disk, skipping unselected rows
TEST(DataPartitionTest, SplitSelectedRowsAligned) {
  auto result_shape =
      ResultShapeData{"DATA", IPosition{2, 3, 9}, 3, DataType::TpComplex, std::nullopt};
  auto selection = SelectionBuilder()
                       .Order('F')
                       .AddEmpty()
                       .AddEmpty()
                       .Add({11, 0, 1, 2, -1, 5, 6, 9, 10})
                       .Build();

  // The even boundary is before row 6, in the unit starting at row 4.
  // The first selected row of that unit is 5
  ASSERT_OK_AND_ASSIGN(
      auto partition,
      DataPartition::Make(selection, result_shape, RowSplitOptions{2, 0, UnitOfFour}));
  ASSERT_EQ(partition.nChunks(), 3);
  EXPECT_THAT(partition.Chunk(0).Disk(2), ::testing::ElementsAre(-1));
  EXPECT_THAT(partition.Chunk(1).Disk(2), ::testing::ElementsAre(0, 1, 2));
  EXPECT_THAT(partition.Chunk(1).Mem(2), ::testing::ElementsAre(1, 2, 3));
  EXPECT_THAT(partition.Chunk(2).Disk(2), ::testing::ElementsAre(5, 6, 9, 10, 11));
  EXPECT_THAT(partition.Chunk(2).Mem(2), ::testing::ElementsAre(5, 6, 7, 8, 0));
}

class RowUnitStartTest : public ::testing::Test {
 protected:
  std::filesystem::path dir_;

  void SetUp() override {
    auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    dir_ = std::filesystem::temp_directory_path() /
           (info->name() + "-"s + arcae::hexuuid(4));
    std::filesystem::create_directories(dir_);
  }

  void TearDown() override { std::filesystem::remove_all(dir_); }

  std::string Path(const std::string& name) { return (dir_ / name).string(); }
};

constexpr std::size_t kUnitRows = 12;

TEST_F(RowUnitStartTest, TiledColumnStMan) {
  TableDesc td;
  td.addColumn(ArrayColumnDesc<Complex>("DATA", IPosition{2, 3}, ColumnDesc::FixedShape));
  SetupNewTable setup(Path("tcsm.table"), td, Table::New);
  casacore::TiledColumnStMan sm("TiledData", IPosition{2, 3, 4});
  setup.bindColumn("DATA", sm);
  Table table(setup, kUnitRows);

  auto unit_start = MakeRowUnitStart(table, "DATA");
  ASSERT_TRUE(unit_start);
  for (::casacore::rownr_t row = 0; row < kUnitRows; ++row) {
    EXPECT_EQ(unit_start(row), row - row % 4) << row;
  }
}

// Each shape of a TiledShapeStMan column has its own hypercube,
// whose tiles start at its first row
TEST_F(RowUnitStartTest, TiledShapeStMan) {
  TableDesc td;
  td.addColumn(ArrayColumnDesc<Complex>("DATA", 2));
  SetupNewTable setup(Path("tssm.table"), td, Table::New);
  casacore::TiledShapeStMan sm("TiledData", IPosition{2, 3, 4});
  setup.bindColumn("DATA", sm);
  Table table(setup, kUnitRows);
  casacore::ArrayColumn<Complex> data(table, "DATA");
  for (::casacore::rownr_t row = 0; row < kUnitRows; ++row) {
    data.put(row, casacore::Array<Complex>(row < 6 ? IPosition{2, 3} : IPosition{2, 2}));
  }

  auto unit_start = MakeRowUnitStart(table, "DATA");
  ASSERT_TRUE(unit_start);
  EXPECT_EQ(unit_start(3), 0);
  EXPECT_EQ(unit_start(5), 4);
  EXPECT_EQ(unit_start(6), 6);
  EXPECT_EQ(unit_start(9), 6);
  EXPECT_EQ(unit_start(10), 10);
  EXPECT_EQ(unit_start(11), 10);
}

// TiledDataStMan can store rows in several axes of a hypercube. Here rows
// step through 4 baselines, then 6 times, and tiles span 2 baselines and
// 3 times, so rows 0-11 and rows 12-23 hold whole tiles
TEST_F(RowUnitStartTest, TiledDataStMan) {
  constexpr std::size_t kNBaseline = 4;
  constexpr std::size_t kNTime = 6;
  TableDesc td;
  td.addColumn(ArrayColumnDesc<Complex>("DATA", IPosition{2}, ColumnDesc::FixedShape));
  td.defineHypercolumn("TiledData", 3, casacore::Vector<casacore::String>{"DATA"});
  SetupNewTable setup(Path("tdsm.table"), td, Table::New);
  casacore::TiledDataStMan sm("TiledData");
  setup.bindColumn("DATA", sm);
  Table table(setup, kNBaseline * kNTime);
  casacore::TiledDataStManAccessor accessor(table, "TiledData");
  accessor.addHypercube(IPosition{2, kNBaseline, kNTime}, IPosition{2, 2, 3},
                        casacore::Record());

  auto unit_start = MakeRowUnitStart(table, "DATA");
  ASSERT_TRUE(unit_start);
  for (::casacore::rownr_t row = 0; row < kNBaseline * kNTime; ++row) {
    EXPECT_EQ(unit_start(row), row < 12 ? 0 : 12) << row;
  }
}

// Dysco compresses rows in blocks of one timestep
TEST_F(RowUnitStartTest, DyscoStMan) {
  constexpr std::size_t kNAnt = 4;
  constexpr std::size_t kNBaselines = kNAnt * (kNAnt - 1) / 2;
  constexpr std::size_t kNTime = 3;
  auto path = Path("dysco.table");
  {
    TableDesc td;
    ArrayColumnDesc<Complex> data_desc("DATA", "", "DyscoStMan", "", IPosition{2, 2});
    data_desc.setOptions(ColumnDesc::Direct | ColumnDesc::FixedShape);
    td.addColumn(data_desc);
    for (auto name : {"ANTENNA1", "ANTENNA2", "FIELD_ID", "DATA_DESC_ID"}) {
      td.addColumn(ScalarColumnDesc<casacore::Int>(name));
    }
    td.addColumn(ScalarColumnDesc<casacore::Double>("TIME"));
    SetupNewTable setup(path, td, Table::New);
    casacore::Record spec;
    spec.define("distribution", "TruncatedGaussian");
    spec.define("normalization", "AF");
    spec.define("distributionTruncation", 2.0);
    spec.define("dataBitCount", 10);
    spec.define("weightBitCount", 12);
    std::unique_ptr<casacore::DataManager> dysco(
        casacore::DataManager::getCtor("DyscoStMan")("DyscoData", spec));
    setup.bindColumn("DATA", *dysco);
    Table table(setup, kNBaselines * kNTime);

    casacore::ScalarColumn<casacore::Int> ant1(table, "ANTENNA1");
    casacore::ScalarColumn<casacore::Int> ant2(table, "ANTENNA2");
    casacore::ScalarColumn<casacore::Int> field(table, "FIELD_ID");
    casacore::ScalarColumn<casacore::Int> ddid(table, "DATA_DESC_ID");
    casacore::ScalarColumn<casacore::Double> time(table, "TIME");
    for (std::size_t t = 0, row = 0; t < kNTime; ++t) {
      for (std::size_t a1 = 0; a1 < kNAnt; ++a1) {
        for (std::size_t a2 = a1 + 1; a2 < kNAnt; ++a2, ++row) {
          ant1.put(row, a1);
          ant2.put(row, a2);
          field.put(row, 0);
          ddid.put(row, 0);
          time.put(row, t);
        }
      }
    }
    casacore::ArrayColumn<Complex> data(table, "DATA");
    for (std::size_t row = 0; row < kNBaselines * kNTime; ++row) {
      data.put(row, casacore::Array<Complex>(IPosition{2, 2}, Complex(row, 0)));
    }
  }

  Table table(path);
  auto unit_start = MakeRowUnitStart(table, "DATA");
  ASSERT_TRUE(unit_start);
  for (::casacore::rownr_t row = 0; row < kNBaselines * kNTime; ++row) {
    EXPECT_EQ(unit_start(row), row - row % kNBaselines) << row;
  }
}

// Storage managers without units worth aligning to, and reference tables
TEST_F(RowUnitStartTest, Unaligned) {
  TableDesc td;
  td.addColumn(ArrayColumnDesc<Complex>("DATA", IPosition{2, 3}, ColumnDesc::FixedShape));
  td.addColumn(
      ArrayColumnDesc<Complex>("TILED", IPosition{2, 3}, ColumnDesc::FixedShape));
  td.addColumn(ArrayColumnDesc<Complex>("CELL", IPosition{2, 3}, ColumnDesc::FixedShape));
  td.defineHypercolumn("TiledCell", 2, casacore::Vector<casacore::String>{"CELL"});
  SetupNewTable setup(Path("ssm.table"), td, Table::New);
  casacore::StandardStMan ssm;
  casacore::TiledColumnStMan tsm("TiledData", IPosition{2, 3, 4});
  casacore::TiledCellStMan tcsm("TiledCell", IPosition{2, 3});
  setup.bindColumn("DATA", ssm);
  setup.bindColumn("TILED", tsm);
  setup.bindColumn("CELL", tcsm);
  Table table(setup, kUnitRows);

  EXPECT_FALSE(MakeRowUnitStart(table, "DATA"));
  EXPECT_TRUE(MakeRowUnitStart(table, "TILED"));
  // Each row has its own hypercube
  EXPECT_FALSE(MakeRowUnitStart(table, "CELL"));
  // Rows of a reference table are numbered differently from its storage managers
  Table reference = table(casacore::Vector<::casacore::rownr_t>{1, 2, 3});
  EXPECT_FALSE(MakeRowUnitStart(reference, "TILED"));
}

}  // namespace
