// Documents the array shapes that ArrayColumn::getSlice/putSlice accept, and the
// partitioner invariant that the single-row slice paths in read_impl.cc and
// write_impl.cc depend upon.
#include <casacore/casa/Arrays.h>
#include <casacore/tables/Tables.h>
#include <casacore/tables/Tables/TableUtil.h>

#include <gtest/gtest.h>

#include <tests/test_utils.h>
#include <string>
#include <vector>

using ::casacore::Array;
using ::casacore::ArrayColumn;
using ::casacore::ArrayColumnDesc;
using ::casacore::ColumnDesc;
using ::casacore::SetupNewTable;
using ::casacore::Slicer;
using ::casacore::Table;
using ::casacore::TableDesc;
using ::casacore::Vector;
using IPos = ::casacore::IPosition;

using namespace std::string_literals;

static constexpr std::size_t knrow = 4;
static constexpr std::size_t kncorr = 2;
static constexpr std::size_t knchan = 3;

class SliceShapeTests : public ::testing::Test {
 protected:
  std::string table_name_;

  void SetUp() override {
    auto* ti = ::testing::UnitTest::GetInstance()->current_test_info();
    table_name_ = std::string(ti->name() + "-"s + arcae::hexuuid(4) + ".table"s);
    auto td = TableDesc();
    td.addColumn(ArrayColumnDesc<casacore::Int>("DATA", IPos({kncorr, knchan}),
                                                ColumnDesc::FixedShape));
    auto snt = SetupNewTable(table_name_, td, Table::New);
    auto table = Table(snt, knrow);
    auto data = Array<casacore::Int>(IPos({kncorr, knchan, knrow}));
    for (auto [i, it] = std::tuple{0, data.begin()}; it != data.end(); ++it, ++i) *it = i;
    ArrayColumn<casacore::Int>(table, "DATA").putColumn(data);
  }

  void TearDown() override { casacore::TableUtil::deleteTable(table_name_); }

  Table Open(Table::TableOption opt = Table::Old) { return Table(table_name_, opt); }
};

// Slicer over the full cell: this is what arcae's DataChunk::SectionSlicer()
// produces for a single row -- (nDim() - 1) dimensions, no row axis.
static Slicer FullCellSlicer() {
  return Slicer(IPos({0, 0}), IPos({kncorr - 1, knchan - 1}), Slicer::endIsLast);
}

TEST_F(SliceShapeTests, GetSliceExactShapeShared) {
  auto table = Open();
  auto column = ArrayColumn<casacore::Int>(table, "DATA");
  auto slicer = FullCellSlicer();
  // slicer.length() is the N-dim slice shape, NOT a flat element count
  EXPECT_EQ(slicer.length(), IPos({kncorr, knchan}));

  std::vector<casacore::Int> buffer(kncorr * knchan, -1);
  auto arr = Array<casacore::Int>(slicer.length(), buffer.data(), casacore::SHARE);
  ASSERT_NO_THROW(column.getSlice(0, slicer, arr));
  // Data landed in the caller's buffer
  EXPECT_EQ(buffer, std::vector<casacore::Int>({0, 1, 2, 3, 4, 5}));
}

TEST_F(SliceShapeTests, GetSliceFlattenedThrows) {
  auto table = Open();
  auto column = ArrayColumn<casacore::Int>(table, "DATA");
  auto slicer = FullCellSlicer();

  std::vector<casacore::Int> buffer(kncorr * knchan, -1);
  // 1-D array, correct number of elements, wrong ndim
  auto flat =
      Array<casacore::Int>(IPos({kncorr * knchan}), buffer.data(), casacore::SHARE);
  EXPECT_THROW(column.getSlice(0, slicer, flat), casacore::TableArrayConformanceError);
  // Nothing was written
  EXPECT_EQ(buffer, std::vector<casacore::Int>(kncorr * knchan, -1));
}

TEST_F(SliceShapeTests, GetSliceFlattenedWithResizeSilentlyDetaches) {
  auto table = Open();
  auto column = ArrayColumn<casacore::Int>(table, "DATA");
  auto slicer = FullCellSlicer();

  std::vector<casacore::Int> buffer(kncorr * knchan, -1);
  auto flat =
      Array<casacore::Int>(IPos({kncorr * knchan}), buffer.data(), casacore::SHARE);
  // resize = True: no exception, but the Array reallocates and abandons `buffer`
  ASSERT_NO_THROW(column.getSlice(0, slicer, flat, /*resize=*/true));
  EXPECT_EQ(flat.shape(), slicer.length());
  EXPECT_NE(flat.data(), buffer.data());
  EXPECT_EQ(buffer, std::vector<casacore::Int>(kncorr * knchan, -1));
}

TEST_F(SliceShapeTests, GetSliceDegenerateRowAxisThrows) {
  auto table = Open();
  auto column = ArrayColumn<casacore::Int>(table, "DATA");
  auto slicer = FullCellSlicer();

  std::vector<casacore::Int> buffer(kncorr * knchan, -1);
  // This is arcae's DataChunk::GetShape() for a single row: cell dims + trailing
  // row axis of length 1. Same element count, one extra degenerate axis.
  auto arr =
      Array<casacore::Int>(IPos({kncorr, knchan, 1}), buffer.data(), casacore::SHARE);
  EXPECT_THROW(column.getSlice(0, slicer, arr), casacore::TableArrayConformanceError);
}

TEST_F(SliceShapeTests, GetSliceEmptyArrayResizesRegardless) {
  auto table = Open();
  auto column = ArrayColumn<casacore::Int>(table, "DATA");
  auto arr = Array<casacore::Int>();  // nelements() == 0
  ASSERT_NO_THROW(column.getSlice(0, FullCellSlicer(), arr));
  EXPECT_EQ(arr.shape(), IPos({kncorr, knchan}));
}

TEST_F(SliceShapeTests, PutSliceExactShapeShared) {
  auto table = Open(Table::Update);
  auto column = ArrayColumn<casacore::Int>(table, "DATA");
  auto slicer = FullCellSlicer();

  std::vector<casacore::Int> buffer{10, 11, 12, 13, 14, 15};
  auto arr = Array<casacore::Int>(slicer.length(), buffer.data(), casacore::SHARE);
  ASSERT_NO_THROW(column.putSlice(1, slicer, arr));
  EXPECT_TRUE(casacore::allEQ(column.get(1), arr));
}

TEST_F(SliceShapeTests, PutSliceFlattenedThrows) {
  auto table = Open(Table::Update);
  auto column = ArrayColumn<casacore::Int>(table, "DATA");
  auto slicer = FullCellSlicer();

  auto flat = Vector<casacore::Int>(IPos({kncorr * knchan}), casacore::Int(7));
  // putSlice has no resize escape hatch at all
  EXPECT_THROW(column.putSlice(1, slicer, flat), casacore::TableArrayConformanceError);
}

TEST_F(SliceShapeTests, PutSliceDegenerateRowAxisThrows) {
  auto table = Open(Table::Update);
  auto column = ArrayColumn<casacore::Int>(table, "DATA");
  auto slicer = FullCellSlicer();

  auto arr = Array<casacore::Int>(IPos({kncorr, knchan, 1}), casacore::Int(7));
  EXPECT_THROW(column.putSlice(1, slicer, arr), casacore::TableArrayConformanceError);
  // reform() to the slicer shape is what makes it acceptable
  ASSERT_NO_THROW(column.putSlice(1, slicer, arr.reform(slicer.length())));
}

TEST_F(SliceShapeTests, SlicerNdimMustMatchCellNdim) {
  auto table = Open();
  auto column = ArrayColumn<casacore::Int>(table, "DATA");
  // A slicer with fewer dimensions than the cell is rejected before any
  // array-shape check happens.
  auto short_slicer = Slicer(IPos({0}), IPos({kncorr - 1}), Slicer::endIsLast);
  auto arr = Array<casacore::Int>(IPos({kncorr}));
  EXPECT_THROW(column.getSlice(0, short_slicer, arr), casacore::ArraySlicerError);
}

// ---------------------------------------------------------------------------
// The invariant PR #233 relies on: for every chunk the partitioner produces,
// SectionSlicer().length() is exactly GetShape() with the row axis dropped.
// If a chunk's secondary disk span were ever strided, the slicer (a stride-1
// bounding box from disk[0] to disk[last]) would be LARGER than the chunk,
// and the SHARE'd Array in read_impl.cc would overrun the output buffer.
// ---------------------------------------------------------------------------
#include <arcae/data_partition.h>
#include <arcae/result_shape.h>
#include <arcae/selection.h>

#include <arrow/testing/gtest_util.h>
#include <optional>

using ::arcae::detail::DataPartition;
using ::arcae::detail::ResultShapeData;
using ::arcae::detail::SelectionBuilder;

namespace {

void CheckSlicerMatchesChunk(const DataPartition& partition) {
  for (std::size_t c = 0; c < partition.nChunks(); ++c) {
    const auto& chunk = partition.Chunk(c);
    if (chunk.IsEmpty()) continue;
    auto shape = chunk.GetShape();
    auto cell_shape = shape.getFirst(shape.size() - 1);
    auto length = chunk.SectionSlicer().length();
    EXPECT_EQ(length, cell_shape) << "chunk " << c << ": " << chunk.ToString();
    // Element count agrees for the single-row case read_impl/write_impl special-case
    if (chunk.ReferenceRows().nrows() == 1) {
      EXPECT_EQ(std::size_t(length.product()), chunk.nElements());
    }
  }
}

}  // namespace

TEST(SliceShapeInvariant, FixedShapeStridedSecondaryDims) {
  auto result_shape = ResultShapeData{"DATA", IPos{4, 8, 10}, 3,
                                      casacore::DataType::TpComplex, std::nullopt};
  auto selection = SelectionBuilder()
                       .Order('F')
                       .Add({0, 2, 3})     // strided then contiguous
                       .Add({7, 1, 2, 5})  // unsorted, discontiguous
                       .Add({4})           // single row
                       .Build();
  ASSERT_OK_AND_ASSIGN(auto partition, DataPartition::Make(selection, result_shape));
  CheckSlicerMatchesChunk(partition);
}

TEST(SliceShapeInvariant, FixedShapeEveryOtherIndex) {
  auto result_shape = ResultShapeData{"DATA", IPos{6, 6, 3}, 3,
                                      casacore::DataType::TpComplex, std::nullopt};
  auto selection = SelectionBuilder()
                       .Order('F')
                       .Add({0, 2, 4})  // stride 2 -> must split into 3 chunks
                       .Add({1, 3, 5})
                       .Add({1})
                       .Build();
  ASSERT_OK_AND_ASSIGN(auto partition, DataPartition::Make(selection, result_shape));
  // One chunk per (dim0 run, dim1 run) pair: no chunk may span a stride
  EXPECT_EQ(partition.nChunks(), 9);
  CheckSlicerMatchesChunk(partition);
}

TEST(SliceShapeInvariant, VaryingShapeSingleRow) {
  // NOTE: ResultShapeData's per-row shapes are indexed by row memory index, so
  // the number of shapes must equal the size of the row selection.
  auto shapes = std::vector<IPos>{IPos{4, 5}};
  auto result_shape =
      ResultShapeData{"DATA", std::nullopt, 3, casacore::DataType::TpComplex, shapes};
  auto selection =
      SelectionBuilder().Order('F').Add({0, 2, 3}).Add({1, 2, 4}).Add({1}).Build();
  ASSERT_OK_AND_ASSIGN(auto partition, DataPartition::Make(selection, result_shape));
  CheckSlicerMatchesChunk(partition);
}

TEST(SliceShapeInvariant, VaryingShapeMultiRow) {
  auto shapes = std::vector<IPos>{IPos{4, 5}, IPos{2, 6}};
  auto result_shape =
      ResultShapeData{"DATA", std::nullopt, 3, casacore::DataType::TpComplex, shapes};
  auto selection =
      SelectionBuilder().Order('F').Add({0, 1, 3}).Add({4, 1, 2}).Add({3, 0}).Build();
  ASSERT_OK_AND_ASSIGN(auto partition, DataPartition::Make(selection, result_shape));
  CheckSlicerMatchesChunk(partition);
}

TEST(SliceShapeInvariant, NoSecondarySelection) {
  auto result_shape = ResultShapeData{"DATA", IPos{4, 8, 10}, 3,
                                      casacore::DataType::TpComplex, std::nullopt};
  auto selection = SelectionBuilder().Order('F').AddEmpty().AddEmpty().Add({7}).Build();
  ASSERT_OK_AND_ASSIGN(auto partition, DataPartition::Make(selection, result_shape));
  EXPECT_EQ(partition.nChunks(), 1);
  CheckSlicerMatchesChunk(partition);
}
