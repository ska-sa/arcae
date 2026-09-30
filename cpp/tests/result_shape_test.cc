#include <arrow/json/from_string.h>
#include <arrow/testing/builder.h>
#include <arrow/testing/gtest_util.h>
#include <arrow/type_fwd.h>

#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/BasicSL/Complexfwd.h>
#include <casacore/casa/Utilities/DataType.h>
#include <casacore/casa/aipsxtype.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/tables/Tables.h>
#include <casacore/tables/Tables/ArrColDesc.h>
#include <casacore/tables/Tables/RefRows.h>
#include <casacore/tables/Tables/TableColumn.h>
#include <casacore/tables/Tables/TableProxy.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "arcae/result_shape.h"
#include "arcae/selection.h"

#include <tests/test_utils.h>

using ::arcae::GetArrayColumn;
using ::arcae::GetScalarColumn;
using ::arcae::detail::IndexType;
using ::arcae::detail::ResultShapeData;
using ::arcae::detail::Selection;
using ::arcae::detail::SelectionBuilder;

using ::arrow::json::ArrayFromJSONString;

using casacore::Array;
using casacore::ArrayColumnDesc;
using casacore::ColumnDesc;
using casacore::Complex;
using casacore::DataType;
using MS = casacore::MeasurementSet;
using MSColumns = casacore::MSMainEnums::PredefinedColumns;
using casacore::SetupNewTable;
using casacore::Table;
using casacore::TableDesc;
using casacore::TableProxy;
using casacore::TiledColumnStMan;
using IPos = casacore::IPosition;

using namespace std::string_literals;

static constexpr std::size_t knrow = 10;
static constexpr std::size_t knchan = 4;
static constexpr std::size_t kncorr = 2;

namespace {

// Create a zeroed complex result array of C-ordered shape (nrow, nchan, ncorr)
arrow::Result<std::shared_ptr<arrow::Array>> MakeComplexResult(std::size_t nrow,
                                                               std::size_t nchan,
                                                               std::size_t ncorr) {
  std::shared_ptr<arrow::Array> result;
  std::vector<float> values(nrow * nchan * ncorr * 2, 0.0);
  arrow::ArrayFromVector<arrow::FloatType>(arrow::float32(), values, &result);
  ARROW_ASSIGN_OR_RAISE(result, arrow::FixedSizeListArray::FromArrays(result, 2));
  ARROW_ASSIGN_OR_RAISE(result, arrow::FixedSizeListArray::FromArrays(result, ncorr));
  return arrow::FixedSizeListArray::FromArrays(result, nchan);
}

class ResultShapeTest : public ::testing::Test {
 protected:
  TableProxy table_proxy_;
  std::string table_name_;
  std::size_t nelements_;
  std::size_t nsparse_elements_;

  void SetUp() override {
    auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    table_name_ = std::string(test_info->name() + "-"s + arcae::hexuuid(4) + ".table"s);

    auto table_desc = TableDesc(MS::requiredTableDesc());
    auto data_shape = IPos({kncorr, knchan});
    auto tile_shape = IPos({kncorr, knchan, 1});
    auto data_column_desc =
        ArrayColumnDesc<Complex>("MODEL_DATA", data_shape, ColumnDesc::FixedShape);

    auto var_column_desc = ArrayColumnDesc<Complex>("VAR_DATA", 2);

    auto var_fixed_column_desc = ArrayColumnDesc<Complex>("VAR_FIXED_DATA", 2);

    auto var_sparse_desc = ArrayColumnDesc<Complex>("VAR_SPARSE", 2);

    table_desc.addColumn(data_column_desc);
    table_desc.addColumn(var_column_desc);
    table_desc.addColumn(var_fixed_column_desc);
    table_desc.addColumn(var_sparse_desc);
    auto storage_manager = TiledColumnStMan("TiledModelData", tile_shape);
    auto setup_new_table = SetupNewTable(table_name_, table_desc, Table::New);
    setup_new_table.bindColumn("MODEL_DATA", storage_manager);
    auto ms = MS(setup_new_table, knrow);

    auto field = GetScalarColumn<casacore::Int>(ms, MS::FIELD_ID);
    auto ddid = GetScalarColumn<casacore::Int>(ms, MS::DATA_DESC_ID);
    auto scan = GetScalarColumn<casacore::Int>(ms, MS::SCAN_NUMBER);
    auto time = GetScalarColumn<casacore::Double>(ms, MS::TIME);
    auto ant1 = GetScalarColumn<casacore::Int>(ms, MS::ANTENNA1);
    auto ant2 = GetScalarColumn<casacore::Int>(ms, MS::ANTENNA2);
    auto data = GetArrayColumn<Complex>(ms, MS::MODEL_DATA);
    auto var_data = GetArrayColumn<Complex>(ms, "VAR_DATA");
    auto var_fixed_data = GetArrayColumn<Complex>(ms, "VAR_FIXED_DATA");
    auto var_sparse_data = GetArrayColumn<Complex>(ms, "VAR_SPARSE");

    time.putColumn({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0});
    field.putColumn({0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    ddid.putColumn({0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    ant1.putColumn({0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    ant2.putColumn({1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    data.putColumn(Array<Complex>(IPos({kncorr, knchan, knrow}), {1, 2}));
    var_fixed_data.putColumn(Array<Complex>(IPos({kncorr, knchan, knrow}), {1, 2}));

    auto var_shapes =
        std::vector<IPos>{{3, 2, 1}, {4, 1, 1}, {4, 2, 1}, {2, 2, 1}, {2, 1, 1},
                          {3, 2, 1}, {4, 1, 1}, {4, 2, 1}, {2, 2, 1}, {2, 1, 1}};

    auto var_sparse_shapes =
        std::vector<IPos>{{3, 2, 1}, {0, 0, 0}, {4, 2, 1}, {2, 2, 1}, {2, 1, 1},
                          {3, 2, 1}, {4, 1, 1}, {0, 0, 0}, {2, 2, 1}, {2, 1, 1}};

    assert(var_shapes.size() == knrow);

    nelements_ = std::accumulate(
        std::begin(var_shapes), std::end(var_shapes), std::size_t{0},
        [](auto init, auto& shape) -> std::size_t { return init + shape.product(); });

    nsparse_elements_ = std::accumulate(
        std::begin(var_sparse_shapes), std::end(var_sparse_shapes), std::size_t{0},
        [](auto init, auto& shape) -> std::size_t { return init + shape.product(); });

    for (std::size_t i = 0; i < knrow; ++i) {
      auto fv = static_cast<float>(i);
      auto corrected_array = Array<Complex>(var_shapes[i], {fv, fv});
      var_data.putColumnCells(casacore::RefRows(i, i), corrected_array);

      auto sparse_array = Array<Complex>(var_sparse_shapes[i], {fv, fv});
      if (sparse_array.size() > 0)
        var_sparse_data.putColumnCells(casacore::RefRows(i, i), sparse_array);
    }

    table_proxy_ = TableProxy(ms);
  }
};

TEST_F(ResultShapeTest, ReadFixed) {
  auto fixed = GetArrayColumn<Complex>(table_proxy_.table(), "MODEL_DATA");
  ASSERT_OK_AND_ASSIGN(auto shape_data, ResultShapeData::MakeRead(fixed));

  EXPECT_EQ(shape_data.GetName(), "MODEL_DATA");
  EXPECT_TRUE(shape_data.IsFixed());
  EXPECT_EQ(shape_data.GetShape(), IPos({2, 4, 10}));
  EXPECT_EQ(shape_data.nDim(), 3);
  EXPECT_EQ(shape_data.nRows(), 10);
  EXPECT_EQ(shape_data.nElements(), 80);
  EXPECT_EQ(shape_data.GetDataType(), DataType::TpComplex);
  ASSERT_OK_AND_ASSIGN(auto offsets, shape_data.GetOffsets());
}

TEST_F(ResultShapeTest, ReadFixedSelection) {
  auto fixed = GetArrayColumn<Complex>(table_proxy_.table(), "MODEL_DATA");
  auto sel = SelectionBuilder::FromInit({{0, 1}});
  ASSERT_OK_AND_ASSIGN(auto shape_data, ResultShapeData::MakeRead(fixed, sel));

  EXPECT_EQ(shape_data.GetName(), "MODEL_DATA");
  EXPECT_TRUE(shape_data.IsFixed());
  EXPECT_EQ(shape_data.GetShape(), IPos({2, 4, 2}));
  EXPECT_EQ(shape_data.nDim(), 3);
  EXPECT_EQ(shape_data.nRows(), 2);
  EXPECT_EQ(shape_data.nElements(), 16);
  EXPECT_EQ(shape_data.GetDataType(), DataType::TpComplex);
  ASSERT_OK_AND_ASSIGN(auto offsets, shape_data.GetOffsets());

  sel = SelectionBuilder::FromInit({{0, 1}, {0, 1}, {0}});
  ASSERT_OK_AND_ASSIGN(shape_data, ResultShapeData::MakeRead(fixed, sel));

  EXPECT_EQ(shape_data.GetName(), "MODEL_DATA");
  EXPECT_TRUE(shape_data.IsFixed());
  EXPECT_EQ(shape_data.GetShape(), IPos({1, 2, 2}));
  EXPECT_EQ(shape_data.nDim(), 3);
  EXPECT_EQ(shape_data.nRows(), 2);
  EXPECT_EQ(shape_data.nElements(), 4);
  EXPECT_EQ(shape_data.GetDataType(), DataType::TpComplex);
  ASSERT_OK_AND_ASSIGN(offsets, shape_data.GetOffsets());
}

TEST_F(ResultShapeTest, ReadVariable) {
  auto var = GetArrayColumn<Complex>(table_proxy_.table(), "VAR_DATA");
  ASSERT_OK_AND_ASSIGN(auto shape_data, ResultShapeData::MakeRead(var));

  EXPECT_EQ(shape_data.GetName(), "VAR_DATA");
  EXPECT_FALSE(shape_data.IsFixed());
  EXPECT_EQ(shape_data.nDim(), 3);
  EXPECT_EQ(shape_data.nRows(), 10);
  EXPECT_EQ(shape_data.GetRowShape(0), IPos({3, 2}));
  EXPECT_EQ(shape_data.GetRowShape(9), IPos({2, 1}));
  EXPECT_EQ(shape_data.nElements(), nelements_);
  EXPECT_EQ(shape_data.GetDataType(), DataType::TpComplex);
  ASSERT_OK_AND_ASSIGN(auto offsets, shape_data.GetOffsets());
}

TEST_F(ResultShapeTest, ReadSparseVariable) {
  auto var = GetArrayColumn<Complex>(table_proxy_.table(), "VAR_SPARSE");
  ASSERT_OK_AND_ASSIGN(auto shape_data,
                       ResultShapeData::MakeRead(var, Selection(), true));

  EXPECT_EQ(shape_data.GetName(), "VAR_SPARSE");
  EXPECT_FALSE(shape_data.IsFixed());
  EXPECT_EQ(shape_data.nDim(), 3);
  EXPECT_EQ(shape_data.nRows(), 10);
  EXPECT_EQ(shape_data.GetRowShape(0), IPos({3, 2}));
  EXPECT_EQ(shape_data.GetRowShape(1), IPos());
  EXPECT_EQ(shape_data.GetRowShape(7), IPos());
  EXPECT_EQ(shape_data.GetRowShape(9), IPos({2, 1}));
  EXPECT_EQ(shape_data.nElements(), nsparse_elements_);
  EXPECT_EQ(shape_data.GetDataType(), DataType::TpComplex);
  ASSERT_OK_AND_ASSIGN(auto offsets, shape_data.GetOffsets());
}

TEST_F(ResultShapeTest, NegateSparseVariable) {
  auto var = GetArrayColumn<Complex>(table_proxy_.table(), "VAR_SPARSE");

  // Create a result array filled with a default value
  std::shared_ptr<arrow::Array> result;
  std::vector<float> values(knrow * knchan * kncorr * 2, 0.0);
  arrow::ArrayFromVector<arrow::FloatType>(arrow::float32(), values, &result);
  ASSERT_OK_AND_ASSIGN(result, arrow::FixedSizeListArray::FromArrays(result, 2));
  ASSERT_OK_AND_ASSIGN(result, arrow::FixedSizeListArray::FromArrays(result, kncorr));
  ASSERT_OK_AND_ASSIGN(result, arrow::FixedSizeListArray::FromArrays(result, knchan));
  EXPECT_EQ(result->length(), knrow);

  ASSERT_OK_AND_ASSIGN(auto shape_data, ResultShapeData::FromArray(var, result));
  ASSERT_OK_AND_ASSIGN(auto selection,
                       shape_data.NegateMissingSelectedRows(var, Selection()));
  EXPECT_TRUE(selection.HasRowSpan());
  auto span = selection.GetRowSpan();
  for (std::size_t r = 0; r < span.size(); ++r) {
    if (r == 1 || r == 7)
      EXPECT_EQ(span[r], -1);
    else
      EXPECT_EQ(span[r], r);
  }
}

TEST_F(ResultShapeTest, NegatePadsAndTruncates) {
  auto fixed = GetArrayColumn<Complex>(table_proxy_.table(), "MODEL_DATA");
  auto var = GetArrayColumn<Complex>(table_proxy_.table(), "VAR_DATA");
  auto var_fixed = GetArrayColumn<Complex>(table_proxy_.table(), "VAR_FIXED_DATA");

  struct NegateResult {
    ResultShapeData shape_data;
    Selection selection;
  };

  auto Negate = [](const auto& column, std::size_t nrow, std::size_t nchan,
                   std::size_t ncorr,
                   const Selection& sel) -> arrow::Result<NegateResult> {
    ARROW_ASSIGN_OR_RAISE(auto result, MakeComplexResult(nrow, nchan, ncorr));
    ARROW_ASSIGN_OR_RAISE(auto shape_data, ResultShapeData::FromArray(column, result));
    ARROW_ASSIGN_OR_RAISE(auto selection,
                          shape_data.NegateMissingSelectedRows(column, sel));
    return NegateResult{std::move(shape_data), std::move(selection)};
  };

  // Get the FORTRAN ordered selection indices in dimension dim, if any
  auto Ids = [](const Selection& sel, std::size_t dim) -> std::vector<IndexType> {
    if (auto res = sel.FSpan(dim, 3); res.ok()) {
      auto span = res.ValueOrDie();
      return {std::begin(span), std::end(span)};
    }
    return {};
  };

  using Ids_ = std::vector<IndexType>;
  auto rows = SelectionBuilder::FromInit({{0, 1}});
  auto no_rows = Selection();

  for (const auto& column : {fixed, var_fixed}) {
    for (const auto& sel : {no_rows, rows}) {
      std::size_t nrow = sel.HasRowSpan() ? 2 : knrow;

      // Exact and undersized results leave the selection untouched
      for (auto [nchan, ncorr] :
           {std::pair{knchan, kncorr}, std::pair{knchan - 1, kncorr},
            std::pair{knchan, kncorr - 1}}) {
        ASSERT_OK_AND_ASSIGN(auto r, Negate(column, nrow, nchan, ncorr, sel));
        EXPECT_FALSE(r.shape_data.HasCellBounds());
        EXPECT_EQ(Ids(r.selection, 0), Ids(sel, 0));
        EXPECT_EQ(Ids(r.selection, 1), Ids(sel, 1));
        EXPECT_EQ(Ids(r.selection, 2), Ids(sel, 2));
      }

      // Oversized results pad the selection with -1
      ASSERT_OK_AND_ASSIGN(auto r, Negate(column, nrow, knchan + 2, kncorr + 1, sel));
      EXPECT_FALSE(r.shape_data.HasCellBounds());
      EXPECT_EQ(Ids(r.selection, 0), Ids_({0, 1, -1}));
      EXPECT_EQ(Ids(r.selection, 1), Ids_({0, 1, 2, 3, -1, -1}));
      EXPECT_EQ(Ids(r.selection, 2), Ids(sel, 2));
    }

    // Explicit selections are preserved, other dimensions are padded
    auto chan_sel = SelectionBuilder::FromInit({{0, 1}, {0, 2}});
    ASSERT_OK_AND_ASSIGN(auto r, Negate(column, 2, 2, kncorr + 1, chan_sel));
    EXPECT_EQ(Ids(r.selection, 0), Ids_({0, 1, -1}));
    EXPECT_EQ(Ids(r.selection, 1), Ids_({0, 2}));
    EXPECT_EQ(Ids(r.selection, 2), Ids_({0, 1}));

    // Explicit -1 padding still works
    auto pad_sel = SelectionBuilder::FromInit({{0, 1}, {0, 1, 2, 3, -1, -1}});
    ASSERT_OK(Negate(column, 2, knchan + 2, kncorr, pad_sel));
  }

  // Selections past the end of a fixed shape cell are an error
  auto past_sel = SelectionBuilder::FromInit({{0, 1}, {0, int(knchan)}});
  ASSERT_RAISES(IndexError, Negate(fixed, 2, 2, kncorr, past_sel));

  // but are not read from variably shaped cells
  ASSERT_OK_AND_ASSIGN(auto past, Negate(var_fixed, 2, 2, kncorr, past_sel));
  EXPECT_FALSE(past.shape_data.HasCellBounds());
  EXPECT_EQ(Ids(past.selection, 0), Ids_{});
  EXPECT_EQ(Ids(past.selection, 1), Ids_({0, -1}));
  EXPECT_EQ(Ids(past.selection, 2), Ids_({0, 1}));

  // Variably shaped rows 2 and 7 share shape (4, 2),
  // so the selection is padded
  ASSERT_OK_AND_ASSIGN(auto r,
                       Negate(var, 2, 3, 6, SelectionBuilder::FromInit({{2, 7}})));
  EXPECT_FALSE(r.shape_data.HasCellBounds());
  EXPECT_EQ(Ids(r.selection, 0), Ids_({0, 1, 2, 3, -1, -1}));
  EXPECT_EQ(Ids(r.selection, 1), Ids_({0, 1, -1}));
  EXPECT_EQ(Ids(r.selection, 2), Ids_({2, 7}));

  // Rows 0, 1 and 2 have shapes (3, 2), (4, 1) and (4, 2),
  // so the cell bounds vary per row and the selection is untouched
  auto sel = SelectionBuilder::FromInit({{0, 1, 2}});
  ASSERT_OK_AND_ASSIGN(r, Negate(var, 3, 2, 4, sel));
  ASSERT_TRUE(r.shape_data.HasCellBounds());
  EXPECT_EQ(r.shape_data.GetCellBound(0), IPos({3, 2}));
  EXPECT_EQ(r.shape_data.GetCellBound(1), IPos({4, 1}));
  EXPECT_EQ(r.shape_data.GetCellBound(2), IPos({4, 2}));
  EXPECT_EQ(Ids(r.selection, 0), Ids_{});
  EXPECT_EQ(Ids(r.selection, 1), Ids_{});
  EXPECT_EQ(Ids(r.selection, 2), Ids_({0, 1, 2}));

  // Truncating and padding in different dimensions
  ASSERT_OK_AND_ASSIGN(r, Negate(var, 3, 1, 4, sel));
  ASSERT_TRUE(r.shape_data.HasCellBounds());
  EXPECT_EQ(r.shape_data.GetCellBound(0), IPos({3, 1}));
  EXPECT_EQ(r.shape_data.GetCellBound(1), IPos({4, 1}));
  EXPECT_EQ(r.shape_data.GetCellBound(2), IPos({4, 1}));

  // Rows 0 and 5 share shape (3, 2), so selection indices
  // past the end of both cells are negated, regardless of order
  ASSERT_OK_AND_ASSIGN(
      r, Negate(var, 2, 2, 3, SelectionBuilder::FromInit({{0, 5}, {1, 0}, {3, 0, 1}})));
  EXPECT_FALSE(r.shape_data.HasCellBounds());
  EXPECT_EQ(Ids(r.selection, 0), Ids_({-1, 0, 1}));
  EXPECT_EQ(Ids(r.selection, 1), Ids_({1, 0}));
  EXPECT_EQ(Ids(r.selection, 2), Ids_({0, 5}));

  // Rows 3 and 4 have shapes (2, 2) and (2, 1), so the selection
  // lies entirely past the end of both cells in the first dimension
  ASSERT_OK_AND_ASSIGN(
      r, Negate(var, 2, 1, 2, SelectionBuilder::FromInit({{3, 4}, {0}, {2, 3}})));
  EXPECT_FALSE(r.shape_data.HasCellBounds());
  EXPECT_EQ(Ids(r.selection, 0), Ids_({-1, -1}));
  EXPECT_EQ(Ids(r.selection, 1), Ids_({0}));

  // Rows 0 and 2 have shapes (3, 2) and (4, 2), so selection
  // indices past the end of a cell vary per row
  auto short_sel = SelectionBuilder::FromInit({{0, 2}, {0, 1}, {0, 1, 2, 3}});
  ASSERT_OK_AND_ASSIGN(r, Negate(var, 2, 2, 4, short_sel));
  ASSERT_TRUE(r.shape_data.HasCellBounds());
  EXPECT_EQ(r.shape_data.GetCellBound(0), IPos({3, 2}));
  EXPECT_EQ(r.shape_data.GetCellBound(1), IPos({4, 2}));
  EXPECT_EQ(Ids(r.selection, 0), Ids_({0, 1, 2, 3}));
  EXPECT_EQ(Ids(r.selection, 1), Ids_({0, 1}));
  EXPECT_EQ(Ids(r.selection, 2), Ids_({0, 2}));

  // Selections past the end of variably shaped cells
  // still raise without a result
  ASSERT_RAISES(IndexError, ResultShapeData::MakeRead(var, short_sel));
}

TEST_F(ResultShapeTest, ReadVariableSelection) {
  auto var = GetArrayColumn<Complex>(table_proxy_.table(), "VAR_DATA");
  auto sel = SelectionBuilder::FromInit({{0, 1}});
  ASSERT_OK_AND_ASSIGN(auto shape_data, ResultShapeData::MakeRead(var, sel));

  EXPECT_EQ(shape_data.GetName(), "VAR_DATA");
  EXPECT_FALSE(shape_data.IsFixed());
  EXPECT_EQ(shape_data.nDim(), 3);
  EXPECT_EQ(shape_data.nRows(), 2);
  EXPECT_EQ(shape_data.GetRowShape(0), IPos({3, 2}));
  EXPECT_EQ(shape_data.GetRowShape(1), IPos({4, 1}));
  EXPECT_EQ(shape_data.nElements(), 10);
  EXPECT_EQ(shape_data.GetDataType(), DataType::TpComplex);
  ASSERT_OK_AND_ASSIGN(auto offsets, shape_data.GetOffsets());

  sel = SelectionBuilder::FromInit({{0, 9}, {0}});
  ASSERT_OK_AND_ASSIGN(shape_data, ResultShapeData::MakeRead(var, sel));
  EXPECT_EQ(shape_data.GetName(), "VAR_DATA");
  EXPECT_FALSE(shape_data.IsFixed());
  EXPECT_EQ(shape_data.nDim(), 3);
  EXPECT_EQ(shape_data.nRows(), 2);
  EXPECT_EQ(shape_data.nElements(), 5);
  EXPECT_EQ(shape_data.GetRowShape(0), IPos({3, 1}));
  EXPECT_EQ(shape_data.GetRowShape(1), IPos({2, 1}));
  EXPECT_EQ(shape_data.GetDataType(), DataType::TpComplex);
  ASSERT_OK_AND_ASSIGN(offsets, shape_data.GetOffsets());

  // Fixed shape selection over variably shaped data
  sel = SelectionBuilder::FromInit({{0, 9}, {0}, {0, 1}});
  ASSERT_OK_AND_ASSIGN(shape_data, ResultShapeData::MakeRead(var, sel));
  EXPECT_EQ(shape_data.GetName(), "VAR_DATA");
  EXPECT_TRUE(shape_data.IsFixed());
  EXPECT_EQ(shape_data.nDim(), 3);
  EXPECT_EQ(shape_data.nRows(), 2);
  EXPECT_EQ(shape_data.nElements(), 4);
  EXPECT_EQ(shape_data.GetShape(), IPos({2, 1, 2}));
  EXPECT_EQ(shape_data.GetDataType(), DataType::TpComplex);
  ASSERT_OK_AND_ASSIGN(offsets, shape_data.GetOffsets());
}

TEST_F(ResultShapeTest, WriteFixed) {
  auto fixed = GetArrayColumn<Complex>(table_proxy_.table(), "MODEL_DATA");
  auto dtype = arrow::fixed_size_list(
      arrow::fixed_size_list(arrow::fixed_size_list(arrow::float32(), 2), 2), 2);
  ASSERT_OK_AND_ASSIGN(auto data, ArrayFromJSONString(dtype,
                                                      R"([[[[0, 0], [1, 1]],
                                          [[2, 2], [3, 3]]],
                                         [[[4, 4], [5, 5]],
                                          [[6, 6], [7, 7]]]])"));

  ASSERT_OK_AND_ASSIGN(auto shape_data, ResultShapeData::MakeWrite(fixed, data));
  EXPECT_EQ(shape_data.GetName(), "MODEL_DATA");
  EXPECT_TRUE(shape_data.IsFixed());
  EXPECT_EQ(shape_data.nDim(), 3);
  EXPECT_EQ(shape_data.nRows(), 2);
  EXPECT_EQ(shape_data.nElements(), 8);
  EXPECT_EQ(shape_data.GetShape(), IPos({2, 2, 2}));
  EXPECT_EQ(shape_data.GetDataType(), DataType::TpComplex);
  ASSERT_OK_AND_ASSIGN(auto offsets, shape_data.GetOffsets());
}

TEST_F(ResultShapeTest, WriteVariable) {
  auto var = GetArrayColumn<Complex>(table_proxy_.table(), "VAR_DATA");
  auto dtype = arrow::list(arrow::list(arrow::list(arrow::float32())));

  // Supplied as a list, but actually fixed
  ASSERT_OK_AND_ASSIGN(auto data, ArrayFromJSONString(dtype,
                                                      R"([[[[0, 0], [1, 1], [2, 2]],
                                          [[3, 3], [4, 4], [5, 5]]],
                                         [[[6, 6], [7, 7], [8, 8]],
                                          [[9, 9], [10, 10], [11, 11]]]])"));

  // Rows 0 and 1 have shapes (3, 2) and (4, 1), too small for the data
  ASSERT_RAISES(IndexError, ResultShapeData::MakeWrite(var, data));

  // Rows 2 and 7 have shape (4, 2)
  auto rows = SelectionBuilder::FromInit({{2, 7}});
  ASSERT_OK_AND_ASSIGN(auto shape_data, ResultShapeData::MakeWrite(var, data, rows));
  EXPECT_EQ(shape_data.GetName(), "VAR_DATA");
  EXPECT_TRUE(shape_data.IsFixed());
  EXPECT_EQ(shape_data.nDim(), 3);
  EXPECT_EQ(shape_data.nRows(), 2);
  EXPECT_EQ(shape_data.nElements(), 12);
  EXPECT_EQ(shape_data.GetShape(), IPos({3, 2, 2}));
  EXPECT_EQ(shape_data.GetDataType(), DataType::TpComplex);
  ASSERT_OK_AND_ASSIGN(auto offsets, shape_data.GetOffsets());

  // Variably shaped
  ASSERT_OK_AND_ASSIGN(data, ArrayFromJSONString(dtype,
                                                 R"([[[[0, 0], [1, 1]],
                                          [[3, 3], [4, 4]]],
                                         [[[6, 6], [7, 7], [8, 8]],
                                          [[9, 9], [10, 10], [11, 11]]]])"));

  ASSERT_OK_AND_ASSIGN(shape_data, ResultShapeData::MakeWrite(var, data, rows));
  EXPECT_EQ(shape_data.GetName(), "VAR_DATA");
  EXPECT_FALSE(shape_data.IsFixed());
  EXPECT_EQ(shape_data.nDim(), 3);
  EXPECT_EQ(shape_data.nRows(), 2);
  EXPECT_EQ(shape_data.nElements(), 10);
  EXPECT_EQ(shape_data.GetDataType(), DataType::TpComplex);
  EXPECT_EQ(shape_data.GetRowShape(0), IPos({2, 2}));
  EXPECT_EQ(shape_data.GetRowShape(1), IPos({3, 2}));
  ASSERT_OK_AND_ASSIGN(offsets, shape_data.GetOffsets());
}

TEST_F(ResultShapeTest, WriteSelectionSize) {
  auto fixed = GetArrayColumn<Complex>(table_proxy_.table(), "MODEL_DATA");
  auto var = GetArrayColumn<Complex>(table_proxy_.table(), "VAR_DATA");

  // (2 rows, 2 chans, 1 corr) complex data
  ASSERT_OK_AND_ASSIGN(auto data, MakeComplexResult(2, 2, 1));
  auto rows = std::initializer_list<int>{2, 7};

  for (const auto& column : {fixed, var}) {
    // Selection sizes match the data, including -1 entries
    ASSERT_OK(ResultShapeData::MakeWrite(
        column, data, SelectionBuilder::FromInit({rows, {0, 1}, {0}})));
    ASSERT_OK(ResultShapeData::MakeWrite(column, data,
                                         SelectionBuilder::FromInit({rows, {-1, 1}})));
    // Selection sizes differ from the data
    ASSERT_RAISES(IndexError, ResultShapeData::MakeWrite(
                                  column, data, SelectionBuilder::FromInit({rows, {0}})));
    ASSERT_RAISES(IndexError,
                  ResultShapeData::MakeWrite(
                      column, data, SelectionBuilder::FromInit({rows, {0, 1, 2}})));
    ASSERT_RAISES(IndexError,
                  ResultShapeData::MakeWrite(
                      column, data, SelectionBuilder::FromInit({rows, {0, 1}, {0, 1}})));
  }

  // Variably shaped data is checked row by row
  auto dtype = arrow::list(arrow::list(arrow::list(arrow::float32())));
  ASSERT_OK_AND_ASSIGN(data, ArrayFromJSONString(dtype, R"([[[[0, 0]], [[1, 1]]],
                                                           [[[2, 2]]]])"));
  ASSERT_RAISES(IndexError, ResultShapeData::MakeWrite(
                                var, data, SelectionBuilder::FromInit({rows, {0, 1}})));
}

}  // namespace
