#include <sys/types.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <string>

#include <arrow/api.h>
#include <arrow/array/array_base.h>
#include <arrow/array/array_nested.h>
#include <arrow/compute/cast.h>
#include <arrow/result.h>
#include <arrow/testing/builder.h>
#include <arrow/testing/gtest_util.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/logging.h>

#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/BasicSL/Complexfwd.h>
#include <casacore/casa/Utilities/DataType.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/tables/Tables/ColumnDesc.h>
#include <casacore/tables/Tables/SetupNewTab.h>
#include <casacore/tables/Tables/TableProxy.h>

#include <tests/test_utils.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "arcae/new_table_proxy.h"
#include "arcae/selection.h"

using ::arcae::GetArrayColumn;
using ::arcae::GetScalarColumn;
using ::arcae::NewTableProxy;
using ::arcae::detail::Index;
using ::arcae::detail::IndexSpan;
using ::arcae::detail::IndexType;
using ::arcae::detail::SelectionBuilder;

using ::casacore::Array;
using ::casacore::ArrayColumnDesc;
using ::casacore::ColumnDesc;
using ::casacore::Complex;
using MS = casacore::MeasurementSet;
using ::casacore::IPosition;
using ::casacore::Record;
using ::casacore::SetupNewTable;
using ::casacore::String;
using ::casacore::Table;
using ::casacore::TableDesc;
using ::casacore::TableLock;
using ::casacore::TableProxy;

using namespace std::string_literals;

namespace {

static constexpr std::size_t knInstances = 4;

static constexpr std::size_t knrow = 10;
static constexpr std::size_t knchan = 16;
static constexpr std::size_t kncorr = 4;

static constexpr std::size_t kNDim = 3;
static constexpr std::array<std::size_t, kNDim> kDimensions = {kncorr, knchan, knrow};
static constexpr std::size_t kElements = knrow * knchan * kncorr;

// Default value in result arrays
static constexpr float kDefaultRealValue = -10.0f;

std::mt19937 kRng = std::mt19937{};

// Structure parametrizing a test case
struct Parametrization {
  // For each dimension, is the selection empty
  std::array<bool, kNDim> empty_selection = {false, false, false};
  // Indicates whether each dimension should be randomised
  std::array<bool, kNDim> randomise = {false, false, false};
  // For each dimension, number of elements to be removed
  std::array<std::size_t, kNDim> remove = {0, 0, 0};
};

class ZeroRowTableProxyTest : public ::testing::Test {
 protected:
  std::string table_name_;

  void SetUp() override {
    auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    table_name_ = test_info->name() + "-"s + arcae::hexuuid(4) + ".table"s;
    auto table_desc = TableDesc(MS::requiredTableDesc());
    auto data_shape = IPosition({kncorr, knchan});
    auto data_column_desc =
        ArrayColumnDesc<Complex>("DATA", data_shape, ColumnDesc::FixedShape);
    table_desc.addColumn(data_column_desc);
    auto setup_new_table = SetupNewTable(table_name_, table_desc, Table::New);
    auto ms = MS(setup_new_table, 0);
    ms.createDefaultSubtables(Table::New);
  }

  arrow::Result<std::shared_ptr<NewTableProxy>> OpenTable() {
    return NewTableProxy::Make([name = table_name_]() {
      auto lock = TableLock(TableLock::LockOption::AutoLocking);
      auto lockoptions = Record();
      lockoptions.define("option", "user");
      lockoptions.define("internal", lock.interval());
      lockoptions.define("maxwait", casacore::Int(lock.maxWait()));
      return std::make_shared<TableProxy>(name, lockoptions, Table::Old);
    });
  }

  void TearDown() override { std::filesystem::remove_all(table_name_); }
};

class FixedTableProxyTest : public ::testing::TestWithParam<Parametrization> {
 protected:
  std::string table_name_;
  Array<Complex> complex_vals_;
  Array<String> str_vals_;

  void SetUp() override {
    auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    table_name_ = test_info->name() + "-"s + arcae::hexuuid(4) + ".table"s;
    // Replace / in the Parametrized Test name with '~'
    for (int i = 0; i < table_name_.size(); ++i) {
      if (table_name_[i] == '/') table_name_[i] = '~';
    }
    auto table_desc = TableDesc(MS::requiredTableDesc());
    auto data_shape = IPosition({kncorr, knchan});

    auto data_column_desc =
        ArrayColumnDesc<Complex>("MODEL_DATA", data_shape, ColumnDesc::FixedShape);
    table_desc.addColumn(data_column_desc);
    auto var_data_column_desc = ArrayColumnDesc<Complex>("VAR_DATA", data_shape.size());
    table_desc.addColumn(var_data_column_desc);
    auto str_data_column_desc =
        ArrayColumnDesc<String>("STRING_DATA", data_shape, ColumnDesc::FixedShape);
    table_desc.addColumn(str_data_column_desc);
    auto var_str_data_column_desc =
        ArrayColumnDesc<String>("VAR_STRING_DATA", data_shape.size());
    table_desc.addColumn(var_str_data_column_desc);

    auto setup_new_table = SetupNewTable(table_name_, table_desc, Table::New);
    auto ms = MS(setup_new_table, knrow);
    auto data = GetArrayColumn<Complex>(ms, "MODEL_DATA");
    auto str_data = GetArrayColumn<String>(ms, "STRING_DATA");
    auto var_str_data = GetArrayColumn<String>(ms, "VAR_STRING_DATA");
    auto var_data = GetArrayColumn<Complex>(ms, "VAR_DATA");
    auto time = GetScalarColumn<casacore::Double>(ms, "TIME");

    // Initialise the time column
    for (std::size_t i = 0; i < knrow; ++i) time.put(i, i);

    // Initialise data and string data
    complex_vals_ = Array<Complex>(IPosition({kncorr, knchan, knrow}));
    str_vals_ = Array<String>(IPosition({kncorr, knchan, knrow}));

    for (std::size_t i = 0; i < complex_vals_.size(); ++i) {
      auto v = Complex::value_type(i);
      complex_vals_.data()[i] = {v, v};
      str_vals_.data()[i] = "FOO-" + std::to_string(i);
    }

    // Write test data to the MS
    // data.putColumn(complex_vals_);
    // var_data.putColumn(complex_vals_);
    // str_data.putColumn(str_vals_);
    // var_str_data.putColumn(str_vals_);
  }

  // Create complex test data for the given indices
  arrow::Result<std::shared_ptr<arrow::Array>> MakeComplexArray(const Index& rows,
                                                                const Index& chans,
                                                                const Index& corrs) {
    std::shared_ptr<arrow::Array> array;
    std::vector<float> values(rows.size() * chans.size() * corrs.size() * 2);
    for (int r = 0; r < rows.size(); ++r) {
      auto row = rows[r];
      for (int ch = 0; ch < chans.size(); ++ch) {
        auto chan = chans[ch];
        for (int co = 0; co < corrs.size(); ++co) {
          auto corr = corrs[co];
          auto buf_i = r * chans.size() * corrs.size() + ch * corrs.size() + co;
          auto real = row * knchan * kncorr + chan * kncorr + corr;
          values[2 * buf_i + 0] = real;
          values[2 * buf_i + 1] = real;
        }
      }
    }

    arrow::ArrayFromVector<arrow::FloatType>(arrow::float32(), values, &array);
    ARROW_ASSIGN_OR_RAISE(array, arrow::FixedSizeListArray::FromArrays(array, 2));
    ARROW_ASSIGN_OR_RAISE(array,
                          arrow::FixedSizeListArray::FromArrays(array, corrs.size()));
    ARROW_ASSIGN_OR_RAISE(array,
                          arrow::FixedSizeListArray::FromArrays(array, chans.size()));
    return array;
  }

  // Create string test data for the given indices
  arrow::Result<std::shared_ptr<arrow::Array>> MakeStringArray(const Index& rows,
                                                               const Index& chans,
                                                               const Index& corrs) {
    std::shared_ptr<arrow::Array> array;
    std::vector<std::string> values(rows.size() * chans.size() * corrs.size());
    for (int r = 0; r < rows.size(); ++r) {
      auto row = rows[r];
      for (int ch = 0; ch < chans.size(); ++ch) {
        auto chan = chans[ch];
        for (int co = 0; co < corrs.size(); ++co) {
          auto corr = corrs[co];
          auto buf_i = r * chans.size() * corrs.size() + ch * corrs.size() + co;
          auto real = row * knchan * kncorr + chan * kncorr + corr;
          values[buf_i] = "FOO-" + std::to_string(real);
        }
      }
    }

    arrow::ArrayFromVector<arrow::StringType>(arrow::utf8(), values, &array);
    ARROW_ASSIGN_OR_RAISE(array,
                          arrow::FixedSizeListArray::FromArrays(array, corrs.size()));
    ARROW_ASSIGN_OR_RAISE(array,
                          arrow::FixedSizeListArray::FromArrays(array, chans.size()));

    return array;
  }

  arrow::Result<std::shared_ptr<NewTableProxy>> OpenTable(std::size_t instances = 1,
                                                          bool readonly = true) {
    return NewTableProxy::Make(
        [&, name = table_name_]() {
          auto lock = TableLock(TableLock::LockOption::AutoLocking);
          auto lockoptions = Record();
          lockoptions.define("option", "user");
          lockoptions.define("internal", lock.interval());
          lockoptions.define("maxwait", casacore::Int(lock.maxWait()));
          auto tp = std::make_shared<TableProxy>(name, lockoptions, Table::Old);
          if (!readonly) tp->reopenRW();
          return tp;
        },
        instances);
  }

  void TearDown() override { std::filesystem::remove_all(table_name_); }
};

INSTANTIATE_TEST_SUITE_P(
    ParametrizationSuite, FixedTableProxyTest,
    ::testing::Values(
        Parametrization{{false, false, false}, {false, false, false}},
        Parametrization{{false, false, false}, {true, true, true}},
        Parametrization{{false, false, false}, {true, true, true}, {1, 3, 5}},
        Parametrization{{false, false, false}, {false, false, false}, {1, 3, 5}},
        Parametrization{{true, true, true}, {true, true, true}, {1, 3, 5}},
        Parametrization{{true, true, true}, {false, false, false}, {1, 3, 5}}));

TEST_P(FixedTableProxyTest, Fixed) {
  ASSERT_OK_AND_ASSIGN(auto wntp, OpenTable(1, false));
  const auto& params = GetParam();
  auto builder = SelectionBuilder().Order('F');

  EXPECT_EQ(kNDim, 3);
  std::vector<Index> indices(kNDim);

  for (std::size_t d = 0; d < kNDim; ++d) {
    // Generate a range of indices
    indices[d] = Index(kDimensions[d], 0);
    std::iota(std::begin(indices[d]), std::end(indices[d]), 0);

    if (params.empty_selection[d]) {
      // indices[d] corresponds to an empty selection
      builder.AddEmpty();
    } else {
      // Maybe shuffle the indices
      if (params.randomise[d]) {
        std::shuffle(std::begin(indices[d]), std::end(indices[d]), kRng);
      }
      // Maybe remove a number of values from the selection
      if (params.remove[d] > 0) {
        EXPECT_TRUE(params.remove[d] < kDimensions[d])
            << "Cannot remove more than " << kDimensions[d] << " indices "
            << "from dimension " << d;
        double ratio = double(kDimensions[d] - 1) / double(params.remove[d]);
        for (std::ptrdiff_t r = params.remove[d] - 1; r >= 0; --r) {
          indices[d].erase(std::begin(indices[d]) + int(round(r * ratio)));
        }
        EXPECT_EQ(indices[d].size(), kDimensions[d] - params.remove[d]);
      }
      builder.Add(indices[d]);
    }
  }

  auto selection = builder.Build();
  EXPECT_EQ(selection.Size(), kNDim);
  EXPECT_EQ(indices.size(), kNDim);

  // Sanity check that empty selection dimensions
  // align with the full index
  for (std::size_t d = 0; d < kNDim; ++d) {
    if (!params.empty_selection[d]) continue;
    EXPECT_FALSE(selection.FSpan(d).ok());
    Index expected(kDimensions[d], 0);
    std::iota(std::begin(expected), std::end(expected), 0);
    EXPECT_EQ(indices[d], expected);
  }

  auto& corrs = indices[0];
  auto& chans = indices[1];
  auto& rows = indices[2];

  ASSERT_OK_AND_ASSIGN(auto write_data, MakeComplexArray(rows, chans, corrs));
  ASSERT_OK_AND_ASSIGN(auto write_str, MakeStringArray(rows, chans, corrs));
  ASSERT_OK(wntp->PutColumn("MODEL_DATA", write_data, selection));
  ASSERT_OK(wntp->PutColumn("VAR_DATA", write_data, selection));
  ASSERT_OK(wntp->PutColumn("STRING_DATA", write_str, selection));
  ASSERT_OK(wntp->PutColumn("VAR_STRING_DATA", write_str, selection));
  ASSERT_OK(wntp->Close());

  ASSERT_OK_AND_ASSIGN(auto ntp, OpenTable(knInstances));
  ASSERT_OK_AND_ASSIGN(auto data, ntp->GetColumn("MODEL_DATA", selection));
  ASSERT_OK_AND_ASSIGN(auto var_data, ntp->GetColumn("VAR_DATA", selection));
  ASSERT_OK_AND_ASSIGN(auto str_data, ntp->GetColumn("STRING_DATA", selection));
  ASSERT_OK_AND_ASSIGN(auto var_str_data, ntp->GetColumn("VAR_STRING_DATA", selection));

  // Get the underlying buffer containing the complex values
  auto fixed_data_buffer = data->data()
                               ->child_data[0]  // chan
                               ->child_data[0]  // corr
                               ->child_data[0]  // complex pair
                               ->buffers[1]     // value buffer
                               ->span_as<Complex>();

  auto var_data_buffer = var_data->data()
                             ->child_data[0]  // chan
                             ->child_data[0]  // corr
                             ->child_data[0]  // complex pair
                             ->buffers[1]
                             ->span_as<Complex>();

  // Get the underyling string array
  str_data = arrow::MakeArray(str_data->data()
                                  ->child_data[0]    // chan
                                  ->child_data[0]);  // corr
  var_str_data = arrow::MakeArray(var_str_data->data()
                                      ->child_data[0]    // chan
                                      ->child_data[0]);  // corr

  auto nIndexElements = rows.size() * chans.size() * corrs.size();
  EXPECT_EQ(fixed_data_buffer.size(), nIndexElements);
  EXPECT_EQ(var_data_buffer.size(), nIndexElements);
  EXPECT_EQ(str_data->length(), nIndexElements);
  EXPECT_EQ(var_str_data->length(), nIndexElements);

  // Check that the expected value at the selected disk indices
  // matches the buffer values
  for (std::size_t r = 0; r < rows.size(); ++r) {
    auto row = rows[r];
    EXPECT_TRUE(row < knrow);  // sanity
    for (std::size_t ch = 0; ch < chans.size(); ++ch) {
      auto chan = chans[ch];
      EXPECT_TRUE(chan < knchan);  // sanity
      for (std::size_t co = 0; co < corrs.size(); ++co) {
        auto corr = corrs[co];
        EXPECT_TRUE(corr < kncorr);  // sanity
        auto buf_i = r * chans.size() * corrs.size() + ch * corrs.size() + co;
        auto real = row * knchan * kncorr + chan * kncorr + corr;
        EXPECT_EQ(complex_vals_(IPosition({corr, chan, row})), Complex(real, real));
        EXPECT_EQ(fixed_data_buffer[buf_i], Complex(real, real))
            << '[' << co << ',' << ch << ',' << r << ']' << ' ' << '[' << corr << ','
            << chan << ',' << row << ']' << ' ' << buf_i << ' ' << real;
        EXPECT_EQ(var_data_buffer[buf_i], Complex(real, real));
        ASSERT_OK_AND_ASSIGN(auto str_val, str_data->GetScalar(buf_i));
        EXPECT_EQ(str_val->ToString(), "FOO-" + std::to_string(real));
        ASSERT_OK_AND_ASSIGN(str_val, var_str_data->GetScalar(buf_i));
        EXPECT_EQ(str_val->ToString(), "FOO-" + std::to_string(real));
      }
    }
  }
}

TEST_F(FixedTableProxyTest, Table) {
  ASSERT_OK_AND_ASSIGN(auto ntp, OpenTable());
  ASSERT_OK_AND_ASSIGN(auto table, ntp->ToArrow({}, {"MODEL_DATA"}));
  EXPECT_THAT(table->ColumnNames(), ::testing::ElementsAre("MODEL_DATA"));

  ASSERT_OK_AND_ASSIGN(table, ntp->ToArrow({}, {}));
}

TEST_F(FixedTableProxyTest, AddColumns) {
  ASSERT_OK_AND_ASSIGN(auto ntp, OpenTable());

  // TODO: Embed knrow, knchan and kncorr in the shape and tileshapes
  auto column_desc = R"""(
  {
    "ACK": {
      "dataManagerGroup": "ACKBAR_GROUP",
      "dataManagerType": "TiledColumnStMan",
      "ndim": 2,
      "shape": [16, 4],
      "valueType": "BOOLEAN"
    },
    "BAR": {
      "dataManagerGroup": "ACKBAR_GROUP",
      "dataManagerType": "TiledColumnStMan",
      "ndim": 2,
      "shape": [16, 4],
      "valueType": "COMPLEX"
    }
  }
  )""";

  std::string dminfo = R"""(
  {
    "*1": {
      "NAME": "ACKBAR_GROUP",
      "TYPE": "TiledColumnStMan",
      "SPEC": {"DEFAULTTILESHAPE": [16, 4, 10]},
      "COLUMNS": ["ACK", "BAR"]
    }
  }
  )""";

  ASSERT_OK(ntp->AddColumns(column_desc, dminfo));
  ASSERT_OK_AND_ASSIGN(auto columns, ntp->Columns());
  EXPECT_THAT(columns, ::testing::Contains("ACK"));
  EXPECT_THAT(columns, ::testing::Contains("BAR"));
  ASSERT_OK_AND_ASSIGN(dminfo, ntp->GetDataManagerInfo());
  EXPECT_TRUE(dminfo.find(R"("NAME": "ACKBAR_GROUP")") != std::string::npos);
  EXPECT_TRUE(dminfo.find(R"("COLUMNS": ["ACK")") != std::string::npos) << dminfo;
  EXPECT_TRUE(dminfo.find(R"("BAR"])") != std::string::npos) << dminfo;
}

TEST_F(ZeroRowTableProxyTest, ZeroRowCase) {
  ASSERT_OK_AND_ASSIGN(auto ntp, OpenTable());
  ASSERT_OK_AND_ASSIGN(auto data, ntp->GetColumn("DATA"));
  ASSERT_OK_AND_ASSIGN(auto time, ntp->GetColumn("TIME"));
  EXPECT_EQ(time->length(), 0);
  EXPECT_EQ(data->length(), 0);
}

TEST_F(FixedTableProxyTest, NegativeSelection) {
  ASSERT_OK_AND_ASSIGN(auto ntp, OpenTable());

  // Create a selection with negative indices
  Index rows(knrow);
  Index chans(knchan);
  Index corrs(kncorr);

  std::iota(std::begin(rows), std::end(rows), 0);
  std::iota(std::begin(chans), std::end(chans), 0);
  std::iota(std::begin(corrs), std::end(corrs), 0);

  for (auto r : {0, 3, 6}) rows[r] = -1;
  for (auto ch : {1, 3, 11}) chans[ch] = -1;
  for (auto co : {1, 3}) corrs[co] = -1;

  auto selection = SelectionBuilder().Order('F').Add(corrs).Add(chans).Add(rows).Build();

  ASSERT_OK_AND_ASSIGN(auto write_data, MakeComplexArray(rows, chans, corrs));
  ASSERT_OK_AND_ASSIGN(auto write_str, MakeStringArray(rows, chans, corrs));
  ASSERT_OK(ntp->PutColumn("MODEL_DATA", write_data, selection));
  ASSERT_OK(ntp->PutColumn("VAR_DATA", write_data, selection));
  ASSERT_OK(ntp->PutColumn("STRING_DATA", write_str, selection));
  ASSERT_OK(ntp->PutColumn("VAR_STRING_DATA", write_str, selection));

  // Reading strings into a result array is not supported
  std::shared_ptr<arrow::Array> str_result;
  std::vector<std::string> str_values(rows.size() * chans.size() * corrs.size(),
                                      "NOTHING");
  arrow::ArrayFromVector<arrow::StringType>(arrow::utf8(), str_values, &str_result);
  ASSERT_OK_AND_ASSIGN(str_result,
                       arrow::FixedSizeListArray::FromArrays(str_result, corrs.size()));
  ASSERT_OK_AND_ASSIGN(str_result,
                       arrow::FixedSizeListArray::FromArrays(str_result, chans.size()));
  EXPECT_EQ(str_result->length(), rows.size());
  ASSERT_NOT_OK(ntp->GetColumn("STRING_DATA", selection, str_result));
  ASSERT_NOT_OK(ntp->GetColumn("VAR_STRING_DATA", selection, str_result));

  // Create a result array filled with default values
  std::shared_ptr<arrow::Array> result;
  std::vector<float> values(rows.size() * chans.size() * corrs.size() * 2,
                            kDefaultRealValue);
  arrow::ArrayFromVector<arrow::FloatType>(arrow::float32(), values, &result);
  ASSERT_OK_AND_ASSIGN(result, arrow::FixedSizeListArray::FromArrays(result, 2));
  ASSERT_OK_AND_ASSIGN(result,
                       arrow::FixedSizeListArray::FromArrays(result, corrs.size()));
  ASSERT_OK_AND_ASSIGN(result,
                       arrow::FixedSizeListArray::FromArrays(result, chans.size()));
  EXPECT_EQ(result->length(), rows.size());
  ASSERT_OK_AND_ASSIGN(auto var_result,
                       result->CopyTo(arrow::default_cpu_memory_manager()));
  ASSERT_OK(ntp->GetColumn("MODEL_DATA", selection, result));
  ASSERT_OK(ntp->GetColumn("VAR_DATA", selection, var_result));

  // Get the underlying buffer containing the complex values
  auto buffer = result->data()
                    ->child_data[0]  // chan
                    ->child_data[0]  // corr
                    ->child_data[0]  // complex pair
                    ->buffers[1]     // value buffer
                    ->span_as<Complex>();

  auto var_buffer = var_result->data()
                        ->child_data[0]  // chan
                        ->child_data[0]  // corr
                        ->child_data[0]  // complex pair
                        ->buffers[1]     // value buffer
                        ->span_as<Complex>();

  // Check that the expected value at the selected disk indices
  // matches the buffer values
  for (std::size_t r = 0; r < rows.size(); ++r) {
    auto row = rows[r];
    EXPECT_TRUE(row < IndexType(knrow));  // sanity
    for (std::size_t ch = 0; ch < chans.size(); ++ch) {
      auto chan = chans[ch];
      EXPECT_TRUE(chan < IndexType(knchan));  // sanity
      for (std::size_t co = 0; co < corrs.size(); ++co) {
        auto corr = corrs[co];
        EXPECT_TRUE(corr < IndexType(kncorr));  // sanity
        auto buf_i = r * chans.size() * corrs.size() + ch * corrs.size() + co;
        if (row < 0 || chan < 0 || corr < 0) {
          // Negative index, the buffer should not be populated
          EXPECT_EQ(buffer[buf_i], Complex(kDefaultRealValue, kDefaultRealValue));
        } else {
          // We expect this value to be populated
          auto real = row * knchan * kncorr + chan * kncorr + corr;
          EXPECT_EQ(complex_vals_(IPosition({corr, chan, row})), Complex(real, real));
          EXPECT_EQ(buffer[buf_i], Complex(real, real));
          EXPECT_EQ(var_buffer[buf_i], Complex(real, real));
        }
      }
    }
  }
}

// Test that adding rows to the table works
TEST_F(FixedTableProxyTest, AddRows) {
  ASSERT_OK_AND_ASSIGN(auto ntp, OpenTable());
  ASSERT_OK_AND_ASSIGN(auto nrow, ntp->nRows());
  EXPECT_EQ(nrow, knrow);
  ASSERT_OK(ntp->AddRows(100));
  ASSERT_OK_AND_ASSIGN(nrow, ntp->nRows());
  EXPECT_EQ(nrow, knrow + 100);
}

TEST_F(FixedTableProxyTest, CastData) {
  ASSERT_OK_AND_ASSIGN(auto ntp, OpenTable());
  std::shared_ptr<arrow::Array> double_array;
  std::shared_ptr<arrow::Array> flag_array;
  std::vector<double> double_values(knrow * knchan * kncorr * 2);
  std::vector<bool> bool_values(knrow * knchan * kncorr);
  for (std::size_t i = 0; i < knrow * knchan * kncorr; ++i) {
    double_values[2 * i + 0] = i;
    double_values[2 * i + 1] = i;
    bool_values[i] = i % 2;
  }

  arrow::ArrayFromVector<arrow::DoubleType>(arrow::float64(), double_values,
                                            &double_array);
  ASSERT_OK_AND_ASSIGN(double_array,
                       arrow::FixedSizeListArray::FromArrays(double_array, 2));
  ASSERT_OK_AND_ASSIGN(double_array,
                       arrow::FixedSizeListArray::FromArrays(double_array, kncorr));
  ASSERT_OK_AND_ASSIGN(double_array,
                       arrow::FixedSizeListArray::FromArrays(double_array, knchan));
  EXPECT_EQ(double_array->length(), knrow);
  ASSERT_OK(double_array->ValidateFull());
  ASSERT_OK(ntp->PutColumn("MODEL_DATA", double_array));

  ASSERT_OK_AND_ASSIGN(auto result, ntp->GetColumn("MODEL_DATA"));
  ASSERT_OK_AND_ASSIGN(auto datum, arrow::compute::Cast(result, double_array->type()));
  EXPECT_TRUE(datum.make_array()->ApproxEquals(double_array));

  arrow::ArrayFromVector<arrow::UInt8Type>(arrow::uint8(), bool_values, &flag_array);
  ASSERT_OK_AND_ASSIGN(flag_array,
                       arrow::FixedSizeListArray::FromArrays(flag_array, kncorr));
  ASSERT_OK_AND_ASSIGN(flag_array,
                       arrow::FixedSizeListArray::FromArrays(flag_array, knchan));
  EXPECT_EQ(flag_array->length(), knrow);
  ASSERT_OK(ntp->PutColumn("FLAG", flag_array));

  ASSERT_OK_AND_ASSIGN(result, ntp->GetColumn("FLAG"));
  ASSERT_OK_AND_ASSIGN(datum, arrow::compute::Cast(result, flag_array->type()));
  EXPECT_TRUE(datum.make_array()->Equals(flag_array));
}

// Test basic column and row getting functionality
TEST_F(FixedTableProxyTest, ColumnAndRowGet) {
  ASSERT_OK_AND_ASSIGN(auto ntp, OpenTable());
  ASSERT_OK_AND_ASSIGN(auto columns, ntp->Columns());
  ASSERT_OK_AND_ASSIGN(auto ncolumns, ntp->nColumns());
  ASSERT_OK_AND_ASSIGN(auto nrow, ntp->nRows());
  EXPECT_EQ(nrow, knrow);
  EXPECT_EQ(ncolumns, 25);
  EXPECT_EQ(ncolumns, columns.size());
  EXPECT_THAT(columns,
              ::testing::ElementsAre(
                  "UVW", "FLAG", "FLAG_CATEGORY", "WEIGHT", "SIGMA", "ANTENNA1",
                  "ANTENNA2", "ARRAY_ID", "DATA_DESC_ID", "EXPOSURE", "FEED1", "FEED2",
                  "FIELD_ID", "FLAG_ROW", "INTERVAL", "OBSERVATION_ID", "PROCESSOR_ID",
                  "SCAN_NUMBER", "STATE_ID", "TIME", "TIME_CENTROID", "MODEL_DATA",
                  "VAR_DATA", "STRING_DATA", "VAR_STRING_DATA"));
}

// Test name getting
TEST_F(FixedTableProxyTest, Name) {
  ASSERT_OK_AND_ASSIGN(auto ntp, OpenTable());
  ASSERT_OK_AND_ASSIGN(auto name, ntp->Name());
  EXPECT_TRUE(table_name_.size() <= name.size());
  std::size_t d = name.size() - table_name_.size();
  for (std::size_t i = 0; i < table_name_.size(); ++i) {
    EXPECT_EQ(table_name_[i], name[d + i]);
  }
}

class VariableProxyTest : public ::testing::TestWithParam<Parametrization> {
 protected:
  std::string table_name_;
  std::vector<casacore::IPosition> row_shapes_ = {{4, 16}, {3, 15}, {4, 14}, {3, 17}};
  std::vector<Array<Complex>> complex_vals_;
  std::vector<Array<String>> str_vals_;

  void SetUp() override {
    auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    table_name_ = test_info->name() + "-"s + arcae::hexuuid(4) + ".table"s;
    // Replace / in the Parametrized Test name with '~'
    for (int i = 0; i < table_name_.size(); ++i) {
      if (table_name_[i] == '/') table_name_[i] = '~';
    }
    auto table_desc = TableDesc(MS::requiredTableDesc());
    auto var_data_column_desc = ArrayColumnDesc<Complex>("VAR_DATA", 2);
    table_desc.addColumn(var_data_column_desc);
    auto var_sparse_data_column_desc = ArrayColumnDesc<Complex>("VAR_SPARSE_DATA", 2);
    table_desc.addColumn(var_sparse_data_column_desc);
    auto var_str_data_column_desc = ArrayColumnDesc<String>("VAR_STRING_DATA", 2);
    table_desc.addColumn(var_str_data_column_desc);

    auto setup_new_table = SetupNewTable(table_name_, table_desc, Table::New);
    auto ms = MS(setup_new_table, nRow());
    auto var_data = GetArrayColumn<Complex>(ms, "VAR_DATA");
    auto var_str_data = GetArrayColumn<String>(ms, "VAR_STRING_DATA");
    auto time = GetScalarColumn<casacore::Double>(ms, "TIME");

    // Initialise the time column
    for (std::size_t i = 0; i < nRow(); ++i) time.put(i, i);

    // Initialise data and string data
    // Write test data to the MS
    std::size_t real = 0;
    complex_vals_.resize(nRow());
    str_vals_.resize(nRow());

    for (std::size_t i = 0; i < nRow(); ++i) {
      complex_vals_[i] = Array<Complex>(row_shapes_[i]);
      str_vals_[i] = Array<String>(row_shapes_[i]);

      for (std::size_t s = 0; s < row_shapes_[i].product(); ++s) {
        complex_vals_[i].data()[s] = Complex(real, real);
        str_vals_[i].data()[s] = "FOO-" + std::to_string(real);
        ++real;
      }

      var_data.put(i, complex_vals_[i]);
      var_str_data.put(i, str_vals_[i]);
    }
  }

  arrow::Result<std::shared_ptr<NewTableProxy>> OpenTable() {
    return NewTableProxy::Make([name = table_name_]() {
      auto lock = TableLock(TableLock::LockOption::AutoLocking);
      auto lockoptions = Record();
      lockoptions.define("option", "user");
      lockoptions.define("internal", lock.interval());
      lockoptions.define("maxwait", casacore::Int(lock.maxWait()));
      return std::make_shared<TableProxy>(name, lockoptions, Table::Old);
    });
  }

  std::size_t RealValue(std::size_t row, std::size_t chan, std::size_t corr) const {
    assert(row < row_shapes_.size());
    const auto& shape = row_shapes_[row];
    auto nchan = shape[1];
    auto ncorr = shape[0];
    assert(chan < nchan);
    assert(corr < ncorr);
    std::size_t offset = chan * ncorr + corr;
    for (std::size_t r = 0; r < row; ++r) {
      offset += row_shapes_[r].product();
    }
    return offset;
  }

  std::size_t nRow() const { return row_shapes_.size(); }

  void TearDown() override { std::filesystem::remove_all(table_name_); }
};

INSTANTIATE_TEST_SUITE_P(
    ParametrizationSuite, VariableProxyTest,
    ::testing::Values(
        Parametrization{{true, true, true}, {false, false, false}},
        Parametrization{{false, false, false}, {false, false, false}},
        Parametrization{{false, false, false}, {true, true, true}},
        Parametrization{{false, false, false}, {true, true, true}, {1, 3, 2}},
        Parametrization{{false, false, false}, {false, false, false}, {1, 3, 2}},
        Parametrization{{true, true, true}, {true, true, true}, {1, 3, 5}},
        Parametrization{{true, true, true}, {false, false, false}, {1, 3, 2}}));

TEST_P(VariableProxyTest, Variable) {
  ASSERT_OK_AND_ASSIGN(auto ntp, OpenTable());
  EXPECT_TRUE(nRow() > 0);
  auto params = GetParam();

  // Derive minimum viable dimension size.
  // i.e. the maximum viable selection size along each dimension
  // as well as the maximum of all dimension sizes
  std::size_t max_dim = nRow();
  casacore::IPosition min_shape(kNDim, std::numeric_limits<ssize_t>::max());
  min_shape[2] = nRow();  // row dimension is trivial
  for (const auto& shape : row_shapes_) {
    EXPECT_EQ(shape.size(), kNDim - 1);
    for (std::size_t d = 0; d < shape.size(); ++d) {
      min_shape[d] = std::min(min_shape[d], shape[d]);
      max_dim = std::max<std::size_t>(max_dim, shape[d]);
    }
  }

  // Setup a dummy index for empty dimensions
  Index dummy_index(max_dim, 0);
  std::iota(std::begin(dummy_index), std::end(dummy_index), 0);

  ASSERT_OK_AND_ASSIGN(auto var_data, ntp->GetColumn("VAR_DATA"));
  ASSERT_OK_AND_ASSIGN(auto str_data, ntp->GetColumn("VAR_STRING_DATA"));

  auto builder = SelectionBuilder().Order('F');
  std::vector<Index> indices(kNDim);

  for (std::size_t d = 0; d < kNDim; ++d) {
    if (params.empty_selection[d]) {
      builder.AddEmpty();
    } else {
      indices[d] = Index(min_shape[d], 0);
      std::iota(std::begin(indices[d]), std::end(indices[d]), 0);

      // Shuffle the selection along this dimension
      if (params.randomise[d]) {
        std::shuffle(std::begin(indices[d]), std::end(indices[d]), kRng);
      }

      // Maybe remove a number of values from the selection
      if (params.remove[d] > 0) {
        EXPECT_TRUE(params.remove[d] < min_shape[d])
            << "Cannot remove more than " << min_shape[d] << " indices "
            << "from dimension " << d;
        double ratio = double(min_shape[d] - 1) / double(params.remove[d]);
        for (std::ptrdiff_t r = params.remove[d] - 1; r >= 0; --r) {
          indices[d].erase(std::begin(indices[d]) + int(round(r * ratio)));
        }
        EXPECT_EQ(indices[d].size(), min_shape[d] - params.remove[d]);
      }

      builder.Add(indices[d]);
    }
  }

  auto selection = builder.Build();
  EXPECT_EQ(selection.Size(), 3);
  ASSERT_OK_AND_ASSIGN(var_data, ntp->GetColumn("VAR_DATA", selection));
  ASSERT_OK_AND_ASSIGN(str_data, ntp->GetColumn("VAR_STRING_DATA", selection));

  auto GetIndexSpan = [&](auto dim, auto dim_size) -> IndexSpan {
    if (auto r = selection.FSpan(dim, kNDim); r.ok()) return r.ValueOrDie();
    return IndexSpan(dummy_index).subspan(0, dim_size);
  };

  auto var_data_buffer = var_data->data()
                             ->child_data[0]  // chan
                             ->child_data[0]  // corr
                             ->child_data[0]  // complex pair
                             ->buffers[1]
                             ->span_as<Complex>();

  // Get the underyling string array
  str_data = arrow::MakeArray(str_data->data()
                                  ->child_data[0]    // chan
                                  ->child_data[0]);  // corr

  auto rows = GetIndexSpan(2, nRow());
  std::size_t row_offset = 0;

  // Check that the expected value at the selected disk indices
  // matches the buffer values
  for (std::size_t r = 0; r < rows.size(); ++r) {
    auto row = rows[r];
    EXPECT_TRUE(row < nRow());  // sanity
    const auto& shape = row_shapes_[row];
    auto chans = GetIndexSpan(1, shape[1]);
    auto corrs = GetIndexSpan(0, shape[0]);
    for (std::size_t ch = 0; ch < chans.size(); ++ch) {
      auto chan = chans[ch];
      EXPECT_TRUE(chan < shape[1]);  // sanity
      for (std::size_t co = 0; co < corrs.size(); ++co) {
        auto corr = corrs[co];
        EXPECT_TRUE(corr < shape[0]);  // sanity
        auto real = RealValue(row, chan, corr);
        auto buf_i = row_offset + ch * corrs.size() + co;
        EXPECT_EQ(var_data_buffer[buf_i], Complex(real, real));
        ASSERT_OK_AND_ASSIGN(auto str_scalar, str_data->GetScalar(buf_i));
        EXPECT_EQ(str_scalar->ToString(), "FOO-" + std::to_string(real));
      }
    }
    // Advance row offset within the output buffer
    row_offset += chans.size() * corrs.size();
  }

  // All values in the output have been considered
  EXPECT_EQ(row_offset, var_data_buffer.size());
  EXPECT_EQ(row_offset, str_data->length());
}

TEST_F(VariableProxyTest, NegativeSelection) {
  ASSERT_OK_AND_ASSIGN(auto ntp, OpenTable());

  // Derive minimum viable dimension size.
  casacore::IPosition min_shape(kNDim, std::numeric_limits<ssize_t>::max());
  min_shape[2] = nRow();  // row dimension is trivial
  for (const auto& shape : row_shapes_) {
    EXPECT_EQ(shape.size(), kNDim - 1);
    for (std::size_t d = 0; d < shape.size(); ++d) {
      min_shape[d] = std::min(min_shape[d], shape[d]);
    }
  }

  // Create a selection with negative indices
  Index rows(min_shape[2]);
  Index chans(min_shape[1]);
  Index corrs(min_shape[0]);

  std::iota(std::begin(rows), std::end(rows), 0);
  std::iota(std::begin(chans), std::end(chans), 0);
  std::iota(std::begin(corrs), std::end(corrs), 0);

  for (auto r : {1, 3}) rows[r] = -1;
  for (auto ch : {1, 3, 11}) chans[ch] = -1;
  for (auto co : {2}) corrs[co] = -1;

  auto selection = SelectionBuilder().Order('F').Add(corrs).Add(chans).Add(rows).Build();

  // Create a result array filled with a default value
  std::shared_ptr<arrow::Array> result;
  std::vector<float> values(rows.size() * chans.size() * corrs.size() * 2,
                            kDefaultRealValue);
  arrow::ArrayFromVector<arrow::FloatType>(arrow::float32(), values, &result);
  ASSERT_OK_AND_ASSIGN(result, arrow::FixedSizeListArray::FromArrays(result, 2));
  ASSERT_OK_AND_ASSIGN(result,
                       arrow::FixedSizeListArray::FromArrays(result, corrs.size()));
  ASSERT_OK_AND_ASSIGN(result,
                       arrow::FixedSizeListArray::FromArrays(result, chans.size()));
  EXPECT_EQ(result->length(), rows.size());
  ASSERT_OK_AND_ASSIGN(auto data, ntp->GetColumn("VAR_DATA", selection, result));

  // Get the underlying buffer containing the complex values
  auto buffer = data->data()
                    ->child_data[0]  // chan
                    ->child_data[0]  // corr
                    ->child_data[0]  // complex pair
                    ->buffers[1]     // value buffer
                    ->span_as<Complex>();

  std::size_t row_offset = 0;

  // Check that the expected value at the selected disk indices
  // matches the buffer values
  for (std::size_t r = 0; r < rows.size(); ++r) {
    auto row = rows[r];
    EXPECT_TRUE(row < IndexType(nRow()));  // sanity
    for (std::size_t ch = 0; ch < chans.size(); ++ch) {
      auto chan = chans[ch];
      for (std::size_t co = 0; co < corrs.size(); ++co) {
        auto corr = corrs[co];
        auto buf_i = row_offset + ch * corrs.size() + co;
        if (row < 0 || chan < 0 || corr < 0) {
          EXPECT_EQ(buffer[buf_i], Complex(kDefaultRealValue, kDefaultRealValue));
        } else {
          auto real = RealValue(row, chan, corr);
          EXPECT_EQ(buffer[buf_i], Complex(real, real));
        }
      }
    }
    // Advance row offset within the output buffer
    row_offset += chans.size() * corrs.size();
  }

  // All values in the output have been considered
  EXPECT_EQ(row_offset, buffer.size());
}

TEST_F(VariableProxyTest, MissingSelection) {
  ASSERT_OK_AND_ASSIGN(auto ntp, OpenTable());

  ASSERT_OK_AND_ASSIGN(auto empty_shapes, ntp->GetRowShapes("VAR_SPARSE_DATA"));
  auto fsl = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(empty_shapes);
  EXPECT_TRUE(fsl);
  // The column is completely empty
  EXPECT_EQ(fsl->length(), fsl->null_count());
  // ndim == 2
  EXPECT_EQ(fsl->list_type()->list_size(), 2);

  // Derive minimum viable dimension size.
  casacore::IPosition min_shape(kNDim, std::numeric_limits<ssize_t>::max());
  min_shape[2] = nRow();  // row dimension is trivial
  for (const auto& shape : row_shapes_) {
    EXPECT_EQ(shape.size(), kNDim - 1);
    for (std::size_t d = 0; d < shape.size(); ++d) {
      min_shape[d] = std::min(min_shape[d], shape[d]);
    }
  }

  EXPECT_EQ(min_shape, IPosition({3, 14, 4}));

  // Create a selection with negative indices
  Index rows(min_shape[2]);
  Index chans(min_shape[1]);
  Index corrs(min_shape[0]);

  std::iota(std::begin(rows), std::end(rows), 0);
  std::iota(std::begin(chans), std::end(chans), 0);
  std::iota(std::begin(corrs), std::end(corrs), 0);

  std::array<IndexType, 2> missing_rows = {1, 3};
  for (auto r : missing_rows) rows[r] = -1;

  auto selection = SelectionBuilder().Order('F').Add(rows).Build();

  ASSERT_OK_AND_ASSIGN(auto test_data, ntp->GetColumn("VAR_DATA", {}));
  ASSERT_OK(ntp->PutColumn("VAR_SPARSE_DATA", test_data, selection));
  ASSERT_OK_AND_ASSIGN(auto row_shapes, ntp->GetRowShapes("VAR_SPARSE_DATA"));

  std::shared_ptr<arrow::Array> result;
  std::vector<float> values(rows.size() * chans.size() * corrs.size() * 2,
                            kDefaultRealValue);
  arrow::ArrayFromVector<arrow::FloatType>(arrow::float32(), values, &result);
  ASSERT_OK_AND_ASSIGN(result, arrow::FixedSizeListArray::FromArrays(result, 2));
  ASSERT_OK_AND_ASSIGN(result,
                       arrow::FixedSizeListArray::FromArrays(result, corrs.size()));
  ASSERT_OK_AND_ASSIGN(result,
                       arrow::FixedSizeListArray::FromArrays(result, chans.size()));
  EXPECT_EQ(result->length(), rows.size());
  ASSERT_OK_AND_ASSIGN(auto data, ntp->GetColumn("VAR_SPARSE_DATA", {}, result));

  // Get the underlying buffer containing the complex values
  auto buffer = data->data()
                    ->child_data[0]  // chan
                    ->child_data[0]  // corr
                    ->child_data[0]  // complex pair
                    ->buffers[1]     // value buffer
                    ->span_as<Complex>();

  std::size_t row_offset = 0;

  // Check that the expected value at the selected disk indices
  // matches the buffer values
  for (std::size_t r = 0; r < rows.size(); ++r) {
    auto row = rows[r];
    EXPECT_TRUE(row < IndexType(nRow()));  // sanity
    for (std::size_t ch = 0; ch < chans.size(); ++ch) {
      auto chan = chans[ch];
      for (std::size_t co = 0; co < corrs.size(); ++co) {
        auto corr = corrs[co];
        auto buf_i = row_offset + ch * corrs.size() + co;
        if (row < 0 || chan < 0 || corr < 0) {
          EXPECT_EQ(buffer[buf_i], Complex(kDefaultRealValue, kDefaultRealValue))
              << row << ' ' << chan << ' ' << corr;
        } else {
          auto real = RealValue(row, chan, corr);
          EXPECT_EQ(buffer[buf_i], Complex(real, real))
              << row << ' ' << chan << ' ' << corr;
        }
      }
    }
    // Advance row offset within the output buffer
    row_offset += chans.size() * corrs.size();
  }

  // All values in the output have been considered
  EXPECT_EQ(row_offset, buffer.size());
}

}  // namespace
