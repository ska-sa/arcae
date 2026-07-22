#include <filesystem>
#include <memory>
#include <string>

#include <arrow/result.h>
#include <arrow/testing/gtest_util.h>

#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/BasicSL/Complex.h>
#include <casacore/tables/DataMan/TiledColumnStMan.h>
#include <casacore/tables/DataMan/TiledStManAccessor.h>
#include <casacore/tables/Tables/ArrColDesc.h>
#include <casacore/tables/Tables/ColumnDesc.h>
#include <casacore/tables/Tables/ScaColDesc.h>
#include <casacore/tables/Tables/SetupNewTab.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/TableDesc.h>
#include <casacore/tables/Tables/TableProxy.h>

#include <tests/test_utils.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "arcae/new_table_proxy.h"
#include "arcae/table_factory.h"
#include "arcae/table_utils.h"

using ::arcae::OpenTable;
using ::arcae::detail::ApplyCacheSizes;
using ::arcae::detail::kDefaultCacheSizeMiB;
using ::arcae::detail::ParseCacheSizeSpec;

using ::casacore::ArrayColumnDesc;
using ::casacore::ColumnDesc;
using ::casacore::Complex;
using ::casacore::IPosition;
using ::casacore::ROTiledStManAccessor;
using ::casacore::ScalarColumnDesc;
using ::casacore::SetupNewTable;
using ::casacore::Table;
using ::casacore::TableDesc;
using ::casacore::TableProxy;

using namespace std::string_literals;

namespace {

constexpr std::size_t knrow = 10;
constexpr std::size_t knchan = 16;
constexpr std::size_t kncorr = 4;

// Names of the two tiled storage managers created in the test table.
constexpr char kDataSm[] = "TiledData";
constexpr char kFlagSm[] = "TiledFlag";

class CacheSizeTest : public ::testing::Test {
 protected:
  std::string table_name_;

  void SetUp() override {
    auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    table_name_ = info->name() + "-"s + arcae::hexuuid(4) + ".table"s;

    auto shape = IPosition({kncorr, knchan});
    auto tile_shape = IPosition({kncorr, knchan, 1});

    TableDesc td;
    td.addColumn(ArrayColumnDesc<Complex>("DATA", shape, ColumnDesc::FixedShape));
    td.addColumn(ArrayColumnDesc<casacore::Bool>("FLAG", shape, ColumnDesc::FixedShape));
    td.addColumn(ScalarColumnDesc<casacore::Double>("TIME"));

    SetupNewTable setup(table_name_, td, Table::New);
    casacore::TiledColumnStMan data_sm(kDataSm, tile_shape);
    casacore::TiledColumnStMan flag_sm(kFlagSm, tile_shape);
    setup.bindColumn("DATA", data_sm);
    setup.bindColumn("FLAG", flag_sm);
    // TIME is left bound to the default (non-tiled) storage manager.

    Table table(setup, knrow);
  }

  void TearDown() override { std::filesystem::remove_all(table_name_); }

  // maximumCacheSize (MiB) of the named tiled storage manager.
  casacore::uInt MaxCacheSize(const Table& table, const std::string& sm) {
    return ROTiledStManAccessor(table, sm, false).maximumCacheSize();
  }
};

TEST_F(CacheSizeTest, EmptySpecAppliesBoundedDefault) {
  TableProxy tp(table_name_, casacore::Record(), Table::Update);
  ASSERT_OK_AND_ASSIGN(auto spec, ParseCacheSizeSpec("{}"));
  ASSERT_OK(ApplyCacheSizes(tp, spec));
  EXPECT_EQ(MaxCacheSize(tp.table(), kDataSm), casacore::uInt(kDefaultCacheSizeMiB));
  EXPECT_EQ(MaxCacheSize(tp.table(), kFlagSm), casacore::uInt(kDefaultCacheSizeMiB));
}

TEST_F(CacheSizeTest, ThreeLevelPrecedence) {
  TableProxy tp(table_name_, casacore::Record(), Table::Update);
  // column:DATA overrides its storage manager (TiledData); stman:TiledFlag
  // overrides the default; TIME's storage manager falls back to the default.
  ASSERT_OK_AND_ASSIGN(
      auto spec, ParseCacheSizeSpec(
                     R"({"default": 100, "stman:TiledFlag": 64, "column:DATA": 32})"));
  ASSERT_OK(ApplyCacheSizes(tp, spec));
  EXPECT_EQ(MaxCacheSize(tp.table(), kDataSm), casacore::uInt(32));
  EXPECT_EQ(MaxCacheSize(tp.table(), kFlagSm), casacore::uInt(64));
}

TEST_F(CacheSizeTest, StmanLevelOnly) {
  TableProxy tp(table_name_, casacore::Record(), Table::Update);
  ASSERT_OK_AND_ASSIGN(auto spec, ParseCacheSizeSpec(R"({"stman:TiledData": 256})"));
  ASSERT_OK(ApplyCacheSizes(tp, spec));
  // TiledData explicitly set; TiledFlag falls back to the bounded default.
  EXPECT_EQ(MaxCacheSize(tp.table(), kDataSm), casacore::uInt(256));
  EXPECT_EQ(MaxCacheSize(tp.table(), kFlagSm), casacore::uInt(kDefaultCacheSizeMiB));
}

TEST_F(CacheSizeTest, ZeroDefaultLeavesUnbounded) {
  TableProxy tp(table_name_, casacore::Record(), Table::Update);
  ASSERT_OK_AND_ASSIGN(auto spec, ParseCacheSizeSpec(R"({"default": 0})"));
  ASSERT_OK(ApplyCacheSizes(tp, spec));
  EXPECT_EQ(MaxCacheSize(tp.table(), kDataSm), casacore::uInt(0));
  EXPECT_EQ(MaxCacheSize(tp.table(), kFlagSm), casacore::uInt(0));
}

TEST_F(CacheSizeTest, ParseRejectsUnknownKeyAndBadValue) {
  EXPECT_RAISES_WITH_MESSAGE_THAT(Invalid, ::testing::HasSubstr("Unknown cache_size key"),
                                  ParseCacheSizeSpec(R"({"bogus": 1})"));
  EXPECT_RAISES_WITH_MESSAGE_THAT(Invalid, ::testing::HasSubstr("must be >= 0"),
                                  ParseCacheSizeSpec(R"({"default": -1})"));
}

TEST_F(CacheSizeTest, ApplyRejectsUnknownNames) {
  TableProxy tp(table_name_, casacore::Record(), Table::Update);
  ASSERT_OK_AND_ASSIGN(auto sm_spec, ParseCacheSizeSpec(R"({"stman:NoSuchSm": 1})"));
  EXPECT_RAISES_WITH_MESSAGE_THAT(Invalid,
                                  ::testing::HasSubstr("Unknown storage manager"),
                                  ApplyCacheSizes(tp, sm_spec));
  ASSERT_OK_AND_ASSIGN(auto col_spec, ParseCacheSizeSpec(R"({"column:NOPE": 1})"));
  EXPECT_RAISES_WITH_MESSAGE_THAT(Invalid, ::testing::HasSubstr("Unknown column"),
                                  ApplyCacheSizes(tp, col_spec));
}

TEST_F(CacheSizeTest, OpenTableAppliesCap) {
  // OpenTable applies the spec inside the per-instance open functor, so every
  // instance is capped; RunSync dispatches to one of them and observes it.
  ASSERT_OK_AND_ASSIGN(auto ntp,
                       OpenTable(table_name_, /*ninstances=*/3, /*readonly=*/true,
                                 R"({"option": "auto"})", R"({"stman:TiledData": 200})"));

  ASSERT_OK_AND_ASSIGN(auto data_cap, ntp->Proxy()->RunSync([](const TableProxy& tp) {
    return ROTiledStManAccessor(tp.table(), kDataSm, false).maximumCacheSize();
  }));
  ASSERT_OK_AND_ASSIGN(auto flag_cap, ntp->Proxy()->RunSync([](const TableProxy& tp) {
    return ROTiledStManAccessor(tp.table(), kFlagSm, false).maximumCacheSize();
  }));
  EXPECT_EQ(data_cap, casacore::uInt(200));
  EXPECT_EQ(flag_cap, casacore::uInt(kDefaultCacheSizeMiB));
}

}  // namespace
