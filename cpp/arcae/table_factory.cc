/// Parts of this file were originally contributed to
/// https://github.com/casacore/python-casacore under a LGPLv3 license by
/// https://github.com/sjperkins in the following PRs:
/// - https://github.com/casacore/python-casacore/pull/72
/// - https://github.com/casacore/python-casacore/pull/183
/// More generally https://github.com/sjperkins made all contributions
/// to pyms.cc up until python-casacore 3.5.2
/// - https://github.com/casacore/python-casacore/commits/v3.5.2/src/pyms.cc
/// Parts of this code are relicensed here under the BSD-3 license
/// available at https://github.com/ratt-ru/arcae/blob/main/LICENSE

#include "arcae/table_factory.h"

#include <algorithm>
#include <memory>

#include <arrow/api.h>
#include <arrow/status.h>

#include <casacore/casa/Exceptions/Error.h>
#include <casacore/casa/Json.h>
#include <casacore/casa/Json/JsonKVMap.h>
#include <casacore/casa/Json/JsonParser.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/tables/Tables.h>
#include <casacore/tables/Tables/SetupNewTab.h>
#include <casacore/tables/Tables/TableProxy.h>

#include "arcae/descriptor.h"
#include "arcae/new_table_proxy.h"
#include "arcae/table_utils.h"

using namespace std::literals;

using ::arrow::Result;
using ::arrow::Status;

using ::casacore::JsonParser;
using ::casacore::Table;
using ::casacore::TableProxy;

using ::casacore::MeasurementSet;
using ::casacore::MS;
using ::casacore::MSAntenna;
using ::casacore::MSDataDescription;
using ::casacore::MSDoppler;
using ::casacore::MSFeed;
using ::casacore::MSField;
using ::casacore::MSFlagCmd;
using ::casacore::MSFreqOffset;
using ::casacore::MSHistory;
using ::casacore::MSObservation;
using ::casacore::MSPointing;
using ::casacore::MSPolarization;
using ::casacore::MSProcessor;
using ::casacore::MSSource;
using ::casacore::MSSpectralWindow;
using ::casacore::MSState;
using ::casacore::MSSysCal;
using ::casacore::MSWeather;

namespace arcae {
namespace {

/// Table and subtable names
static constexpr char kMain[] = "MAIN";

}  // namespace

Result<std::shared_ptr<NewTableProxy>> OpenTable(const std::string& filename,
                                                 std::size_t ninstances, bool readonly,
                                                 const std::string& json_lockoptions,
                                                 const std::string& json_cache_size) {
  ARROW_ASSIGN_OR_RAISE(auto cache_spec, detail::ParseCacheSizeSpec(json_cache_size));
  return NewTableProxy::Make(
      [&filename, &readonly, &json_lockoptions,
       &cache_spec]() -> Result<std::shared_ptr<TableProxy>> {
        auto lock_record = JsonParser::parse(json_lockoptions).toRecord();
        try {
          auto proxy = std::make_shared<TableProxy>(filename, lock_record,
                                                    Table::TableOption::Old);
          if (!readonly) proxy->reopenRW();
          ARROW_RETURN_NOT_OK(detail::ApplyCacheSizes(*proxy, cache_spec));
          return proxy;
        } catch (std::exception& e) {
          return Status::Invalid(e.what());
        }
      },
      ninstances);
}

Result<std::shared_ptr<NewTableProxy>> DefaultMS(const std::string& name,
                                                 const std::string& subtable,
                                                 const std::string& json_table_desc,
                                                 const std::string& json_dminfo,
                                                 const std::string& json_cache_size) {
  // Upper case subtable name
  casacore::String usubtable(subtable.size(), '0');
  std::transform(std::begin(subtable), std::end(subtable), std::begin(usubtable),
                 [](unsigned char c) { return std::toupper(c); });

  auto modname = name.empty() ? "measurementset.ms"s : name;

  // Subtables are relative to the MS name
  if (usubtable != kMain) {
    modname.append(1, '/');
    modname.append(usubtable);
  }

  ARROW_ASSIGN_OR_RAISE(
      auto setup_new_table,
      DefaultMSFactory(modname, usubtable, json_table_desc, json_dminfo));
  ARROW_ASSIGN_OR_RAISE(auto cache_spec, detail::ParseCacheSizeSpec(json_cache_size));

  return NewTableProxy::Make([&]() -> Result<std::shared_ptr<TableProxy>> {
    // MAIN Measurement Set case
    if (usubtable.empty() || usubtable == kMain) {
      auto ms = MeasurementSet(setup_new_table);
      // Create the MS default subtables
      ms.createDefaultSubtables(Table::New);
      // Create a table proxy
      auto proxy = std::make_shared<TableProxy>(ms);
      ARROW_RETURN_NOT_OK(detail::ApplyCacheSizes(*proxy, cache_spec));
      return proxy;
    }

    // Open the base Measurement Set table in order to link the subtable
    Table ms;

    try {
      ms = Table(name, Table::Update);
    } catch (const casacore::AipsError& error) {
      return arrow::Status::IOError("Error opening Measurement Set ", name,
                                    " for linking against subtable ", usubtable, ": ",
                                    error.what());
    }

    // Create the subtable
    std::shared_ptr<TableProxy> subtable = nullptr;

    if (usubtable == MS::keywordName(MS::ANTENNA)) {
      subtable = std::make_shared<TableProxy>(MSAntenna(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::DATA_DESCRIPTION)) {
      subtable = std::make_shared<TableProxy>(MSDataDescription(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::DOPPLER)) {
      subtable = std::make_shared<TableProxy>(MSDoppler(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::FEED)) {
      subtable = std::make_shared<TableProxy>(MSFeed(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::FIELD)) {
      subtable = std::make_shared<TableProxy>(MSField(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::FLAG_CMD)) {
      subtable = std::make_shared<TableProxy>(MSFlagCmd(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::FREQ_OFFSET)) {
      subtable = std::make_shared<TableProxy>(MSFreqOffset(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::HISTORY)) {
      subtable = std::make_shared<TableProxy>(MSHistory(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::OBSERVATION)) {
      subtable = std::make_shared<TableProxy>(MSObservation(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::POINTING)) {
      subtable = std::make_shared<TableProxy>(MSPointing(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::POLARIZATION)) {
      subtable = std::make_shared<TableProxy>(MSPolarization(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::PROCESSOR)) {
      subtable = std::make_shared<TableProxy>(MSProcessor(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::SOURCE)) {
      subtable = std::make_shared<TableProxy>(MSSource(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::SPECTRAL_WINDOW)) {
      subtable = std::make_shared<TableProxy>(MSSpectralWindow(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::STATE)) {
      subtable = std::make_shared<TableProxy>(MSState(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::SYSCAL)) {
      subtable = std::make_shared<TableProxy>(MSSysCal(setup_new_table));
    } else if (usubtable == MS::keywordName(MS::WEATHER)) {
      subtable = std::make_shared<TableProxy>(MSWeather(setup_new_table));
    }

    if (!subtable) {
      return arrow::Status::Invalid("Uknown table type: ", usubtable);
    }

    // Link the table against the Measurement Set
    ms.rwKeywordSet().defineTable(usubtable, subtable->table());
    ARROW_RETURN_NOT_OK(detail::ApplyCacheSizes(*subtable, cache_spec));
    return subtable;
  });
}

// Execute a TAQL query on the supplied tables
Result<std::shared_ptr<NewTableProxy>> Taql(
    const std::string& taql, const std::vector<std::shared_ptr<NewTableProxy>>& tables) {
  // Easy case
  if (tables.size() == 0) {
    return NewTableProxy::Make([taql = taql]() -> Result<std::shared_ptr<TableProxy>> {
      return std::make_shared<TableProxy>(taql, std::vector<TableProxy>{});
    });
  } else if (tables.size() == 1) {
    return tables[0]->Spawn(
        [taql = taql](const TableProxy& tp) -> Result<std::shared_ptr<TableProxy>> {
          return std::make_shared<TableProxy>(taql, std::vector<TableProxy>{tp});
        });
  }

  return Status::NotImplemented("Taql queries with more than one table argument");
}

}  // namespace arcae
