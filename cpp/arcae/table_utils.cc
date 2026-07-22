#include "arcae/table_utils.h"

#include <exception>
#include <set>
#include <string>
#include <vector>

#include <arrow/api.h>

#include <casacore/casa/BasicSL/String.h>
#include <casacore/casa/Containers/Record.h>
#include <casacore/casa/Json.h>
#include <casacore/casa/Json/JsonKVMap.h>
#include <casacore/casa/Json/JsonParser.h>
#include <casacore/casa/Utilities/DataType.h>
#include <casacore/tables/Tables/TableProxy.h>

using ::arrow::Result;
using ::arrow::Status;

using ::casacore::DataType;
using ::casacore::JsonParser;
using ::casacore::Record;
using ::casacore::String;
using ::casacore::TableProxy;
using ::casacore::Vector;

namespace arcae {
namespace detail {

namespace {

// Prefixes distinguishing the storage-manager and column levels of a
// cache_size specification.
constexpr char kStManPrefix[] = "stman:";
constexpr char kColumnPrefix[] = "column:";

// Returns true and strips the prefix from name if it starts with prefix.
bool StripPrefix(std::string& name, const std::string& prefix) {
  if (name.rfind(prefix, 0) != 0) return false;
  name.erase(0, prefix.size());
  return true;
}

}  // namespace

Status ColumnExists(const TableProxy& tp, const std::string& column) {
  if (tp.table().tableDesc().isColumn(column)) return Status::OK();
  return Status::Invalid("Column ", column, " does not exist");
}

bool IsPrimitiveType(DataType data_type) {
  return casacore::isNumeric(data_type) || data_type == DataType::TpBool ||
         data_type == DataType::TpInt64;
}

bool MaybeReopenRW(TableProxy& tp) {
  if (tp.isWritable()) return false;
  tp.reopenRW();
  return true;
}

Result<CacheSizeSpec> ParseCacheSizeSpec(const std::string& json) {
  CacheSizeSpec spec;
  Record record;
  try {
    record = JsonParser::parse(json).toRecord();
  } catch (std::exception& e) {
    return Status::Invalid("Failed to parse cache_size '", json, "': ", e.what());
  }

  for (casacore::uInt i = 0; i < record.nfields(); ++i) {
    std::string name = record.name(i);
    int value;
    try {
      value = record.asInt(i);
    } catch (std::exception&) {
      return Status::Invalid("cache_size value for '", name, "' is not an integer");
    }
    if (value < 0) {
      return Status::Invalid("cache_size value for '", name, "' must be >= 0");
    }

    if (name == "default") {
      spec.default_mib = value;
    } else if (StripPrefix(name, kStManPrefix)) {
      spec.sm_mib.emplace(std::move(name), value);
    } else if (StripPrefix(name, kColumnPrefix)) {
      spec.col_mib.emplace(std::move(name), value);
    } else {
      return Status::Invalid("Unknown cache_size key '", record.name(i),
                             "'. Expected 'default', 'stman:<NAME>' or 'column:<NAME>'");
    }
  }

  return spec;
}

Status ApplyCacheSizes(TableProxy& tp, const CacheSizeSpec& spec) {
  int default_mib = spec.default_mib.value_or(kDefaultCacheSizeMiB);

  // Map each storage-manager NAME to the columns it serves (and collect the
  // full set of columns) from the data manager info.
  Record dminfo = tp.getDataManagerInfo();
  std::map<std::string, std::vector<std::string>> sm_columns;
  std::set<std::string> all_columns;
  for (casacore::uInt i = 0; i < dminfo.nfields(); ++i) {
    const Record& dm_record = dminfo.subRecord(i);
    std::string sm_name = dm_record.asString("NAME");
    Vector<String> columns = dm_record.asArrayString("COLUMNS");
    std::vector<std::string>& column_vector = sm_columns[sm_name];
    column_vector.reserve(columns.size());
    for (const auto& column : columns) {
      column_vector.push_back(column);
      all_columns.insert(column);
    }
  }

  // Validate every storage-manager / column referenced by the spec exists.
  for (const auto& [sm, mib] : spec.sm_mib) {
    if (sm_columns.find(sm) == sm_columns.end()) {
      return Status::Invalid("Unknown storage manager '", sm, "' in cache_size");
    }
  }
  for (const auto& [column, mib] : spec.col_mib) {
    if (all_columns.find(column) == all_columns.end()) {
      return Status::Invalid("Unknown column '", column, "' in cache_size");
    }
  }

  // A resolved size of <= 0 leaves that scope unbounded (casacore default).
  auto set_column = [&](const std::string& column, int mib) {
    if (mib > 0) tp.setMaximumCacheSize(column, mib);
  };

  // Three passes so that column: > stman: > default precedence holds even for
  // columns that share a storage manager (casacore caps the whole storage
  // manager, last write wins).
  for (const auto& column : all_columns) set_column(column, default_mib);
  for (const auto& [sm, mib] : spec.sm_mib) {
    for (const auto& column : sm_columns[sm]) set_column(column, mib);
  }
  for (const auto& [column, mib] : spec.col_mib) set_column(column, mib);

  return Status::OK();
}

}  // namespace detail
}  // namespace arcae
