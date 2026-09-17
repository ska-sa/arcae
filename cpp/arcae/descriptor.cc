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

#ifndef ARCAE_DESCRIPTOR_H
#define ARCAE_DESCRIPTOR_H

#include <sstream>

#include <arrow/result.h>

#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/Containers/RecordInterface.h>
#include <casacore/casa/Containers/ValueHolder.h>
#include <casacore/casa/Json.h>
#include <casacore/casa/Json/JsonKVMap.h>
#include <casacore/casa/Json/JsonParser.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/tables/Tables/ArrColDesc.h>
#include <casacore/tables/Tables/ScaColDesc.h>
#include <casacore/tables/Tables/TableDesc.h>
#include <casacore/tables/Tables/TableProxy.h>
#include <casacore/tables/Tables/TableRecord.h>

#include "arcae/descriptor.h"

using ::arrow::Result;

using ::casacore::JsonOut;
using ::casacore::JsonParser;
using ::casacore::ArrayColumnDesc;
using ::casacore::Bool;
using ::casacore::Double;
using ::casacore::IPosition;
using ::casacore::Int;
using ::casacore::ScalarColumnDesc;
using ::casacore::String;
using ::casacore::Vector;

using ::casacore::Record;
using ::casacore::RecordInterface;
using ::casacore::SetupNewTable;
using ::casacore::Table;
using ::casacore::TableDesc;
using ::casacore::TableProxy;

using ::casacore::MeasurementSet;
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

// Table and subtable names
static constexpr char kAntenna[] = "ANTENNA";
static constexpr char kMain[] = "MAIN";
static constexpr char kDataDescription[] = "DATA_DESCRIPTION";
static constexpr char kDoppler[] = "DOPPLER";
static constexpr char kFeed[] = "FEED";
static constexpr char kField[] = "FIELD";
static constexpr char kFlagCmd[] = "FLAG_CMD";
static constexpr char kFreqOffset[] = "FREQ_OFFSET";
static constexpr char kHistory[] = "HISTORY";
static constexpr char kObservation[] = "OBSERVATION";
static constexpr char kPhasedArray[] = "PHASED_ARRAY";
static constexpr char kPointing[] = "POINTING";
static constexpr char kPolarization[] = "POLARIZATION";
static constexpr char kProcessor[] = "PROCESSOR";
static constexpr char kSource[] = "SOURCE";
static constexpr char kSpectralWindow[] = "SPECTRAL_WINDOW";
static constexpr char kState[] = "STATE";
static constexpr char kSyscal[] = "SYSCAL";
static constexpr char kWeather[] = "WEATHER";

// Category related columns
static constexpr char kFlagCategory[] = "FLAG_CATEGORY";
static constexpr char kCategory[] = "CATEGORY";

TableDesc MainMSDesc(bool complete) {
  // Get required descriptor
  TableDesc td = MeasurementSet::requiredTableDesc();

  if (!complete) {
    // Remove the CATEGORY keyword from the FLAG_CATEGORY column
    // This empty Vector<String> gets converted to a python dictionary as
    // 'FLAG_CATEGORY' : {
    //     ...
    //     keywords': {'CATEGORY' : []},
    //     ...
    // }
    //
    // Due to the missing type information this gets converted
    // into something like Vector<int> when passed to the C++ layer,
    // which results in Table Conformance errors
    // This is an OK solution since the C++ layer always adds this keyword
    // if it is missing from the MS
    // (see addCat())
    td.rwColumnDesc(kFlagCategory).rwKeywordSet().removeField(kCategory);
    return td;
  }

  using CEnum = typename MeasurementSet::PredefinedColumns;
  using KEnum = typename MeasurementSet::PredefinedKeywords;

  // Add remaining columns
  for (int i = CEnum::NUMBER_REQUIRED_COLUMNS + 1; i <= CEnum::NUMBER_PREDEFINED_COLUMNS;
       ++i) {
    MeasurementSet::addColumnToDesc(td, static_cast<CEnum>(i));
  }

  // Add remaining keywords
  for (int i = KEnum::NUMBER_REQUIRED_KEYWORDS + 1;
       i <= KEnum::NUMBER_PREDEFINED_KEYWORDS; ++i) {
    MeasurementSet::addKeyToDesc(td, static_cast<KEnum>(i));
  }

  return td;
}

template <typename SubTable>
TableDesc MSSubtableDesc(bool complete) {
  if (!complete) {
    return SubTable::requiredTableDesc();
  }

  using CEnum = typename SubTable::PredefinedColumns;

  // Get required descriptor
  TableDesc td = SubTable::requiredTableDesc();

  // Add remaining columns
  for (int i = CEnum::NUMBER_REQUIRED_COLUMNS + 1; i <= CEnum::NUMBER_PREDEFINED_COLUMNS;
       ++i) {
    SubTable::addColumnToDesc(td, static_cast<CEnum>(i));
  }

  // NOTE(sjperkins)
  // Inspection of the casacore code base seems to indicate
  // that there are no optional MS subtable keywords.
  // NUMBER_REQUIRED_KEYWORDS is only defined in the MS
  return td;
}

void AddMeasureMetadata(TableDesc& td, const String& column, const String& type,
                        const String& ref, const String& unit) {
  auto& keywords = td.rwColumnDesc(column).rwKeywordSet();
  keywords.define("QuantumUnits", Vector<String>(1, unit));

  Record measinfo;
  measinfo.define("type", type);
  measinfo.define("Ref", ref);
  keywords.defineRecord("MEASINFO", measinfo);
}

TableDesc PhasedArrayMSDesc(bool complete) {
  TableDesc td;
  td.addColumn(ScalarColumnDesc<Int>("ANTENNA_ID", "Antenna ID"));
  td.addColumn(ArrayColumnDesc<Double>(
      "POSITION", "Position of antenna field", IPosition({3})));
  td.addColumn(ArrayColumnDesc<Double>(
      "COORDINATE_AXES", "Local coordinate system", IPosition({3, 3})));
  td.addColumn(ArrayColumnDesc<Double>(
      "ELEMENT_OFFSET", "Offset per element", 2));
  td.addColumn(ArrayColumnDesc<Bool>(
      "ELEMENT_FLAG", "Flag of elements in array, for each polarization", 2));

  AddMeasureMetadata(td, "POSITION", "position", "ITRF", "m");
  AddMeasureMetadata(td, "COORDINATE_AXES", "direction", "ITRF", "m");
  AddMeasureMetadata(td, "ELEMENT_OFFSET", "position", "ITRF", "m");

  if (complete) {
    td.addColumn(ScalarColumnDesc<Int>("BEAM_ID", "Beam ID"));
  }

  return td;
}

std::string RecordToJson(const Record& record) {
  std::ostringstream json_oss;
  auto record_json = JsonOut(json_oss);
  record_json.start();

  for (casacore::uInt i = 0; i < record.nfields(); ++i) {
    record_json.write(record.name(i), record.asValueHolder(i));
  }

  record_json.end();
  return json_oss.str();
}

// Get the required table descriptions for the given table.
// If "" or "MAIN", the table descriptions for a Measurement Set
// will be supplied, otherwise table should be some valid
// MeasurementSet subtable
Result<TableDesc> MSTableDescriptor(const String& table, bool complete) {
  String table_ = table;

  // Upper case things to be sure
  table_.upcase();

  if (table.empty() || table_ == kMain) {
    return MainMSDesc(complete);
  } else if (table_ == kAntenna) {
    return MSSubtableDesc<MSAntenna>(complete);
  } else if (table_ == kDataDescription) {
    return MSSubtableDesc<MSDataDescription>(complete);
  } else if (table_ == kDoppler) {
    return MSSubtableDesc<MSDoppler>(complete);
  } else if (table_ == kFeed) {
    return MSSubtableDesc<MSFeed>(complete);
  } else if (table_ == kField) {
    return MSSubtableDesc<MSField>(complete);
  } else if (table_ == kFlagCmd) {
    return MSSubtableDesc<MSFlagCmd>(complete);
  } else if (table_ == kFreqOffset) {
    return MSSubtableDesc<MSFreqOffset>(complete);
  } else if (table_ == kHistory) {
    return MSSubtableDesc<MSHistory>(complete);
  } else if (table_ == kObservation) {
    return MSSubtableDesc<MSObservation>(complete);
  } else if (table_ == kPhasedArray) {
    return PhasedArrayMSDesc(complete);
  } else if (table_ == kPointing) {
    return MSSubtableDesc<MSPointing>(complete);
  } else if (table_ == kPolarization) {
    return MSSubtableDesc<MSPolarization>(complete);
  } else if (table_ == kProcessor) {
    return MSSubtableDesc<MSProcessor>(complete);
  } else if (table_ == kSource) {
    return MSSubtableDesc<MSSource>(complete);
  } else if (table_ == kSpectralWindow) {
    return MSSubtableDesc<MSSpectralWindow>(complete);
  } else if (table_ == kState) {
    return MSSubtableDesc<MSState>(complete);
  } else if (table_ == kSyscal) {
    return MSSubtableDesc<MSSysCal>(complete);
  } else if (table_ == kWeather) {
    return MSSubtableDesc<MSWeather>(complete);
  }

  return arrow::Status::Invalid("Unknown table type ", table_);
}

// Merge required and user supplied Table Descriptions
TableDesc MergeRequiredAndUserTableDescs(const TableDesc& required_td,
                                         const TableDesc& user_td) {
  TableDesc result = required_td;

  // Overwrite required columns with user columns
  for (casacore::uInt i = 0; i < user_td.ncolumn(); ++i) {
    const String& name = user_td[i].name();

    // Remove if present in required
    if (result.isColumn(name)) {
      result.removeColumn(name);
    }

    // Add the column
    result.addColumn(user_td[i]);
  }

  // Overwrite required hypercolumns with user hypercolumns
  // In practice this shouldn't be necessary as requiredTableDesc
  // doesn't define hypercolumns by default...
  Vector<String> user_hc = user_td.hypercolumnNames();

  for (casacore::uInt i = 0; i < user_hc.size(); ++i) {
    // Remove if hypercolumn is present
    if (result.isHypercolumn(user_hc[i])) {
      result.removeHypercolumnDesc(user_hc[i]);
    }

    Vector<String> dataColumnNames;
    Vector<String> coordColumnNames;
    Vector<String> idColumnNames;

    // Get the user hypercolumn
    casacore::uInt ndims = user_td.hypercolumnDesc(user_hc[i], dataColumnNames,
                                                   coordColumnNames, idColumnNames);
    // Place it in result
    result.defineHypercolumn(user_hc[i], ndims, dataColumnNames, coordColumnNames,
                             idColumnNames);
  }

  // Overwrite required keywords with user keywords
  result.rwKeywordSet().merge(user_td.keywordSet(), RecordInterface::OverwriteDuplicates);

  return result;
}

}  // namespace

// Get the table descriptions for the given table.
// If "" or "MAIN", the table descriptions for a Measurement Set
// will be supplied, otherwise table should be some valid
// MeasurementSet subtable.
// If complete is true, the full descriptor is returned, otherwise
// only the required descriptor is returned
Result<std::string> MSDescriptor(const std::string& table, bool complete) {
  ARROW_ASSIGN_OR_RAISE(auto table_desc, MSTableDescriptor(table, complete));
  return RecordToJson(TableProxy::getTableDesc(table_desc, true));
}

Result<SetupNewTable> DefaultMSFactory(const std::string& name,
                                       const std::string& subtable,
                                       const std::string& json_table_desc,
                                       const std::string& json_dminfo) {
  String msg;
  TableDesc user_td;
  auto table_desc = JsonParser::parse(json_table_desc).toRecord();

  // Create Table Description object from extra user table description
  if (!TableProxy::makeTableDesc(table_desc, user_td, msg)) {
    return arrow::Status::Invalid("Failed to create Table Description", msg);
  }

  ARROW_ASSIGN_OR_RAISE(auto required_desc, MSTableDescriptor(subtable, false));

  // Merge required and user table descriptions
  TableDesc final_desc = MergeRequiredAndUserTableDescs(required_desc, user_td);

  // Return SetupNewTable object
  SetupNewTable setup = SetupNewTable(name, final_desc, Table::New);

  // Apply any data manager info
  auto dminfo = JsonParser::parse(json_dminfo).toRecord();
  setup.bindCreate(dminfo);

  return setup;
}

}  // namespace arcae

#endif  // ARCAE_DESCRIPTOR_H
