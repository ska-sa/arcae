#include <string>

#include <arrow/result.h>

#include <casacore/tables/Tables/SetupNewTab.h>

namespace arcae {

arrow::Result<std::string> MSDescriptor(const std::string& table, bool complete = false);

arrow::Result<casacore::SetupNewTable> DefaultMSFactory(
    const std::string& name, const std::string& subtable,
    const std::string& json_table_desc = "{}", const std::string& json_dminfo = "{}");

// Set up a plain CASA table from a user supplied table descriptor.
// Unlike DefaultMSFactory, no MeasurementSet columns are merged in,
// so the resulting table contains exactly what the caller described
arrow::Result<casacore::SetupNewTable> TableFactory(
    const std::string& name, const std::string& json_table_desc = "{}",
    const std::string& json_dminfo = "{}");

}  // namespace arcae
