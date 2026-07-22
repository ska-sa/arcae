#ifndef ARCAE_TABLE_FACTORY_H
#define ARCAE_TABLE_FACTORY_H

#include <memory>
#include <string>

#include <arrow/result.h>

#include "arcae/new_table_proxy.h"

namespace arcae {

arrow::Result<std::shared_ptr<NewTableProxy>> OpenTable(
    const std::string& filename, std::size_t ninstances = 1, bool readonly = true,
    const std::string& json_lockoptions = R"({"option": "auto"})",
    const std::string& json_cache_size = "{}");
arrow::Result<std::shared_ptr<NewTableProxy>> DefaultMS(
    const std::string& name, const std::string& subtable = "MAIN",
    const std::string& json_table_desc = "{}", const std::string& json_dminfo = "{}",
    const std::string& json_cache_size = "{}");
arrow::Result<std::shared_ptr<NewTableProxy>> Taql(
    const std::string& taql,
    const std::vector<std::shared_ptr<NewTableProxy>>& tables = {});

}  // namespace arcae

#endif  // ARCAE_TABLE_FACTORY_H
