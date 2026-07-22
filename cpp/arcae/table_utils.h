#ifndef ARCAE_TABLE_UTILS_H
#define ARCAE_TABLE_UTILS_H

#include <map>
#include <optional>
#include <string>

#include <arrow/api.h>

#include <casacore/casa/Utilities/DataType.h>
#include <casacore/tables/Tables/TableProxy.h>

namespace arcae {
namespace detail {

// Bounded default per-column storage-manager cache size (MiB) applied when the
// cache_size argument does not otherwise specify a value.
inline constexpr int kDefaultCacheSizeMiB = 128;

// A parsed cache_size specification. All sizes are in MiB (0 == unbounded),
// matching casacore's TiledStMan interpretation of setMaximumCacheSize.
struct CacheSizeSpec {
  std::optional<int> default_mib;      // "default": global default for unmatched columns
  std::map<std::string, int> sm_mib;   // "stman:<NAME>": per storage-manager cap
  std::map<std::string, int> col_mib;  // "column:<NAME>": per-column cap
};

// Returns OK if the ColumnExists otherwise returns an error Status
arrow::Status ColumnExists(const casacore::TableProxy& tp, const std::string& column);

// Returns true if this is a primite CASA type
bool IsPrimitiveType(casacore::DataType data_type);

// Returns true if the table was opened readonly and was re-opened in readwrite mode
// otherwise returns false
bool MaybeReopenRW(casacore::TableProxy& tp);

// Parse a cache_size JSON object into a CacheSizeSpec. Accepts an empty object
// ("{}") which yields an empty spec (the bounded default is then applied).
// Keys must be "default", "stman:<NAME>" or "column:<NAME>"; values must be
// non-negative integers.
arrow::Result<CacheSizeSpec> ParseCacheSizeSpec(const std::string& json);

// Apply per-column maximum storage-manager cache sizes to tp, resolving each
// column with precedence column: > stman: > default > kDefaultCacheSizeMiB.
// Storage-manager and column names referenced by the spec must exist in the
// table (else Status::Invalid). A resolved size of 0 leaves that scope
// unbounded (the casacore default).
arrow::Status ApplyCacheSizes(casacore::TableProxy& tp, const CacheSizeSpec& spec);

}  // namespace detail
}  // namespace arcae

#endif  // ARCAE_TABLE_UTILS_H
