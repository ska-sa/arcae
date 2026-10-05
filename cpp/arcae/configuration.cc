#include "arcae/configuration.h"

#include <string>

#include <arrow/result.h>
#include <arrow/status.h>

using ::arrow::Result;
using ::arrow::Status;

namespace arcae {

Status SafeMultiThreadedWrites() { return Status::OK(); }

Result<std::string> Configuration::Get(const std::string& key) const {
  if (auto it = kvmap_.find(key); it != kvmap_.end()) {
    return it->second;
  }
  return Status::KeyError(key);
}

std::string Configuration::GetDefault(const std::string& key,
                                      std::string default_value) const {
  if (auto it = kvmap_.find(key); it != kvmap_.end()) {
    return it->second;
  }
  return default_value;
}

Result<bool> Configuration::Delete(const std::string& key) {
  if (auto it = kvmap_.find(key); it != kvmap_.end()) {
    kvmap_.erase(it);
    return true;
  }
  return Status::KeyError(key);
}

}  // namespace arcae
