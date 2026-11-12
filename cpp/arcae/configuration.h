#ifndef ARCAE_CONFIG_H
#define ARCAE_CONFIG_H

#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include <arrow/result.h>

namespace arcae {

// Returns OK if safe multi-threaded write support is
// generally available
arrow::Status SafeMultiThreadedWrites();

// Thread-safe: Python threads set and delete keys with the GIL released
// (or under free-threading), while C++ worker threads read them.
class Configuration {
 private:
  mutable std::shared_mutex mutex_;
  std::map<std::string, std::string> kvmap_;

 public:
  std::size_t Size() const {
    std::shared_lock lock(mutex_);
    return kvmap_.size();
  }

  std::vector<std::string> GetKeys() const {
    std::shared_lock lock(mutex_);
    std::vector<std::string> keys;
    for (auto& kv : kvmap_) {
      keys.push_back(kv.first);
    }
    return keys;
  }

  void Set(const std::string& key, std::string value) {
    std::unique_lock lock(mutex_);
    kvmap_[key] = std::move(value);
  }

  arrow::Result<std::string> Get(const std::string& key) const;
  std::string GetDefault(const std::string& key, std::string default_value) const;
  arrow::Result<bool> Delete(const std::string& key);
};

}  // namespace arcae

#endif
