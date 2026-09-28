#include <cstddef>
#include <cstdint>
#include <cstring>
#include <queue>
#include <vector>

namespace arcae {

// Types supported by PartitionMerge. Expand this with more types as needed
enum class MergeType { INT32, INT64, FLOAT32, FLOAT64 };

namespace {

// Three-way comparison of lhs[i] and rhs[j]
template <typename T>
int Compare(const void * lhs, std::size_t i, const void * rhs, std::size_t j) {
  auto lhs_value = static_cast<const T *>(lhs)[i];
  auto rhs_value = static_cast<const T *>(rhs)[j];
  return (lhs_value > rhs_value) - (lhs_value < rhs_value);
}

using CompareFn = int (*)(const void *, std::size_t, const void *, std::size_t);

struct TypeInfo {
  CompareFn compare;
  std::size_t itemsize;
};

TypeInfo GetTypeInfo(MergeType type) {
  switch (type) {
    case MergeType::INT32: return {&Compare<std::int32_t>, sizeof(std::int32_t)};
    case MergeType::INT64: return {&Compare<std::int64_t>, sizeof(std::int64_t)};
    case MergeType::FLOAT32: return {&Compare<float>, sizeof(float)};
    case MergeType::FLOAT64: return {&Compare<double>, sizeof(double)};
  }
  return {nullptr, 0};
}

// Priority queue element, pointing at a row within a partition
struct MergeData {
  std::size_t row;
  std::size_t partition;
};

}  // namespace

// Performs a k-way merge of lexicographically sorted partitions.
//
// inputs[p][a] points to the contiguous, native-endian data of array a
// in partition p, which contains nrows[p] elements of types[a].
// outputs[a] must have space for the sum of nrows elements of types[a].
// The caller is responsible for validating these invariants and for
// keeping all buffers alive for the duration of the call.
//
// No Python objects are touched, so this may be called without the GIL.
static void PartitionMerge(const std::vector<std::vector<const void *>> & inputs,
                           const std::vector<std::size_t> & nrows,
                           const std::vector<MergeType> & types,
                           const std::vector<void *> & outputs) {
  std::vector<TypeInfo> info(types.size());
  for (std::size_t a = 0; a < types.size(); ++a) info[a] = GetTypeInfo(types[a]);

  // Returns true if lhs sorts after rhs,
  // so that the priority queue is a min-heap
  auto greater = [&](const MergeData & lhs, const MergeData & rhs) {
    for (std::size_t a = 0; a < info.size(); ++a) {
      int cmp = info[a].compare(inputs[lhs.partition][a], lhs.row,
                                inputs[rhs.partition][a], rhs.row);
      if (cmp != 0) return cmp > 0;
    }
    return false;
  };

  std::priority_queue<MergeData, std::vector<MergeData>, decltype(greater)> queue(greater);
  for (std::size_t p = 0; p < inputs.size(); ++p) {
    if (nrows[p] > 0) queue.push(MergeData{0, p});
  }

  for (std::size_t row = 0; !queue.empty(); ++row) {
    auto [prow, p] = queue.top();
    queue.pop();
    for (std::size_t a = 0; a < info.size(); ++a) {
      auto itemsize = info[a].itemsize;
      std::memcpy(static_cast<char *>(outputs[a]) + row * itemsize,
                  static_cast<const char *>(inputs[p][a]) + prow * itemsize,
                  itemsize);
    }
    if (prow + 1 < nrows[p]) queue.push(MergeData{prow + 1, p});
  }
}

}  // namespace arcae
