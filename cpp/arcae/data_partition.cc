#include "arcae/data_partition.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>

#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/Arrays/Slicer.h>
#include <casacore/casa/aipsxtype.h>
#include <casacore/tables/Tables/RefRows.h>

#include <arrow/result.h>

#include "arcae/result_shape.h"
#include "arcae/selection.h"

using ::arrow::Result;
using ::arrow::Status;

using ::casacore::IPosition;
using ::casacore::RefRows;
using ::casacore::rownr_t;
using ::casacore::Slicer;
using ::casacore::Vector;

namespace arcae {
namespace detail {

namespace {

// Given a range of sub-spans in each dimension, determine whether
// the memory ordering is contiguous. This means there must be a
// single range which is an arithmetic progression
// spanning the entire dimension
bool SecondaryDimensionsContiguous(const std::vector<SpanPairs>& spans) {
  // For non-row ranges,
  for (std::size_t d = 0; d < spans.size() - 1; ++d) {
    if (spans[d].size() > 1) return false;
    const auto& mem = spans[d][0].mem;
    for (std::size_t i = 0; i < mem.size(); ++i) {
      if (mem[i] != IndexType(i)) return false;
    }
  }

  return true;
}

// Returns true if the memory ordering is contiguous
bool IsMemoryContiguous(const SpanPair& spans) {
  const auto& mem = spans.mem;
  for (std::size_t i = 1; i < mem.size(); ++i) {
    if (mem[i] - mem[i - 1] != 1) return false;
  }
  return true;
}

struct IndexResult {
  Index disk;
  Index mem;
};

// Given a span of disk indices,
// return the sorted indices, as well as
// the associated argsort (which form the memory indices)
IndexResult MakeSortedIndices(const IndexSpan& ids) {
  std::vector<IndexType> mem(ids.size(), 0);
  std::iota(mem.begin(), mem.end(), 0);
  Index disk(ids.size(), 0);

  // Avoid O(N log N) sort with O(n) check
  if (std::is_sorted(std::begin(ids), std::end(ids))) {
    std::copy(std::begin(ids), std::end(ids), std::begin(disk));
    return IndexResult{std::move(disk), std::move(mem)};
  }

  std::sort(mem.begin(), mem.end(), [&ids](auto l, auto r) { return ids[l] < ids[r]; });
  for (std::size_t i = 0; i < ids.size(); ++i) disk[i] = ids[mem[i]];
  return IndexResult{std::move(disk), std::move(mem)};
}

// Decompose disk and memory spans into span pairs
// associated with contiguous disk id ranges
Result<std::vector<SpanPair>> MakeSubSpans(const IndexSpan& disk_span,
                                           const IndexSpan& mem_span, bool entire_range) {
  assert(disk_span.size() == mem_span.size());
  std::vector<SpanPair> result;
  std::size_t start = 0;

  auto MakeSubSpan = [&](std::size_t current) {
    auto disk = disk_span.subspan(start, current - start);
    auto mem = mem_span.subspan(start, current - start);
    result.emplace_back(SpanPair{std::move(disk), std::move(mem)});
    start = current;
  };

  for (std::size_t i = 1; i < disk_span.size(); ++i) {
    if (disk_span[i] < 0) {
      // negative indices, advance
      continue;
    } else if (disk_span[start] < 0) {
      // start was negative, but disk_span[i] is not
      // create the negative range
      MakeSubSpan(i);
    } else if (disk_span[i] == disk_span[i - 1]) {
      return Status::IndexError("Duplicate selection index ", disk_span[i]);
    } else if (disk_span[i] - disk_span[i - 1] == 1) {
      // monotonic range, advance
      continue;
    } else if (!entire_range) {
      // discontinuity in positive indices
      // create a positive range
      MakeSubSpan(i);
    }
  }

  MakeSubSpan(disk_span.size());
  return result;
}

struct CartesianProductResult {
  std::vector<SpanPairs> dim_spans;
  std::vector<bool> contiguous;
};

// Creates a cartesian product of span pairs
CartesianProductResult CartesianProduct(const std::vector<SpanPairs>& dim_spans) {
  // Total number of elements in the cartesian product
  std::size_t nproducts = 1;
  for (auto& sp : dim_spans) nproducts *= sp.size();

  std::vector<SpanPairs> product;
  product.reserve(nproducts);
  product.push_back({});

  for (const auto& span_pairs : dim_spans) {
    std::vector<SpanPairs> next_product;
    next_product.reserve(nproducts);
    for (const auto& result : product) {
      for (const auto& span_pair : span_pairs) {
        next_product.push_back(result);
        next_product.back().push_back(span_pair);
      }
    }
    product = std::move(next_product);
  }

  assert(nproducts == product.size());
  bool secondary_contig = SecondaryDimensionsContiguous(dim_spans);
  std::vector<bool> contiguous(nproducts, secondary_contig);

  // Exit early if secondary dimensions are not contiguous
  if (!secondary_contig) return {std::move(product), std::move(contiguous)};

  std::size_t stride = 1;
  for (std::size_t d = 1; d < dim_spans.size(); ++d) {
    stride *= dim_spans[d - 1].size();
  }

  auto row_spans = dim_spans[dim_spans.size() - 1];
  std::size_t last = 0;

  for (std::size_t rs = 0; rs < row_spans.size(); ++rs) {
    bool rows_contiguous = IsMemoryContiguous(row_spans[rs]);
    auto first = rs * stride;
    last = (rs + 1) * stride;
    for (std::size_t i = first; i < last; ++i) {
      contiguous[i] = contiguous[i] & rows_contiguous;
    }
  }

  assert(last == nproducts);
  return {std::move(product), std::move(contiguous)};
}

Result<std::vector<DataChunk>> MakeDataChunks(std::vector<SpanPairs>&& dim_spans,
                                              std::vector<bool>&& contiguous,
                                              std::vector<Index>&& id_cache,
                                              const ResultShapeData& data_shape) {
  auto nchunks = dim_spans.size();
  if (nchunks == 0) return std::vector<DataChunk>{};
  auto ndim = dim_spans[0].size();
  // Number of chunks should match
  assert(nchunks == contiguous.size());

  // NOTE: We can really assume offset == nchunks * ndim
  // but check just in case
  std::size_t offset = 0;
  for (std::size_t i = 0; i < nchunks; ++i) {
    // All chunks should have the same dimension size
    assert(dim_spans[i].size() == ndim);
    offset += dim_spans[i].size();
  }

  assert(offset == nchunks * ndim);

  auto shared = std::make_shared<AggregateAdapter<SharedChunkData>>(
      nchunks,                              // nchunks_
      ndim,                                 // ndim_
      data_shape.GetDataType(),             // casa_dtype_
      std::move(id_cache),                  // id_cache_
      std::move(dim_spans),                 // dim_spans_
      std::vector<IndexType>(offset),       // min_elements_
      std::vector<std::size_t>(nchunks),    // flat_offsets_
      std::vector<std::size_t>(offset, 1),  // buffer_stride_
      std::vector<std::size_t>(offset, 1),  // chunk_strides_
      std::vector<std::size_t>(offset, 0),  // position_
      std::move(contiguous));               // contiguous_

  std::vector<DataChunk> chunks(nchunks);
  offset = 0;

  // Iteratively create chunks, pre-computing relevant info
  for (std::size_t chunk = 0; chunk < nchunks; ++chunk) {
    // Compute buffer strides
    const auto& dim_span = shared->dim_spans_[chunk];
    auto ndim = dim_span.size();
    if (data_shape.IsFixed()) {
      const auto& shape = data_shape.GetShape();
      for (std::size_t d = 1, product = 1; d < ndim; ++d) {
        product *= shape[d - 1];
        shared->buffer_strides_[offset + d] *= product;
      }
    } else {
      auto row_span = dim_span[ndim - 1].mem;
      if (row_span.size() != 1) return Status::Invalid("Expected a single row dimension");
      const auto& shape = data_shape.GetRowShape(row_span[0]);
      for (std::size_t d = 1, product = 1; d < ndim; ++d) {
        product *= shape[d - 1];
        shared->buffer_strides_[offset + d] *= product;
      }
    }

    // Compute chunk stride
    for (std::size_t d = 1, product = 1; d < ndim; ++d) {
      product *= dim_span[d - 1].mem.size();
      shared->chunk_strides_[offset + d] *= product;
    }

    // Compute minimum element
    for (std::size_t d = 0; d < ndim; ++d) {
      shared->min_mem_index_[offset + d] = [&]() -> std::size_t {
        if (dim_span[d].mem.size() == 0) return 0;
        return *std::min_element(std::begin(dim_span[d].mem), std::end(dim_span[d].mem));
      }();
    }

    // Compute flat offsets
    auto min_span = absl::MakeSpan(&shared->min_mem_index_[offset], ndim);
    ARROW_ASSIGN_OR_RAISE(shared->flat_offsets_[chunk], data_shape.FlatOffset(min_span));

    // Create the chunk
    chunks[chunk] = DataChunk{chunk, shared};
    offset += dim_span.size();
  }

  assert(offset == ndim * nchunks);
  return chunks;
}

}  // namespace

// Get a Row Slicer for the disk span
Slicer DataChunk::RowSlicer() const noexcept {
  auto row_span = DimensionSpans()[nDim() - 1];
  return Slicer(IPosition({row_span.disk[0]}),
                IPosition({row_span.disk[row_span.disk.size() - 1]}), Slicer::endIsLast);
}

RefRows DataChunk::ReferenceRows() const noexcept {
  auto s = DimensionSpans()[nDim() - 1].disk;
  return RefRows(Vector<rownr_t>(s.begin(), s.end()), false, false);
}

// Get a Section Slicer for the disk span
Slicer DataChunk::SectionSlicer() const noexcept {
  auto dim_spans = DimensionSpans();
  auto row_dim = nDim() - 1;
  IPosition start(row_dim, 0);
  IPosition end(row_dim, 0);

  for (std::size_t d = 0; d < row_dim; ++d) {
    auto span = dim_spans[d];
    start[d] = span.disk[0];
    end[d] = span.disk[span.disk.size() - 1];
  }
  return Slicer(std::move(start), std::move(end), Slicer::endIsLast);
}

// Number of elements in the chunk
std::size_t DataChunk::nElements() const noexcept {
  auto dim_spans = DimensionSpans();
  return std::accumulate(std::begin(dim_spans), std::end(dim_spans), std::size_t(1),
                         [](auto i, auto v) { return i * v.disk.size(); });
}

// FORTRAN ordered shape of the chunk
IPosition DataChunk::GetShape() const noexcept {
  auto dim_spans = DimensionSpans();
  IPosition shape(dim_spans.size(), 0);
  for (std::size_t d = 0; d < dim_spans.size(); ++d) shape[d] = dim_spans[d].disk.size();
  return shape;
}

bool DataChunk::IsEmpty() const noexcept {
  for (const auto& span : DimensionSpans()) {
    for (auto i : span.disk) {
      if (i < 0) return true;
    }
  }
  return nElements() == 0;
}
// Commented out debugging function
std::string DataChunk::ToString() const {
  std::ostringstream oss;
  oss << '\n';
  bool first_dim = true;
  std::size_t nelements = 1;
  auto dim_spans = DimensionSpans();
  for (std::size_t d = 0; d < dim_spans.size(); ++d) {
    if (!first_dim) oss << ' ';
    first_dim = false;
    oss << '[';
    bool first_index = true;
    nelements *= dim_spans[d].mem.size();
    for (auto i : dim_spans[d].mem) {
      if (!first_index) oss << ',';
      first_index = false;
      oss << std::to_string(i);
    }
    oss << ']';
  }

  oss << '\n';

  first_dim = true;
  for (std::size_t d = 0; d < dim_spans.size(); ++d) {
    if (!first_dim) oss << ' ';
    first_dim = false;
    oss << '[';
    bool first_index = true;
    for (auto i : dim_spans[d].disk) {
      if (!first_index) oss << ',';
      first_index = false;
      oss << std::to_string(i);
    }
    oss << ']';
  }

  oss << '\n';

  first_dim = true;
  auto chunk_strides = ChunkStrides();
  oss << '[';
  for (auto cs : chunk_strides) {
    if (!first_dim) oss << ',';
    first_dim = false;
    oss << std::to_string(cs);
  }

  oss << ']' << '\n';

  first_dim = true;
  auto buf_strides = BufferStrides();
  oss << '[';
  for (auto bs : buf_strides) {
    if (!first_dim) oss << ',';
    first_dim = false;
    oss << std::to_string(bs);
  }

  oss << ']' << '\n';

  oss << '\n';

  first_dim = true;
  auto ref_rows = ReferenceRows();
  oss << "row slicer: [";
  for (auto r : ref_rows.rowVector()) {
    if (!first_dim) oss << ',';
    first_dim = false;
    oss << r;
  }
  oss << "] \n";

  // oss << "row slicer:" << ReferenceRows() << '\n';
  oss << "sec slicer:" << SectionSlicer() << '\n';
  oss << " elements: " + std::to_string(nelements);
  oss << " offset: " + std::to_string(FlatOffset());
  oss << " contiguous: ";
  oss << (IsContiguous() ? 'Y' : 'N');

  return oss.str();
}

Result<DataPartition> DataPartition::Make(const Selection& selection,
                                          const ResultShapeData& result_shape) {
  // Construct an id cache, holding Index vectors
  // Spans will be constructed over these vectors
  // Initially, this holds indices that monotically
  // increase up to the maximum dimension size.
  auto id_cache = std::vector<Index>{Index(result_shape.MaxDimensionSize())};
  std::iota(id_cache.back().begin(), id_cache.back().end(), 0);
  auto monotonic_ids = IndexSpan(id_cache.back());
  ARROW_ASSIGN_OR_RAISE(auto result_ndim, result_shape.ConsistentNDim());

  // Generate disk and memory spans for the specified dimension
  auto GetSpanPair = [&](auto dim, auto dim_size) -> SpanPair {
    if (auto dim_span = selection.FSpan(dim, result_ndim); dim_span.ok()) {
      // Sort selection indices to create
      // disk and memory indices
      auto [disk_ids, mem_ids] = MakeSortedIndices(dim_span.ValueOrDie());
      id_cache.emplace_back(std::move(disk_ids));
      auto disk_span = IndexSpan(id_cache.back());
      id_cache.emplace_back(std::move(mem_ids));
      auto mem_span = IndexSpan(id_cache.back());
      return SpanPair{std::move(disk_span), std::move(mem_span)};
    }
    assert(std::size_t(dim_size) <= monotonic_ids.size());
    auto disk_span = monotonic_ids.subspan(0, dim_size);
    auto mem_span = monotonic_ids.subspan(0, dim_size);
    return SpanPair{std::move(disk_span), std::move(mem_span)};
  };

  // Generate disk and memory spans for an unselected dimension in which
  // only the leading extent elements are read from disk. The remaining
  // elements have negative disk indices and are not read.
  // Span pairs are cached on (dim, extent)
  std::map<std::pair<int, IndexType>, SpanPair> padded_spans;
  auto GetPaddedSpanPair = [&](auto dim, auto dim_size, auto extent) -> SpanPair {
    auto key = std::make_pair(int(dim), IndexType(extent));
    if (auto it = padded_spans.find(key); it != padded_spans.end()) return it->second;
    Index ids(dim_size, -1);
    std::iota(std::begin(ids), std::begin(ids) + extent, 0);
    auto [disk_ids, mem_ids] = MakeSortedIndices(IndexSpan(ids));
    id_cache.emplace_back(std::move(disk_ids));
    auto disk_span = IndexSpan(id_cache.back());
    id_cache.emplace_back(std::move(mem_ids));
    auto mem_span = IndexSpan(id_cache.back());
    return padded_spans[key] = SpanPair{std::move(disk_span), std::move(mem_span)};
  };

  // In the fixed case, create disk and memory spans
  // over each dimension in FORTRAN order
  if (result_shape.IsFixed() && !result_shape.HasCellExtents()) {
    std::vector<SpanPairs> dim_subspans;
    dim_subspans.reserve(result_ndim);
    const auto& shape = result_shape.GetShape();
    auto row_dim = result_ndim - 1;
    for (int d = 0; d < result_ndim; ++d) {
      auto [disk_span, mem_span] = GetSpanPair(d, shape[d]);
      ARROW_ASSIGN_OR_RAISE(auto spans, MakeSubSpans(disk_span, mem_span, d == row_dim));
      dim_subspans.emplace_back(std::move(spans));
    }

    auto [dim_spans, contiguous] = CartesianProduct(dim_subspans);
    ARROW_ASSIGN_OR_RAISE(auto chunks,
                          MakeDataChunks(std::move(dim_spans), std::move(contiguous),
                                         std::move(id_cache), result_shape))
    return DataPartition{std::move(chunks), result_shape.GetDataType()};
  }

  // In the varying case, or if cell extents vary per row,
  // start with the row dimension
  auto nrows = result_shape.nRows();
  auto row_dim = result_ndim - 1;
  auto [row_disk_span, row_mem_span] = GetSpanPair(row_dim, nrows);
  auto fixed_row_shape =
      result_shape.IsFixed() ? result_shape.GetShape().getFirst(row_dim) : IPosition();
  std::vector<SpanPairs> dim_spans;
  std::vector<bool> contiguous;

  for (std::size_t r = 0; r < nrows; ++r) {
    std::vector<SpanPairs> dim_subspans;
    dim_subspans.reserve(result_ndim);
    // The result shape aligns with the row memory indices
    // so we obtain the row shape through indirection
    auto row = row_mem_span[r];
    const auto& row_shape =
        result_shape.IsFixed() ? fixed_row_shape : result_shape.GetRowShape(row);

    // Create span pairs for the secondary dimensions
    for (int dim = 0; dim < row_dim; ++dim) {
      auto [disk_span, mem_span] = [&]() {
        if (result_shape.HasCellExtents()) {
          auto extent = result_shape.GetCellExtent(row)[dim];
          if (extent < row_shape[dim]) {
            return GetPaddedSpanPair(dim, row_shape[dim], extent);
          }
        }
        return GetSpanPair(dim, row_shape[dim]);
      }();
      ARROW_ASSIGN_OR_RAISE(auto spans, MakeSubSpans(disk_span, mem_span, false));
      dim_subspans.emplace_back(std::move(spans));
    }

    // Create span pairs for the row dimension
    dim_subspans.emplace_back(
        SpanPairs{{row_disk_span.subspan(r, 1), row_mem_span.subspan(r, 1)}});

    auto [row_dim_spans, row_contiguous] = CartesianProduct(dim_subspans);
    dim_spans.insert(std::end(dim_spans), std::move_iterator(std::begin(row_dim_spans)),
                     std::move_iterator(std::end(row_dim_spans)));
    contiguous.insert(std::end(contiguous), std::begin(row_contiguous),
                      std::end(row_contiguous));
  }

  if (dim_spans.size() == 0) return Status::Invalid("Data partition produced no chunks!");

  ARROW_ASSIGN_OR_RAISE(auto chunks,
                        MakeDataChunks(std::move(dim_spans), std::move(contiguous),
                                       std::move(id_cache), result_shape));

  return DataPartition{std::move(chunks), result_shape.GetDataType()};
}

}  // namespace detail
}  // namespace arcae
