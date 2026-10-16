#include "arcae/read_impl.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>

#include <absl/types/span.h>

#include <arrow/api.h>
#include <arrow/array/array_base.h>
#include <arrow/array/array_nested.h>
#include <arrow/buffer.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/future.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/util/logging.h>
#include <arrow/util/thread_pool.h>

#include <casacore/casa/Json.h>
#include <casacore/casa/Utilities/DataType.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <casacore/tables/Tables/TableColumn.h>
#include <casacore/tables/Tables/TableProxy.h>

#include "arcae/data_partition.h"
#include "arcae/isolated_table_proxy.h"
#include "arcae/result_shape.h"
#include "arcae/selection.h"
#include "arcae/service_locator.h"
#include "arcae/table_utils.h"
#include "arcae/type_traits.h"

using ::arrow::Array;
using ::arrow::ArrayData;
using ::arrow::Buffer;
using ::arrow::FixedSizeListArray;
using ::arrow::Future;
using ::arrow::Int32Builder;
using ::arrow::ListArray;
using ::arrow::Result;
using ::arrow::Status;
using ::arrow::StringBuilder;
using ::arrow::Table;

template <class CT>
using CasaArray = ::casacore::Array<CT>;
using ::casacore::ArrayColumn;
using ::casacore::DataType;
using ::casacore::JsonOut;
using ::casacore::Record;
using ::casacore::ScalarColumn;
using ::casacore::TableColumn;
using ::casacore::TableProxy;
using CasaTable = ::casacore::Table;
template <class CT>
using CasaVector = ::casacore::Vector<CT>;
using ::casacore::String;

using ::arcae::detail::DataChunk;

namespace arcae {
namespace detail {

namespace {

static constexpr char kArcaeMetadata[] = "__arcae_metadata__";
static constexpr char kCasaDescriptorKey[] = "__casa_descriptor__";

// Allocators generally require a power of 2 when requesting aligned allocationg
// For example, Arrow can use mimalloc which fails to allocate
// if the alignment is not a power of 2
// https://microsoft.github.io/mimalloc/group__aligned.html
std::size_t GetSafeAlignment(std::size_t type_size) {
  bool is_power_of_two = (type_size & (type_size - 1)) == 0;
  if (type_size == 0 || !is_power_of_two) return arrow::kDefaultBufferAlignment;
  return type_size;
}

// Read a chunk that is contiguous in memory
// directly into its position in the output buffer
template <typename CT>
void ReadContiguousChunk(const TableProxy& tp, const std::string& column_name,
                         const DataChunk& chunk, Buffer& buffer) {
  auto out_ptr = buffer.template mutable_data_as<CT>() + chunk.FlatOffset();
  auto shape = chunk.GetShape();
  if (shape.size() == 1) {
    auto column = ScalarColumn<CT>(tp.table(), column_name);
    auto data = CasaVector<CT>(shape, out_ptr, casacore::SHARE);
    column.getColumnCells(chunk.ReferenceRows(), data);
    return;
  }
  auto column = ArrayColumn<CT>(tp.table(), column_name);
  auto ref_rows = chunk.ReferenceRows();
  if (ref_rows.nrows() == 1) {
    auto data = CasaArray<CT>(chunk.SectionSlicer().length(), out_ptr, casacore::SHARE);
    column.getSlice(ref_rows.firstRow(), chunk.SectionSlicer(), data);
    return;
  }
  auto data = CasaArray<CT>(shape, out_ptr, casacore::SHARE);
  column.getColumnCells(ref_rows, chunk.SectionSlicer(), data);
}

// Read a chunk into a CASA array
template <typename CT>
CasaArray<CT> ReadChunkArray(const TableProxy& tp, const std::string& column_name,
                             const DataChunk& chunk) {
  if (chunk.nDim() == 1) {
    auto column = ScalarColumn<CT>(tp.table(), column_name);
    return column.getColumnCells(chunk.ReferenceRows());
  }
  auto column = ArrayColumn<CT>(tp.table(), column_name);
  auto ref_rows = chunk.ReferenceRows();
  if (ref_rows.nrows() == 1) {
    return column.getSlice(ref_rows.firstRow(), chunk.SectionSlicer())
        .reform(chunk.GetShape());
  }
  return column.getColumnCells(ref_rows, chunk.SectionSlicer());
}

// Transpose a CASA array holding a chunk into its positions in the output buffer
template <typename CT>
void CopyChunkArray(DataChunk chunk, const CasaArray<CT>& data, Buffer& buffer) {
  std::ptrdiff_t ndim = chunk.nDim();
  auto spans = chunk.DimensionSpans();
  auto min_mem = chunk.MinMemIndex();
  auto chunk_strides = chunk.ChunkStrides();
  auto buffer_strides = chunk.BufferStrides();
  const CT* in_ptr = data.data();
  CT* out_ptr = buffer.template mutable_data_as<CT>() + chunk.FlatOffset();
  auto pos = chunk.ScratchPositions();
  for (std::size_t i = 0; i < pos.size(); ++i) pos[i] = 0;

  // Iterate over the spans in memory, copying data
  for (auto done = false; !done;) {
    std::size_t i = 0, o = 0;
    for (std::ptrdiff_t d = 0; d < ndim; ++d) {
      i += pos[d] * chunk_strides[d];
      o += (spans[d].mem[pos[d]] - min_mem[d]) * buffer_strides[d];
    }
    // Moves degrade to copies for simple (i.e. numeric) types
    // but it should make casacore::String more efficient by avoiding copies
    out_ptr[o] = std::move(in_ptr[i]);
    for (std::ptrdiff_t d = 0; d < ndim; ++d) {   // Iterate in FORTRAN order
      if (++pos[d] < spans[d].mem.size()) break;  // Iteration doesn't reach dim end
      pos[d] = 0;                                 // Otherwise reset, next dim
      done = d + 1 == ndim;  // We're done if the last dimension is reset
    }
  }
}

// Read chunks into the output buffer, in the calling thread.
// This must be the isolation thread owning tp
Status ReadChunks(const TableProxy& tp, const std::string& column_name,
                  DataType casa_dtype, absl::Span<const DataChunk> chunks,
                  Buffer& buffer) {
  return DispatchCasaType(casa_dtype, [&](auto cdt) -> Status {
    using CT = typename CasaDataTypeTraits<decltype(cdt)::value>::CasaType;
    for (const auto& chunk : chunks) {
      if (!chunk) return Status::Invalid("Invalid chunk");
      if (chunk.IsEmpty()) continue;
      if (chunk.IsContiguous()) {
        ReadContiguousChunk<CT>(tp, column_name, chunk, buffer);
      } else {
        CopyChunkArray<CT>(chunk, ReadChunkArray<CT>(tp, column_name, chunk), buffer);
      }
      RecordIoBytes(chunk.nElements() * sizeof(CT));
    }
    return Status::OK();
  });
}

// Split chunks into at most ngroups consecutive groups
// with similar numbers of elements, returning the end of each group
std::vector<std::size_t> GroupChunks(const std::vector<DataChunk>& chunks,
                                     std::size_t ngroups) {
  std::size_t nelements = 0;
  for (const auto& chunk : chunks) nelements += chunk.nElements();
  ngroups = std::max<std::size_t>(1, std::min(ngroups, chunks.size()));
  std::vector<std::size_t> ends;
  ends.reserve(ngroups);
  std::size_t cumulative_elements = 0;

  for (std::size_t c = 0; c < chunks.size(); ++c) {
    cumulative_elements += chunks[c].nElements();
    // Close the group once it reaches its share of the elements,
    // leaving at least one chunk for each remaining group
    auto target = nelements * (ends.size() + 1) / ngroups;
    auto remaining_groups = ngroups - ends.size() - 1;
    if (ends.size() + 1 < ngroups &&
        (cumulative_elements >= target || chunks.size() - c - 1 == remaining_groups)) {
      ends.push_back(c + 1);
    }
  }

  ends.push_back(chunks.size());
  return ends;
}

// Extracts the data buffer of the underlying result array
// ensuring it equals nbytes.
Result<std::shared_ptr<Buffer>> GetResultBuffer(const std::shared_ptr<Array>& result,
                                                std::size_t nbytes) {
  if (!result) return Status::Invalid("Result array is null");
  // Extract underlying buffer from result array
  auto tmp = result->data();
  while (true) {
    switch (tmp->type->id()) {
      case arrow::Type::LARGE_LIST:
      case arrow::Type::LIST:
      case arrow::Type::FIXED_SIZE_LIST:
        if (tmp->child_data.size() == 0) return Status::Invalid("No child data");
        tmp = tmp->child_data[0];
        break;
      case arrow::Type::BOOL:
      case arrow::Type::UINT8:
      case arrow::Type::UINT16:
      case arrow::Type::UINT32:
      case arrow::Type::UINT64:
      case arrow::Type::INT8:
      case arrow::Type::INT16:
      case arrow::Type::INT32:
      case arrow::Type::INT64:
      case arrow::Type::FLOAT:
      case arrow::Type::DOUBLE: {
        // The value buffer is in the last position
        // https://arrow.apache.org/docs/format/Columnar.html#fixed-size-primitive-layout
        if (tmp->buffers.size() == 0 || !tmp->buffers[tmp->buffers.size() - 1]) {
          return Status::Invalid("Result array does not contain a buffer");
        }
        auto buffer = tmp->buffers[tmp->buffers.size() - 1];
        if (std::size_t(buffer->size()) != nbytes) {
          return Status::Invalid("Result buffer of ", buffer->size(),
                                 " bytes does not contain the"
                                 " expected number of bytes ",
                                 nbytes);
        }
        return buffer;
      }
      case arrow::Type::STRING:
      default:
        return Status::NotImplemented("Extracting array buffer for type ",
                                      tmp->type->ToString());
    }
  }
}

// Extracts the data buffer of the underlying result array
// ensuring it equals nbytes.
// Otherwise allocates a buffer of nbytes
arrow::Result<std::shared_ptr<Buffer>> GetResultBufferOrAllocate(
    std::size_t nelements, DataType casa_type, const std::shared_ptr<Array>& result) {
  ARROW_ASSIGN_OR_RAISE(auto casa_type_size, CasaDataTypeSize(casa_type));
  auto nbytes = nelements * casa_type_size;
  if (result) return GetResultBuffer(result, nbytes);
  auto alignment = GetSafeAlignment(casa_type_size);
  ARROW_ASSIGN_OR_RAISE(auto allocation, arrow::AllocateBuffer(nbytes, alignment));

  if (IsPrimitiveType(casa_type)) {
    return std::shared_ptr<Buffer>(std::move(allocation));
  } else if (casa_type == DataType::TpString) {
    // We need to use placement new and delete for non-POD types
    // Probably could use some std::is_pod_v<T> strategy,
    // just use if(casa_type == TpString) for now
    for (auto& s : allocation->mutable_span_as<String>()) new (&s) String;
    return std::shared_ptr<Buffer>(std::unique_ptr<Buffer, void (*)(Buffer*)>(
        allocation.release(), [](Buffer* buffer) {
          for (auto& s : buffer->mutable_span_as<String>()) s.String::~String();
          delete buffer;
        }));
  }
  return Status::TypeError("Unhandled CASA type", casa_type);
}

enum ConvertStrategy { FIXED, LIST };

Result<ConvertStrategy> GetConvertStrategy() {
  auto& config = ServiceLocator::configuration();
  auto strategy = config.GetDefault("casa.convert.strategy", "fixed");
  if (auto pos = strategy.find("fixed"); pos != std::string::npos) {
    return ConvertStrategy::FIXED;
  } else if (auto pos = strategy.find("list"); pos != std::string::npos) {
    return ConvertStrategy::LIST;
  }
  return Status::Invalid("Invalid 'casa.convert.strategy=", strategy, "'");
}

// Avoid a long string of templated NumericArray cases by using the
// lower level ArrayData construction mechanism
std::shared_ptr<Array> MakePrimitiveArray(std::shared_ptr<arrow::DataType> dtype,
                                          const std::int64_t N,
                                          std::shared_ptr<Buffer> buffer) {
  auto array_data = std::make_shared<ArrayData>(
      std::move(dtype), N, std::vector<std::shared_ptr<Buffer>>{nullptr, buffer});
  return arrow::MakeArray(array_data);
}

// Given a buffer populated with values
// and an expected result shape,
// create an output array
Result<std::shared_ptr<Array>> MakeArray(const ResultShapeData& result_shape,
                                         const std::shared_ptr<Buffer>& buffer) {
  auto casa_type = result_shape.GetDataType();
  ARROW_ASSIGN_OR_RAISE(auto arrow_dtype, ArrowDataType(casa_type));
  auto nelements = result_shape.nElements();
  std::shared_ptr<arrow::Array> result;
  ARROW_ASSIGN_OR_RAISE(auto strat, GetConvertStrategy());

  if (casa_type == casacore::TpString) {
    // Buffer holds casacore::Strings, convert to arrow strings
    StringBuilder builder;
    auto span = buffer->mutable_span_as<casacore::String>();
    for (std::size_t i = 0; i < span.size(); ++i) {
      ARROW_RETURN_NOT_OK(builder.Append(std::move(span[i])));
    }
    ARROW_ASSIGN_OR_RAISE(result, builder.Finish());
  } else if (casacore::isComplex(casa_type)) {
    // Buffer holds complex values, created a fixed or nested list
    auto base =
        MakePrimitiveArray(std::move(arrow_dtype), 2 * nelements, std::move(buffer));
    if (strat == ConvertStrategy::FIXED) {
      ARROW_ASSIGN_OR_RAISE(result, FixedSizeListArray::FromArrays(base, 2));
    } else if (strat == ConvertStrategy::LIST) {
      arrow::Int32Builder builder;
      ARROW_RETURN_NOT_OK(builder.Reserve(nelements + 1));
      for (std::size_t i = 0; i < nelements + 1; ++i) {
        ARROW_RETURN_NOT_OK(builder.Append(2 * i));
      }
      ARROW_ASSIGN_OR_RAISE(auto offsets, builder.Finish())
      ARROW_ASSIGN_OR_RAISE(result, ListArray::FromArrays(*offsets, *base));
    }
  } else if (IsPrimitiveType(casa_type)) {
    result = MakePrimitiveArray(std::move(arrow_dtype), nelements, std::move(buffer));
  } else {
    return Status::TypeError("Unhandled CASA Type ", casa_type);
  }

  // Introduce shape nesting
  if (result_shape.IsFixed() && strat == ConvertStrategy::FIXED) {
    // Exclude the row dimension
    ARROW_ASSIGN_OR_RAISE(auto ndim, result_shape.ConsistentNDim());
    auto shape = result_shape.GetShape().getFirst(ndim - 1);
    for (auto dim : shape) {
      ARROW_ASSIGN_OR_RAISE(result, FixedSizeListArray::FromArrays(result, dim));
    }
  } else {
    ARROW_ASSIGN_OR_RAISE(auto offsets, result_shape.GetOffsets());
    assert(int(offsets.size()) == result_shape.nDim() - 1);
    for (const auto& offset : offsets) {
      ARROW_ASSIGN_OR_RAISE(result, ListArray::FromArrays(*offset, *result));
    }
  }

  return result;
}

}  // namespace

Future<std::shared_ptr<Array>> ReadImpl(const std::shared_ptr<IsolatedTableProxy>& itp,
                                        const std::string& column,
                                        const Selection& selection,
                                        const std::shared_ptr<Array>& result) {
  // Partition the result into contiguous chunks of data to read from disk.
  // Unless the chunks can be spread across several instances,
  // read them in the same task: each hand-off between threads
  // costs more than reading a small column
  struct PartitionResult {
    std::shared_ptr<DataPartition> partition;
    std::shared_ptr<ResultShapeData> shape;
    // Populated if the chunks were read in the partitioning task
    std::shared_ptr<Array> array;
  };

  auto fan_out = itp->nInstances() > 1;
  // Split the rows of large reads so that they can be spread over the instances.
  // Callers issuing many concurrent requests already keep the instances busy,
  // and splitting their requests only adds hand-offs: requests of a few MB
  // read up to a quarter slower when split into pieces of about 1 MB
  constexpr std::size_t kMinSplitBytes = 8 * 1024 * 1024;
  auto row_split = RowSplitOptions{itp->nInstances(), kMinSplitBytes};

  auto part_fut =
      itp->RunAsync([column = column, selection = selection, result = result, fan_out,
                     row_split](const TableProxy& tp) mutable -> Result<PartitionResult> {
        ARROW_RETURN_NOT_OK(ColumnExists(tp.table(), column));
        auto table_column = TableColumn(tp.table(), column);

        // Get the shape of the result, given the supplied selection.
        // If a result array is supplied we can derive ResultShapeData from it
        // Also iterate over the selected rows and negate any missing rows as
        // this elides reads in these cases.
        // Otherwise the ResultShapeData must be derived wholly from
        // the contents of the column
        auto shape_data = std::make_shared<ResultShapeData>();
        if (result) {
          ARROW_ASSIGN_OR_RAISE(*shape_data,
                                ResultShapeData::FromArray(table_column, result));
          ARROW_ASSIGN_OR_RAISE(
              selection, shape_data->NegateMissingSelectedRows(table_column, selection));
        } else {
          ARROW_ASSIGN_OR_RAISE(*shape_data,
                                ResultShapeData::MakeRead(table_column, selection));
        }

        // Align splits to the column's tiles, read on this thread
        if (row_split.max_splits > 1) {
          row_split.unit_start = MakeRowUnitStart(tp.table(), column);
        }
        ARROW_ASSIGN_OR_RAISE(auto partition,
                              DataPartition::Make(selection, *shape_data, row_split));
        auto shared_partition = std::make_shared<DataPartition>(std::move(partition));
        if (fan_out && shared_partition->nChunks() > 1) {
          return PartitionResult{std::move(shared_partition), std::move(shape_data),
                                 nullptr};
        }

        ARROW_ASSIGN_OR_RAISE(auto buffer, GetResultBufferOrAllocate(
                                               shared_partition->nElements(),
                                               shared_partition->GetDataType(), result));
        ARROW_RETURN_NOT_OK(ReadChunks(tp, column, shared_partition->GetDataType(),
                                       shared_partition->data_chunks_, *buffer));
        ARROW_ASSIGN_OR_RAISE(auto array, MakeArray(*shape_data, buffer));
        return PartitionResult{nullptr, nullptr, std::move(array)};
      });

  // Otherwise spread groups of chunks across the instances,
  // reading each group in a single task, then create an output array
  return part_fut.Then(
      [column = column, itp = itp, result_array = result](
          const PartitionResult& result) -> Future<std::shared_ptr<Array>> {
        if (result.array) return result.array;
        auto casa_dtype = result.partition->GetDataType();
        auto nelements = result.partition->nElements();
        ARROW_ASSIGN_OR_RAISE(
            auto buffer, GetResultBufferOrAllocate(nelements, casa_dtype, result_array));

        auto chunks = std::make_shared<std::vector<DataChunk>>(
            std::move(result.partition->TakeChunks()));
        auto ends = GroupChunks(*chunks, itp->nInstances());
        ARROW_ASSIGN_OR_RAISE(auto dtype_size, CasaDataTypeSize(casa_dtype));
        std::vector<Future<bool>> reads;
        reads.reserve(ends.size());

        for (std::size_t g = 0, begin = 0; g < ends.size(); begin = ends[g++]) {
          std::size_t nbytes = 0;
          for (auto c = begin; c < ends[g]; ++c) {
            nbytes += (*chunks)[c].nElements() * dtype_size;
          }
          reads.push_back(itp->RunAsync(
              [column = column, casa_dtype, chunks = chunks, buffer = buffer,
               begin = begin, end = ends[g]](const TableProxy& tp) -> Result<bool> {
                auto group = absl::MakeConstSpan(*chunks).subspan(begin, end - begin);
                ARROW_RETURN_NOT_OK(ReadChunks(tp, column, casa_dtype, group, *buffer));
                return true;
              },
              std::int64_t(nbytes)));
        }

        // Make the array from the now populated buffer
        return arrow::All(std::move(reads))
            .Then([result_shape = result.shape,
                   buffer = buffer](const std::vector<Result<bool>>& reads)
                      -> Result<std::shared_ptr<Array>> {
              for (const auto& read : reads) ARROW_RETURN_NOT_OK(read);
              return MakeArray(*result_shape, buffer);
            });
      });
}

Future<std::shared_ptr<Table>> ReadTableImpl(
    const std::shared_ptr<IsolatedTableProxy>& itp,
    const std::vector<std::string>& columns, const Selection& selection) {
  if (selection.Size() > 1) {
    return Status::IndexError(
        "Selection along secondary indices is "
        "unsupported when converting to Arrow tables");
  }

  struct TableMetadata {
    Record table_desc;
    std::vector<Record> column_descs;
    std::vector<std::string> columns;
  };

  // Read table metadata
  auto metadata_fut =
      itp->RunAsync([columns = columns](TableProxy& tp) mutable -> Result<TableMetadata> {
        auto table_desc = tp.getTableDescription(true, true);

        if (columns.size() == 0) {
          auto all_columns = tp.columnNames();
          columns = decltype(columns)(std::begin(all_columns), std::end(all_columns));
        }

        std::vector<Record> column_descs;
        column_descs.reserve(columns.size());
        for (const auto& column : columns) {
          ARROW_RETURN_NOT_OK(ColumnExists(tp, column));
          column_descs.push_back(tp.getColumnDescription(column, true));
        }

        return TableMetadata{tp.getTableDescription(true, true), std::move(column_descs),
                             std::move(columns)};
      });

  auto read_columns = metadata_fut.Then(
      [itp = itp, selection = selection](const TableMetadata& table_metadata) {
        std::vector<Future<std::shared_ptr<Array>>> read_futures;
        read_futures.reserve(table_metadata.columns.size());
        for (const auto& column : table_metadata.columns) {
          read_futures.push_back(ReadImpl(itp, column, selection, nullptr));
        }
        return arrow::All(read_futures);
      });

  return read_columns.Then(
      [metadata_fut =
           metadata_fut](const std::vector<Result<std::shared_ptr<Array>>>& array_results)
          -> Result<std::shared_ptr<Table>> {
        if (!metadata_fut.status().ok()) return metadata_fut.status();
        auto table_metadata = metadata_fut.result().ValueOrDie();
        const auto& columns = table_metadata.columns;
        const auto& column_descs = table_metadata.column_descs;
        assert(columns.size() == array_results.size());
        auto fields = arrow::FieldVector();
        auto arrays = arrow::ArrayVector();
        for (std::size_t i = 0; i < array_results.size(); ++i) {
          if (!array_results[i].ok()) continue;
          auto array = array_results[i].ValueOrDie();

          // Attach the CASA column descriptor
          std::ostringstream oss;
          JsonOut column_json(oss);
          column_json.start();
          column_json.write(kCasaDescriptorKey, column_descs[i]);
          column_json.end();
          auto metadata = arrow::KeyValueMetadata::Make({kArcaeMetadata}, {oss.str()});

          auto field = arrow::field(columns[i], array->type(), std::move(metadata));
          fields.emplace_back(std::move(field));
          arrays.emplace_back(std::move(array));
        }
        // Attach the CASA Table descriptor
        std::ostringstream oss;
        JsonOut table_json(oss);
        table_json.start();
        table_json.write(kCasaDescriptorKey, table_metadata.table_desc);
        table_json.end();
        auto metadata = arrow::KeyValueMetadata::Make({kArcaeMetadata}, {oss.str()});
        auto schema = arrow::schema(fields, std::move(metadata));
        return arrow::Table::Make(std::move(schema), std::move(arrays));
      });
}

}  // namespace detail
}  // namespace arcae
