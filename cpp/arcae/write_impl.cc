#include "arcae/write_impl.h"

#include <cstddef>
#include <memory>
#include <string>

#include <absl/types/span.h>

#include <arrow/api.h>
#include <arrow/array/array_base.h>
#include <arrow/array/array_nested.h>
#include <arrow/buffer.h>
#include <arrow/compute/cast.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/future.h>
#include <arrow/util/logging.h>

#include <casacore/casa/Utilities/DataType.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <casacore/tables/Tables/TableColumn.h>
#include <casacore/tables/Tables/TableProxy.h>

#include "arcae/data_partition.h"
#include "arcae/isolated_table_proxy.h"
#include "arcae/result_shape.h"
#include "arcae/selection.h"
#include "arcae/table_utils.h"
#include "arcae/type_traits.h"

using ::arrow::Array;
using ::arrow::Buffer;
using ::arrow::Future;
using ::arrow::Result;
using ::arrow::Status;

template <class CT>
using CasaArray = ::casacore::Array<CT>;
using ::casacore::ArrayColumn;
using ::casacore::DataType;
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

// Write a chunk that is contiguous in memory
// directly from its position in the input buffer
template <typename CT>
void WriteContiguousChunk(const TableProxy& tp, const std::string& column_name,
                          const DataChunk& chunk, const Buffer& buffer) {
  // casacore Arrays sharing storage require a mutable pointer, but don't write to it
  CT* in_ptr = const_cast<CT*>(buffer.template data_as<CT>()) + chunk.FlatOffset();
  auto shape = chunk.GetShape();
  if (shape.size() == 1) {
    auto column = ScalarColumn<CT>(tp.table(), column_name);
    auto data = CasaVector<CT>(shape, in_ptr, casacore::SHARE);
    column.putColumnCells(chunk.ReferenceRows(), data);
    return;
  }
  auto column = ArrayColumn<CT>(tp.table(), column_name);
  auto ref_rows = chunk.ReferenceRows();
  if (ref_rows.nrows() == 1) {
    auto data = CasaArray<CT>(chunk.SectionSlicer().length(), in_ptr, casacore::SHARE);
    column.putSlice(ref_rows.firstRow(), chunk.SectionSlicer(), data);
    return;
  }
  auto data = CasaArray<CT>(shape, in_ptr, casacore::SHARE);
  column.putColumnCells(ref_rows, chunk.SectionSlicer(), data);
}

// Transpose a chunk from its positions in the input buffer into a CASA array
template <typename CT>
CasaArray<CT> GatherChunkArray(DataChunk chunk, const Buffer& buffer) {
  std::ptrdiff_t ndim = chunk.nDim();
  auto spans = chunk.DimensionSpans();
  auto min_mem = chunk.MinMemIndex();
  auto chunk_strides = chunk.ChunkStrides();
  auto buffer_strides = chunk.BufferStrides();
  // Elements are moved out of the input buffer, see below
  CT* in_ptr = const_cast<CT*>(buffer.template data_as<CT>()) + chunk.FlatOffset();
  assert(chunk.GetShape().product() > 0);
  auto array = CasaArray<CT>(chunk.GetShape());
  CT* out_ptr = array.data();
  auto pos = chunk.ScratchPositions();
  for (std::size_t i = 0; i < pos.size(); ++i) pos[i] = 0;

  // Iterate over the spans in memory, copying data
  for (auto done = false; !done;) {
    std::size_t i = 0, o = 0;
    for (std::ptrdiff_t d = 0; d < ndim; ++d) {
      i += (spans[d].mem[pos[d]] - min_mem[d]) * buffer_strides[d];
      o += pos[d] * chunk_strides[d];
    }
    // Moves degrade to copies for simple (i.e. numeric) types
    // but it should make casacore::String more efficient by avoiding copies.
    // Strings are only moved out of the buffer holding their conversion
    out_ptr[o] = std::move(in_ptr[i]);
    for (std::ptrdiff_t d = 0; d < ndim; ++d) {   // Iterate in FORTRAN order
      if (++pos[d] < spans[d].mem.size()) break;  // Iteration doesn't reach dim end
      pos[d] = 0;                                 // Otherwise reset, next dim
      done = d + 1 == ndim;  // We're done if the last dimension is reset
    }
  }

  return array;
}

// Write a chunk held in a CASA array
template <typename CT>
void WriteChunkArray(const TableProxy& tp, const std::string& column_name,
                     const DataChunk& chunk, const CasaArray<CT>& data) {
  if (chunk.nDim() == 1) {
    auto column = ScalarColumn<CT>(tp.table(), column_name);
    column.putColumnCells(chunk.ReferenceRows(), data);
    return;
  }
  auto column = ArrayColumn<CT>(tp.table(), column_name);
  auto ref_rows = chunk.ReferenceRows();
  if (ref_rows.nrows() == 1) {
    column.putSlice(ref_rows.firstRow(), chunk.SectionSlicer(),
                    data.reform(chunk.SectionSlicer().length()));
    return;
  }
  column.putColumnCells(ref_rows, chunk.SectionSlicer(), data);
}

// Write chunks from the input buffer, in the calling thread.
// This must be the isolation thread owning tp
Status WriteChunks(const TableProxy& tp, const std::string& column_name,
                   DataType casa_dtype, absl::Span<const DataChunk> chunks,
                   const Buffer& buffer) {
  return DispatchCasaType(casa_dtype, [&](auto cdt) -> Status {
    using CT = typename CasaDataTypeTraits<decltype(cdt)::value>::CasaType;
    for (const auto& chunk : chunks) {
      if (!chunk) return Status::Invalid("Invalid chunk");
      if (chunk.IsEmpty()) continue;
      if (chunk.IsContiguous()) {
        WriteContiguousChunk<CT>(tp, column_name, chunk, buffer);
      } else {
        WriteChunkArray<CT>(tp, column_name, chunk, GatherChunkArray<CT>(chunk, buffer));
      }
    }
    return Status::OK();
  });
}

Result<std::shared_ptr<Array>> GetFlatArray(std::shared_ptr<Array> array,
                                            bool nulls = false) {
  auto array_data = array->data();

  while (true) {
    switch (array_data->type->id()) {
      case arrow::Type::LIST:
      case arrow::Type::LARGE_LIST:
      case arrow::Type::FIXED_SIZE_LIST: {
        if (!nulls && array_data->null_count > 0) {
          return Status::Invalid(
              "Null values were encountered "
              "during array flattening.");
        }
        array_data = array_data->child_data[0];
        break;
      }
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
      case arrow::Type::DOUBLE:
      case arrow::Type::STRING:
        return arrow::MakeArray(array_data);
      default:
        return Status::NotImplemented("Flattening of type ", array->type(),
                                      " is not supported");
    }
  }
}

// Attempt to return the underlying buffer of the supplied array
// Arrow strings are converted to casacore Strings
// Otherwise, if the Arrow data type of the array doesn't match the
// associated Arrow data type of the column, a cast is performed
Result<std::shared_ptr<Buffer>> ExtractBufferOrCopyValues(
    const std::shared_ptr<arrow::Array>& flat_array, DataType casa_dtype) {
  if (IsPrimitiveType(casa_dtype)) {
    ARROW_ASSIGN_OR_RAISE(auto arrow_dtype, ArrowDataType(casa_dtype));
    if (arrow_dtype->Equals(flat_array->type())) return flat_array->data()->buffers[1];
    // Need to cast the supplied data to the type appropriate to the CASA column
    ARROW_ASSIGN_OR_RAISE(auto datum, arrow::compute::Cast(flat_array, arrow_dtype));
    return datum.make_array()->data()->buffers[1];
  } else if (casa_dtype == DataType::TpString) {
    // We need to copy arrow strings to casacore strings.
    auto flat_strings = std::dynamic_pointer_cast<arrow::StringArray>(flat_array);
    if (!flat_strings) return Status::Invalid("Cast to StringArray");
    auto nbytes = sizeof(String) * flat_strings->length();
    ARROW_ASSIGN_OR_RAISE(auto allocation, arrow::AllocateBuffer(nbytes));
    auto span = allocation->mutable_span_as<String>();
    assert(span.size() == decltype(span.size())(flat_strings->length()));
    for (std::size_t i = 0; i < span.size(); ++i) {
      // Strings aren't POD, use in-place new on the buffer elements
      auto sv = flat_strings->GetView(i);
      new (&span[i]) String(std::begin(sv), std::end(sv));
    }

    // and in-place delete when releasing the buffer
    return std::shared_ptr<Buffer>(std::unique_ptr<Buffer, void (*)(Buffer*)>(
        allocation.release(), [](Buffer* buffer) {
          for (auto& s : buffer->mutable_span_as<String>()) s.String::~String();
          delete buffer;
        }));
  }

  return Status::TypeError("Unhandled CASA type", casa_dtype);
}

}  // namespace

Future<bool> WriteImpl(const std::shared_ptr<IsolatedTableProxy>& itp,
                       const std::string& column,
                       const std::shared_ptr<arrow::Array>& data,
                       const Selection& selection) {
  // Writes always use a single instance, so derive the shape of the input data,
  // partition it into contiguous chunks and write them in the same task:
  // each hand-off between threads costs more than writing a small column
  return itp->RunAsync([column = column, selection = selection,
                        data = data](TableProxy& tp) -> Result<bool> {
    if (!tp.isWritable()) tp.reopenRW();
    ARROW_RETURN_NOT_OK(ColumnExists(tp.table(), column));
    auto table_column = TableColumn(tp.table(), column);
    ARROW_ASSIGN_OR_RAISE(auto shape_data,
                          ResultShapeData::MakeWrite(table_column, data, selection));
    ARROW_ASSIGN_OR_RAISE(auto partition, DataPartition::Make(selection, shape_data));

    auto casa_dtype = partition.GetDataType();
    auto nelements = partition.nElements();
    if (casacore::isComplex(casa_dtype)) nelements *= 2;
    ARROW_ASSIGN_OR_RAISE(auto flat_array, GetFlatArray(data));
    if (decltype(nelements)(flat_array->length()) != nelements) {
      return Status::Invalid("Partition elements ", nelements,
                             " don't match flat array elements ", flat_array->length());
    }

    ARROW_ASSIGN_OR_RAISE(auto buffer, ExtractBufferOrCopyValues(flat_array, casa_dtype));
    ARROW_RETURN_NOT_OK(
        WriteChunks(tp, column, casa_dtype, partition.data_chunks_, *buffer));
    return true;
  });
}

}  // namespace detail
}  // namespace arcae
