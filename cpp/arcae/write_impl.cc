#include "arcae/write_impl.h"

#include <cstddef>
#include <memory>
#include <string>

#include <arrow/api.h>
#include <arrow/array/array_base.h>
#include <arrow/array/array_nested.h>
#include <arrow/buffer.h>
#include <arrow/compute/cast.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/async_generator.h>
#include <arrow/util/future.h>
#include <arrow/util/logging.h>
#include <arrow/util/thread_pool.h>

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
using ::arrow::CallbackOptions;
using ::arrow::CollectAsyncGenerator;
using ::arrow::Future;
using ::arrow::MakeMappedGenerator;
using ::arrow::MakeVectorGenerator;
using ::arrow::Result;
using ::arrow::ShouldSchedule;
using ::arrow::Status;
using ::arrow::internal::GetCpuThreadPool;

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

// Functor implementing dispatch of disk writing functionality
struct WriteCallback {
  // Column name
  std::string column;
  // Isolated TableProxy
  std::shared_ptr<IsolatedTableProxy> itp;
  // Output buffer
  std::shared_ptr<Buffer> buffer;

  // Write data, dependent on chunk characteristics and CASA data type
  template <DataType CDT>
  Future<bool> DoWrite(const DataChunk& chunk) const {
    using CT = typename CasaDataTypeTraits<CDT>::CasaType;

    // If the chunk is contiguous in memory, we can write
    // from a CASA Array view over that position in the buffer
    if (chunk.IsContiguous()) {
      return itp->RunAsync([column_name = std::move(column), chunk = chunk,
                            buffer = buffer](const TableProxy& tp) -> bool {
        CT* in_ptr = const_cast<CT*>(buffer->template data_as<CT>());
        in_ptr += chunk.FlatOffset();
        auto shape = chunk.GetShape();
        if (shape.size() == 1) {
          auto column = ScalarColumn<CT>(tp.table(), column_name);
          auto vector = CasaVector<CT>(shape, in_ptr, casacore::SHARE);
          column.putColumnCells(chunk.ReferenceRows(), vector);
          return true;
        }
        auto column = ArrayColumn<CT>(tp.table(), column_name);
        auto ref_rows = chunk.ReferenceRows();
        if (ref_rows.nrows() == 1) {
          auto array = CasaArray<CT>(chunk.SectionSlicer().length(), in_ptr, casacore::SHARE);
          column.putSlice(ref_rows.firstRow(), chunk.SectionSlicer(), array);
          return true;
        }
        auto array = CasaArray<CT>(shape, in_ptr, casacore::SHARE);
        column.putColumnCells(ref_rows, chunk.SectionSlicer(), array);
        return true;
      });
    }

    // Transpose the array into the output buffer
    auto transpose_fut = arrow::DeferNotOk(GetCpuThreadPool()->Submit(
        [chunk = chunk, buffer = buffer]() mutable -> CasaArray<CT> {
          std::ptrdiff_t ndim = chunk.nDim();
          auto spans = chunk.DimensionSpans();
          auto min_mem = chunk.MinMemIndex();
          auto chunk_strides = chunk.ChunkStrides();
          auto buffer_strides = chunk.BufferStrides();
          assert(buffer->template mutable_data_as<CT>() != nullptr);
          CT* in_ptr = const_cast<CT*>(buffer->template data_as<CT>());
          in_ptr += chunk.FlatOffset();
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
            // but it should make casacore::String more efficient by avoiding copies
            out_ptr[o] = std::move(in_ptr[i]);
            for (std::ptrdiff_t d = 0; d < ndim; ++d) {  // Iterate in FORTRAN order
              if (++pos[d] < spans[d].mem.size())
                break;               // Iteration doesn't reach dim end
              pos[d] = 0;            // Otherwise reset, next dim
              done = d + 1 == ndim;  // We're done if the last dimension is reset
            }
          }

          return array;
        }));

    return itp->Then(transpose_fut,
                     [column_name = std::move(column), chunk = chunk](
                         const CasaArray<CT>& data, const TableProxy& tp) -> bool {
                       if (chunk.nDim() == 1) {
                         auto column = ScalarColumn<CT>(tp.table(), column_name);
                         column.putColumnCells(chunk.ReferenceRows(), data);
                         return true;
                       }
                       auto column = ArrayColumn<CT>(tp.table(), column_name);
                       auto ref_rows = chunk.ReferenceRows();
                       if (ref_rows.nrows() == 1) {
                         column.putSlice(ref_rows.firstRow(), chunk.SectionSlicer(),
                                         data.reform(chunk.SectionSlicer().length()));
                         return true;
                       }
                       column.putColumnCells(ref_rows, chunk.SectionSlicer(),
                                             data);
                       return true;
                     });
  }

  // Write a chunk of data from the encapsulated buffer
  // returning true on success
  Future<bool> operator()(const DataChunk& chunk) const {
    if (!chunk) return Status::Invalid("Invalid chunk");
    if (chunk.IsEmpty()) return true;
    switch (chunk.CasaDataType()) {
      case DataType::TpBool:
        return DoWrite<DataType::TpBool>(chunk);
        break;
      case DataType::TpChar:
        return DoWrite<DataType::TpChar>(chunk);
        break;
      case DataType::TpUChar:
        return DoWrite<DataType::TpUChar>(chunk);
        break;
      case DataType::TpShort:
        return DoWrite<DataType::TpShort>(chunk);
        break;
      case DataType::TpUShort:
        return DoWrite<DataType::TpUShort>(chunk);
        break;
      case DataType::TpInt:
        return DoWrite<DataType::TpInt>(chunk);
        break;
      case DataType::TpUInt:
        return DoWrite<DataType::TpUInt>(chunk);
        break;
      case DataType::TpInt64:
        return DoWrite<DataType::TpInt64>(chunk);
        break;
      case DataType::TpFloat:
        return DoWrite<DataType::TpFloat>(chunk);
        break;
      case DataType::TpDouble:
        return DoWrite<DataType::TpDouble>(chunk);
        break;
      case DataType::TpComplex:
        return DoWrite<DataType::TpComplex>(chunk);
        break;
      case DataType::TpDComplex:
        return DoWrite<DataType::TpDComplex>(chunk);
        break;
      case DataType::TpString:
        return DoWrite<DataType::TpString>(chunk);
        break;
      default:
        return Status::NotImplemented("Column ", column, " with data type ",
                                      chunk.CasaDataType());
    }
  }
};

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
  // Get the shape of the input data, given the supplied selection
  // and the given result array
  struct ShapeResult {
    std::shared_ptr<ResultShapeData> shape;
    std::shared_ptr<Selection> selection;
  };

  auto shape_fut = itp->RunAsync([column = column, selection = selection, data = data](
                                     TableProxy& tp) mutable -> Result<ShapeResult> {
    if (!tp.isWritable()) tp.reopenRW();
    ARROW_RETURN_NOT_OK(ColumnExists(tp.table(), column));
    auto table_column = TableColumn(tp.table(), column);
    ARROW_ASSIGN_OR_RAISE(auto shape_data,
                          ResultShapeData::MakeWrite(table_column, data, selection));
    return ShapeResult{std::make_shared<ResultShapeData>(std::move(shape_data)),
                       std::make_shared<Selection>(std::move(selection))};
  });

  // Partition the resulting shape into contiguous chunks
  // of data to write to disk
  struct PartitionResult {
    std::shared_ptr<DataPartition> partition;
    std::shared_ptr<ResultShapeData> shape;
  };
  auto part_fut = shape_fut.Then(
      [](const ShapeResult& result) mutable -> Result<PartitionResult> {
        ARROW_ASSIGN_OR_RAISE(auto partition,
                              DataPartition::Make(*result.selection, *result.shape));
        return PartitionResult{std::make_shared<DataPartition>(std::move(partition)),
                               std::move(result.shape)};
      },
      {}, CallbackOptions{ShouldSchedule::Always, GetCpuThreadPool()});

  // Write each contiguous chunk of data on disk independently
  // then create an output array
  auto write_fut = part_fut.Then([column = column, itp = itp,
                                  data = data](const PartitionResult& result) mutable
                                 -> Future<bool> {
    auto casa_dtype = result.partition->GetDataType();
    auto nelements = result.partition->nElements();
    if (casacore::isComplex(casa_dtype)) nelements *= 2;
    ARROW_ASSIGN_OR_RAISE(auto flat_array, GetFlatArray(data));
    if (decltype(nelements)(flat_array->length()) != nelements) {
      return Status::Invalid("Partition elements ", nelements,
                             " don't match flat array elements ", flat_array->length());
    }

    ARROW_ASSIGN_OR_RAISE(auto buffer, ExtractBufferOrCopyValues(flat_array, casa_dtype));

    // Make an async generator over the data chunks
    // Map WriteCallback over all data chunks
    auto data_chunk_gen = MakeVectorGenerator(std::move(result.partition->data_chunks_));
    auto copy_and_write_gen = MakeMappedGenerator(
        std::move(data_chunk_gen), WriteCallback{std::move(column), itp, buffer});

    // Collect the write results
    auto collect = CollectAsyncGenerator(std::move(copy_and_write_gen));
    return collect.Then(
        [](const std::vector<bool>& result) -> Future<bool> {
          for (auto b : result) {
            if (!b) return Status::Invalid("Write failed");
          }
          return true;
        },
        {}, CallbackOptions{ShouldSchedule::Always, GetCpuThreadPool()});
  });

  return write_fut;
}

}  // namespace detail
}  // namespace arcae
