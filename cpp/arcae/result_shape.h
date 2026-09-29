#ifndef ARCAE_RESULT_SHAPE_H
#define ARCAE_RESULT_SHAPE_H

#include <cassert>
#include <memory>
#include <optional>
#include <vector>

#include <arrow/api.h>
#include <arrow/result.h>

#include <absl/types/span.h>

#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/Utilities/DataType.h>
#include <casacore/tables/Tables/TableColumn.h>

#include "arcae/selection.h"
#include "arrow/array/array_base.h"
#include "arrow/status.h"

namespace arcae {
namespace detail {

using RowShapes = std::vector<casacore::IPosition>;

// Holds information about the shape of:
//
// 1. A read from a column
// 2. A write to a column
//
// In the case of reads, this information is either derived
// from a synthesis of:
//
// 1. A selection of column data.
// 2. A result array
//
// In the case of writes, this data is derived from
//
// 1. The result array
//
// This shape may be fixed, in which case `IsFixed` returns true
// and the shape can obtained by calling `GetShape`.
//
// Otherwise the shape varies per row, in which case `IsFixed` return false
// and each individual row shape can be obtained by calling `GetRowShape`.
struct ResultShapeData {
  std::string column_name_;
  std::optional<casacore::IPosition> shape_;
  int ndim_;
  casacore::DataType dtype_;
  std::optional<RowShapes> row_shapes_;
  // Extent of the cell read from disk for each row of a fixed result.
  // Only present when a result exceeds the cells of a variably shaped
  // column in an unselected dimension, and the extents vary by row.
  // The remainder of the result in that dimension is left untouched.
  std::optional<RowShapes> cell_extents_ = std::nullopt;

  // Return the Column Name
  const std::string& GetName() const noexcept { return column_name_; }

  // Return number of dimensions, if they are >= 0
  // else an error status
  arrow::Result<int> ConsistentNDim() const {
    if (ndim_ >= 0) return ndim_;
    return arrow::Status::Invalid("Result shape dimensions are not consistent");
  }

  // Return the Number of Dimensions in the Column
  // Prefer ConsistentNDim for code where row shapes
  // are assumed to have the same rank
  int nDim() const noexcept { return ndim_; }

  // Number of Rows in the Shape
  std::size_t nRows() const noexcept {
    if (IsFixed()) return shape_->last();
    assert(row_shapes_);
    return row_shapes_->size();
  }

  // Maximum dimension size
  std::size_t MaxDimensionSize() const noexcept;

  // Obtain the flat offset at the specified
  arrow::Result<std::size_t> FlatOffset(
      const absl::Span<const IndexType>& index) const noexcept;

  // Is the result shape fixed?
  bool IsFixed() const noexcept { return shape_.has_value(); }

  // Return the shape if it is fixed
  // Requires IsFixed() == true.
  const casacore::IPosition& GetShape() const noexcept {
    assert(IsFixed());
    return shape_.value();
  }

  // Return the shape of the row
  // Requires IsFixed() == false.
  const casacore::IPosition& GetRowShape(std::size_t row) const noexcept {
    assert(!IsFixed());
    assert(row_shapes_);
    assert(row < row_shapes_->size());
    return row_shapes_->operator[](row);
  }

  // Are there per-row cell extents?
  bool HasCellExtents() const noexcept { return cell_extents_.has_value(); }

  // Return the extent of the cell read from disk for the result row
  // Requires HasCellExtents() == true.
  const casacore::IPosition& GetCellExtent(std::size_t row) const noexcept {
    assert(HasCellExtents());
    assert(row < cell_extents_->size());
    return cell_extents_->operator[](row);
  }

  // Get the underlying CASA Data Type
  casacore::DataType GetDataType() const noexcept { return dtype_; }

  // Get the number of elements in the result
  std::size_t nElements() const noexcept;

  // Get an Arrow Array describing the row shapes
  arrow::Result<std::shared_ptr<arrow::Array>> GetShapeArray() const noexcept;

  // Get ListArray offsets
  arrow::Result<std::vector<std::shared_ptr<arrow::Int32Array>>> GetOffsets()
      const noexcept;

  // For any degenerate shapes representing missing rows in the ResultShapeData object
  // convert the associated row in the selection to -1.
  // Where the result exceeds a cell in an unselected dimension,
  // either pad the selection with -1, or set per-row cell extents
  // if the padding varies by row
  arrow::Result<Selection> NegateMissingSelectedRows(const casacore::TableColumn& column,
                                                     const Selection& selection);

  // Create a ResultShapeData instance suitable for reading
  static arrow::Result<ResultShapeData> MakeRead(const casacore::TableColumn& column,
                                                 const Selection& selection = Selection(),
                                                 bool allow_missing_rows = false);

  // Create a ResultShapeData instance from an array
  static arrow::Result<ResultShapeData> FromArray(
      const casacore::TableColumn& column, const std::shared_ptr<arrow::Array>& data);

  // Create a ResultShapeData instance suitable for writing
  static arrow::Result<ResultShapeData> MakeWrite(
      const casacore::TableColumn& column, const std::shared_ptr<arrow::Array>& data,
      const Selection& selection = Selection());
};

}  // namespace detail
}  // namespace arcae

#endif  // ARCAE_RESULT_SHAPE_H
