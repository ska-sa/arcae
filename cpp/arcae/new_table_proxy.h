#ifndef ARCAE_NEW_TABLE_PROXY_H
#define ARCAE_NEW_TABLE_PROXY_H

#include <memory>
#include <type_traits>

#include <arrow/result.h>
#include <arrow/util/thread_pool.h>

#include <casacore/tables/Tables/TableProxy.h>

#include "arcae/isolated_table_proxy.h"
#include "arcae/selection.h"
#include "arcae/type_traits.h"

namespace arcae {

class NewTableProxy {
 public:
  // Construct a NewTableProxy with the supplied function
  template <typename Fn, typename = std::enable_if<std::is_same_v<
                             detail::ArrowResultType<Fn>,
                             arrow::Result<std::shared_ptr<casacore::TableProxy>>>>>
  static arrow::Result<std::shared_ptr<NewTableProxy>> Make(Fn&& functor,
                                                            std::size_t ninstances = 1) {
    struct enable_make_shared_ntp : public NewTableProxy {};
    std::shared_ptr<NewTableProxy> ntp = std::make_shared<enable_make_shared_ntp>();
    ARROW_ASSIGN_OR_RAISE(
        ntp->itp_, detail::IsolatedTableProxy::Make(std::move(functor), ninstances));
    return ntp;
  }

  // Spawn a NewTableProxy on the underlying IsolatedTableProxy
  template <typename Fn, typename = std::enable_if<std::is_same_v<
                             detail::ArrowResultType<Fn, const casacore::TableProxy&>,
                             arrow::Result<std::shared_ptr<casacore::TableProxy>>>>>
  arrow::Result<std::shared_ptr<NewTableProxy>> Spawn(Fn&& functor) {
    struct enable_make_shared_ntp : public NewTableProxy {};
    std::shared_ptr<NewTableProxy> ntp = std::make_shared<enable_make_shared_ntp>();
    ARROW_ASSIGN_OR_RAISE(ntp->itp_, itp_->Spawn(std::forward<Fn>(functor)));
    return ntp;
  }

  arrow::Result<std::shared_ptr<arrow::Table>> ToArrow(
      const detail::Selection& selection = {},
      const std::vector<std::string>& columns = {}) const;

  // Get the table descriptor as a JSON string
  arrow::Result<std::string> GetTableDescriptor() const;

  // Get the column descriptor as a JSON string
  arrow::Result<std::string> GetColumnDescriptor(const std::string& column) const;

  // Get the table locking options as a JSON string
  arrow::Result<std::string> GetLockOptions() const;

  // Get the table data manager information as a JSON string
  arrow::Result<std::string> GetDataManagerInfo() const;

  // Get data from the column, possibly guided by
  // a selection along each index, and possibly
  // writing into a provided result array
  arrow::Result<std::shared_ptr<arrow::Array>> GetColumn(
      const std::string& column, const detail::Selection& selection = {},
      const std::shared_ptr<arrow::Array>& result = nullptr) const;

  // Put data into the column from the given array,
  // possibly guided by a selection along each index
  arrow::Result<bool> PutColumn(const std::string& column,
                                const std::shared_ptr<arrow::Array>& data,
                                const detail::Selection& selection = {}) const;

  // Get the row shapes for the given column and row selection
  arrow::Result<std::shared_ptr<arrow::Array>> GetRowShapes(
      const std::string& column, const detail::Selection& selection = {}) const;

  // Return the URL of this table
  arrow::Result<std::string> Name() const;

  // Return the names of the columns in this table
  arrow::Result<std::vector<std::string>> Columns() const;

  // Return the number of columns in this table
  arrow::Result<std::size_t> nColumns() const;

  // Return the number of rows in this table
  arrow::Result<std::size_t> nRows() const;

  // Return true if this table is writable
  arrow::Result<bool> IsWritable() const;

  // Get the table (column empty) or column keywords as a JSON string
  arrow::Result<std::string> GetKeywords(const std::string& column = {}) const;

  // Define table (column empty) or column keywords from a JSON record string.
  // Existing keywords not named in the record are left untouched
  arrow::Result<bool> PutKeywords(const std::string& json_keywords,
                                  const std::string& column = {});

  // Remove a table (column empty) or column keyword
  arrow::Result<bool> RemoveKeyword(const std::string& keyword,
                                    const std::string& column = {});

  // Add rows to the table
  arrow::Result<bool> AddRows(std::size_t nrows);

  // Add a column to the table
  arrow::Result<bool> AddColumns(const std::string& json_columndescs,
                                 const std::string& json_dminfo = {});
  // Get a pointer to the IsolatedTableProxy
  std::shared_ptr<detail::IsolatedTableProxy> Proxy() const { return itp_; }

  // Close the table
  arrow::Result<bool> Close();

 protected:
  // Returns true if multithreaded writes are supported
  // by this NewTableProxy
  arrow::Status SafeMultithreadedWrites() const;

 private:
  std::shared_ptr<detail::IsolatedTableProxy> itp_;
};

}  // namespace arcae

#endif  // ARCAE_NEW_TABLE_PROXY_H
