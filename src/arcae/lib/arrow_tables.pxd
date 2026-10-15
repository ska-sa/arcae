# distutils: language = c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory import shared_ptr
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *

cimport numpy as cnp

cdef extern from "<climits>" nogil:
    cdef unsigned int UINT_MAX

cdef extern from "<absl/types/span.h>" namespace "absl" nogil:
    cdef cppclass Span[T]:
        Span() except +
        Span(T * array, size_t length) except +
        Span(Span&) except +
        size()

cdef extern from "arcae/selection.h" namespace "arcae::detail" nogil:
    ctypedef int64_t IndexType
    ctypedef vector[IndexType] Index
    ctypedef Span[const IndexType] IndexSpan
    cdef cppclass CSelection" arcae::detail::Selection":
        pass

    cdef cppclass CSelectionBuilder" arcae::detail::SelectionBuilder":
        CSelectionBuilder() except +
        CSelectionBuilder & Add[T](vector[T] ids)
        CSelectionBuilder & Add(const IndexSpan & ids)
        CSelectionBuilder & AddEmpty()
        CSelectionBuilder & Order(char order)
        CSelection Build()

cdef extern from "arcae/service_locator.h" namespace "arcae" nogil:
    cdef cppclass CServiceLocator" arcae::ServiceLocator":
        @staticmethod
        CConfiguration & configuration" ServiceLocator::configuration"()

cdef extern from "arcae/configuration.h" namespace "arcae" nogil:
    CStatus SafeMultiThreadedWrites()
    cdef cppclass CConfiguration" arcae::Configuration":
        void Set" Configuration::Set"(const string & key, string value)
        CResult[string] Get" Configuration::Get"(const string & key)
        CResult[bool] Delete" Configuration::Delete"(const string & key)
        vector[string] GetKeys" Configuration::GetKeys"()
        size_t Size" Configuration::Size"()

# The C++ entry points below report errors through arrow::Result.
# except + is a backstop: any C++ exception that still escapes them
# raises a Python exception instead of terminating the process
cdef extern from "arcae/descriptor.h" namespace "arcae" nogil:
    cdef CResult[string] CMSDescriptor" arcae::MSDescriptor"(
        const string & table, bool complete) except +


cdef extern from "arcae/isolated_table_proxy.h" namespace "arcae::detail" nogil:
    cdef struct CInstanceStats" arcae::detail::InstanceStatsSnapshot":
        int64_t submitted
        int64_t completed
        int64_t wait_ns
        int64_t busy_ns
        int64_t io_bytes


cdef extern from "arcae/new_table_proxy.h" namespace "arcae" nogil:
    cdef cppclass CCasaTable" arcae::NewTableProxy":
        @staticmethod
        CResult[bool] Close" NewTableProxy::Close"() except +

        CResult[shared_ptr[CTable]] ToArrow " NewTableProxy::ToArrow"(
            const CSelection & selection,
            const vector[string] & columns) except +
        CResult[shared_ptr[CArray]] GetColumn " NewTableProxy::GetColumn"(
            const string & column,
            const CSelection & selection,
            shared_ptr[CArray] result) except +
        CResult[bool] PutColumn " NewTableProxy::PutColumn"(
            const string & column,
            const shared_ptr[CArray] & data,
            const CSelection & selection) except +
        CResult[shared_ptr[CArray]] GetRowShapes " NewTableProxy::GetRowShapes"(
            const string & column,
            const CSelection & selection) except +
        CResult[string] Name " NewTableProxy::Name"() except +
        CResult[string] GetTableDescriptor " NewTableProxy::GetTableDescriptor"(
            ) except +
        CResult[string] GetColumnDescriptor "NewTableProxy::GetColumnDescriptor"(
            const string & column) except +
        CResult[string] GetDataManagerInfo "NewTableProxy::GetDataManagerInfo"(
            ) except +
        CResult[string] GetLockOptions "NewTableProxy::GetLockOptions"() except +
        CResult[unsigned int] nRows " NewTableProxy::nRows"() except +
        CResult[unsigned int] nColumns " NewTableProxy::nColumns"() except +
        CResult[vector[string]] Columns " NewTableProxy::Columns"() except +
        CResult[bool] AddRows " NewTableProxy::AddRows"(unsigned int nrows) except +
        CResult[bool] AddColumns " NewTableProxy::AddColumns"(
            const string & json_columndescs,
            const string & json_dminfo) except +
        CResult[bool] IsWritable " NewTableProxy::IsWritable"() except +
        CResult[string] GetKeywords " NewTableProxy::GetKeywords"(
            const string & column) except +
        CResult[bool] PutKeywords " NewTableProxy::PutKeywords"(
            const string & json_keywords,
            const string & column) except +
        CResult[bool] RemoveKeyword " NewTableProxy::RemoveKeyword"(
            const string & keyword,
            const string & column) except +
        vector[CInstanceStats] GetInstanceStats " NewTableProxy::GetInstanceStats"(
            ) except +


cdef extern from "arcae/table_factory.h" namespace "arcae" nogil:
    cdef CResult[shared_ptr[CCasaTable]] COpenTable" arcae::OpenTable"(
            const string & filename,
            size_t ninstances,
            bool readonly,
            const string & json_lockoptions,
            const string & json_cache_size) except +
    cdef CResult[shared_ptr[CCasaTable]] CDefaultMS" arcae::DefaultMS"(
            const string & name,
            const string & subtable,
            const string & json_table_desc,
            const string & json_dminfo,
            const string & json_cache_size) except +
    cdef CResult[shared_ptr[CCasaTable]] CCreateTable" arcae::CreateTable"(
            const string & name,
            size_t ninstances,
            const string & json_table_desc,
            const string & json_dminfo,
            size_t nrow,
            const string & json_cache_size) except +
    cdef CResult[shared_ptr[CCasaTable]] CTaql" arcae::Taql"(
            const string & taql,
            const vector[shared_ptr[CCasaTable]] & tables) except +

cdef extern from "merge_sort.cc" namespace "arcae" nogil:
    cdef enum class MergeType "arcae::MergeType":
        INT32
        INT64
        FLOAT32
        FLOAT64

    void PartitionMerge(
        const vector[vector[const void *]] & inputs,
        const vector[size_t] & nrows,
        const vector[MergeType] & types,
        const vector[void *] & outputs)
