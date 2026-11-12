include(vcpkg_find_fortran)

vcpkg_from_github(
    OUT_SOURCE_PATH src
    REPO "casacore/casacore"
    REF "v3.8.2"
    SHA512 987b78a1556c434b23c0c3d4ddd7b48816cb5a0076879836d4c8f5552940dfe6169636463db78e0a6c72c2290c1f1b4cfd49626a0240f7e4f177ac3e8e0fadd7
    PATCHES
        001-casacore-cmake.patch
        002-json-parser-mutex.patch
        003-dysco-rows-per-block.patch
        004-thread-local-table-cache.patch
        005-refrows-move-constructor.patch
        006-tableproxy-nolock-option.patch
        007-quiet-ms-keyword-logging.patch
        008-disable-rethrow-macro.patch
        009-jsonout-public-put-record.patch
        010-jsonout-round-trip-floats.patch
)

vcpkg_find_acquire_program(FLEX)
vcpkg_find_acquire_program(BISON)
vcpkg_find_fortran(fortran_args)

vcpkg_check_features(
    OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        tablelocking        ENABLE_TABLELOCKING
        deprecated          BUILD_DEPRECATED
        dysco               BUILD_DYSCO
        mutex               USE_THREADS
        readline            USE_READLINE
        adios2              USE_ADIOS2
        fftw3               BUILD_FFTW3
        hdf5                USE_HDF5
        mpi                 USE_MPI
        openmp              USE_OPENMP
        python3             BUILD_PYTHON3
        stacktrace          USE_STACKTRACE
        casabuild           CASA_BUILD)

if(${BUILD_PYTHON3} STREQUAL "ON")
    message(FATAL_ERROR "Python3 Support not available: https://github.com/microsoft/vcpkg/discussions/29645")
endif()

if(${USE_ADIOS2} STREQUAL "ON")
    message(FATAL_ERROR "Adios2 Support not available")
endif()

# This works around vcpkg_fixup_cmake searching for a non-existent ${debug_share} directory
if(NOT DEFINED VCPKG_BUILD_TYPE OR VCPKG_BUILD_TYPE STREQUAL "debug")
    if(NOT EXISTS "${CURRENT_PACKAGES_DIR}/debug/share/${PORT}")
        file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/debug/share/${PORT}")
    endif()
endif()


vcpkg_cmake_configure(
    SOURCE_PATH "${src}"
    OPTIONS
        ${FEATURE_OPTIONS}
        -DBUILD_FFTW3=${BUILD_FFTW3}
        -DBUILD_PYTHON=OFF
        -DBUILD_TESTING=OFF
        -DUSE_PCH=OFF
        -DCMAKE_CXX_STANDARD=20
        # Pass the vcpkg-acquired flex/bison to casacore's own find_package
        # calls. casacore 3.8 requires bison >= 3; without these, its
        # find_package(BISON 3) picks up the system bison (e.g. macOS's 2.3)
        # and configuration fails.
        -DBISON_EXECUTABLE=${BISON}
        -DFLEX_EXECUTABLE=${FLEX}
        ${fortran_args}
)

vcpkg_cmake_install()
vcpkg_fixup_pkgconfig()
vcpkg_cmake_config_fixup()

vcpkg_install_copyright(FILE_LIST "${src}/COPYING")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
