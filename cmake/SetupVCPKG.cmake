include(FetchContent)

# Read the environment file and extract the sha1 and sha256 sum
file(READ "${CMAKE_SOURCE_DIR}/.env" ENV_FILE)
string(REGEX MATCH "[ \t]*VCPKG_SHA1[ \t]*=[ \t]*([0-9a-f]*)" IGNORED ${ENV_FILE})

if(NOT DEFINED CMAKE_MATCH_1)
    message(FATAL_ERROR "Unable to find VCPKG_SHA1=version in ${CMAKE_SOURCE_DIR}/.env")
endif()

# Download VCPKG at the desired SHA1
set(VCPKG_URL "https://github.com/microsoft/vcpkg/archive/${CMAKE_MATCH_1}.tar.gz")
message("VCPKG_VERSION=${CMAKE_MATCH_1}")
message("VCPKG_URL=${VCPKG_URL}")

string(REGEX MATCH "[ \t]*VCPKG_SHA256[ \t]*=[ \t]*([0-9a-f]*)" IGNORED ${ENV_FILE})

if(NOT DEFINED CMAKE_MATCH_1)
    message(FATAL_ERROR "Unable to find VCPKG_SHA256=version in ${CMAKE_SOURCE_DIR}/.env")
endif()

set(VCPKG_HASH ${CMAKE_MATCH_1})

# Discover or infer the source directory where vcpkg will be decompressed
if(DEFINED ENV{VCPKG_SOURCE_DIR})
    set(VCPKG_SOURCE_DIR $ENV{VCPKG_SOURCE_DIR})
else()
    set(VCPKG_SOURCE_DIR ${CMAKE_SOURCE_DIR}/vcpkg/source)
endif()

# Download and decompress vcpkg
FetchContent_Declare(vcpkg URL ${VCPKG_URL} URL_HASH SHA256=${VCPKG_HASH} SOURCE_DIR ${VCPKG_SOURCE_DIR})
FetchContent_MakeAvailable(vcpkg)

# Discover or infer vcpkg's installation directory
if(DEFINED ENV{VCPKG_INSTALLED_DIR})
    set(VCPKG_INSTALLED_DIR $ENV{VCPKG_INSTALLED_DIR})
else()
    set(VCPKG_INSTALLED_DIR ${CMAKE_SOURCE_DIR}/vcpkg/installed)
endif()

# Set the vcpkg manifest directory and toolchain
set(VCPKG_MANIFEST_DIR ${CMAKE_SOURCE_DIR}/vcpkg/manifest)
set(CMAKE_TOOLCHAIN_FILE
    ${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake
    CACHE FILEPATH "")

set(VCPKG_OVERLAY_TRIPLETS ${CMAKE_SOURCE_DIR}/vcpkg/overlay-triplets)
set(VCPKG_OVERLAY_PORTS ${CMAKE_SOURCE_DIR}/vcpkg/overlay-ports/casacore)

# Discover or infer the vcpkg target triplet
if(DEFINED ENV{VCPKG_TARGET_TRIPLET})
    set(VCPKG_TARGET_TRIPLET $ENV{VCPKG_TARGET_TRIPLET})
else()
    if(UNIX)
        set(VCPKG_TARGET_TRIPLET "x64-linux-cxx20-abi1-rel")
    else()
        message(FATAL_ERROR "Only unix currently supported")
    endif()
endif()

set(VCPKG_LIBDIRS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib")

message("CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
message("VCPKG_SOURCE_DIR: ${VCPKG_SOURCE_DIR}")
message("VCPKG_INSTALLED_DIR: ${VCPKG_INSTALLED_DIR}")
message("VCPKG_MANIFEST_DIR: ${VCPKG_MANIFEST_DIR}")
message("CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
message("VCPKG_OVERLAY_TRIPLETS: ${VCPKG_OVERLAY_TRIPLETS}")
message("VCPKG_OVERLAY_PORTS: ${VCPKG_OVERLAY_PORTS}")
message("VCPKG_TARGET_TRIPLET: ${VCPKG_TARGET_TRIPLET}")
message("VCPKG_LIBDIRS: ${VCPKG_LIBDIRS}")
