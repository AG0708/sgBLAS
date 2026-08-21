cmake_minimum_required(VERSION 3.24)

project(sgblas VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(SGBLAS_ENABLE_CUDA "Build the CUDA SGEMM implementation" OFF)
option(SGBLAS_EXPERIMENTAL_SM80_ASYNC
  "Use the experimental SM80 cp.async SGEMM kernel for aligned full tiles" OFF)
option(SGBLAS_EXPERIMENTAL_SM80_MEDIUM
  "Use a narrower SM80 cp.async tile for medium aligned NN shapes" OFF)
option(SGBLAS_EXPERIMENTAL_SM80_SMALL
  "Use a configurable 64-row SM80 cp.async tile for underfilled aligned NN shapes" OFF)
set(SGBLAS_SM80_MEDIUM_MIN_WIDE_CTAS 128 CACHE STRING
  "Minimum number of equivalent 128x128 CTAs for the SM80 medium kernel")
set(SGBLAS_SM80_MEDIUM_MAX_WIDE_CTAS 2147483647 CACHE STRING
  "Maximum number of equivalent 128x128 CTAs for the SM80 medium kernel")
set(SGBLAS_SM80_MEDIUM_THREAD_ROWS 32 CACHE STRING
  "Logical thread rows in the 256-thread SM80 medium kernel")
option(SGBLAS_SM80_MEDIUM_N_MAJOR_RASTER
  "Raster SM80 medium CTAs across N before M" OFF)
set(SGBLAS_SM80_MEDIUM_L2_PREFETCH_BYTES 0 CACHE STRING
  "cp.async L2 prefetch hint for the SM80 medium kernel: 0, 128, or 256")
set(SGBLAS_SM80_MEDIUM_STAGES 2 CACHE STRING
  "Number of shared-memory pipeline stages in the SM80 medium kernel")
set(SGBLAS_SM80_SMALL_MAX_WIDE_CTAS 128 CACHE STRING
  "Maximum number of equivalent 128x128 CTAs in the first SM80 small-kernel dispatch pocket")
set(SGBLAS_SM80_SMALL_SECOND_MIN_WIDE_CTAS 196 CACHE STRING
  "Minimum equivalent 128x128 CTAs in the second SM80 small-kernel dispatch pocket")
set(SGBLAS_SM80_SMALL_SECOND_MAX_WIDE_CTAS 256 CACHE STRING
  "Maximum equivalent 128x128 CTAs in the second SM80 small-kernel dispatch pocket")
set(SGBLAS_SM80_SMALL_SECOND_MAX_M 2048 CACHE STRING
  "Maximum M dimension in the second SM80 small-kernel dispatch pocket")
set(SGBLAS_SM80_SMALL_SECOND_MAX_N 4096 CACHE STRING
  "Maximum N dimension in the second SM80 small-kernel dispatch pocket")
set(SGBLAS_SM80_SMALL_TILE_COLUMNS 32 CACHE STRING
  "CTA columns in the SM80 small kernel")
set(SGBLAS_SM80_SMALL_THREAD_ROWS 32 CACHE STRING
  "Logical thread rows in the 128-thread SM80 small kernel")
set(SGBLAS_SM80_SMALL_MIN_BLOCKS_PER_SM 5 CACHE STRING
  "Launch-bounds residency target for the SM80 small kernel")
set(SGBLAS_SM80_SMALL_MIN_K 128 CACHE STRING
  "Minimum K dimension dispatched to the SM80 small kernel")
set(SGBLAS_BUILD_TESTS_DEFAULT OFF)
if(PROJECT_IS_TOP_LEVEL)
  set(SGBLAS_BUILD_TESTS_DEFAULT ON)
endif()
option(SGBLAS_BUILD_TESTS "Build sgBLAS tests" ${SGBLAS_BUILD_TESTS_DEFAULT})
option(SGBLAS_BUILD_BENCHMARKS "Build the cuBLAS comparison benchmark" OFF)

add_library(sgblas
  src/handle.cpp
  src/sgemm_validation.cpp
)
add_library(sgblas::sgblas ALIAS sgblas)

target_sources(sgblas
  PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/sgblas/sgblas.h
)
target_compile_features(sgblas PRIVATE cxx_std_17)
target_include_directories(sgblas
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(sgblas PROPERTIES
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)
if(BUILD_SHARED_LIBS)
  set_target_properties(sgblas PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
  )
endif()

if(MSVC)
  target_compile_options(sgblas PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
  target_compile_options(sgblas PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  )
endif()

if(SGBLAS_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "SGBLAS_ENABLE_CUDA=ON, but no CUDA compiler was found. "
      "Configure with -DSGBLAS_ENABLE_CUDA=OFF for a host-only build.")
  endif()

  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  target_sources(sgblas PRIVATE src/sgemm_cuda.cu)
  if(SGBLAS_EXPERIMENTAL_SM80_ASYNC)
    target_sources(sgblas PRIVATE src/sgemm_async_sm80.cu)
    target_compile_definitions(sgblas PRIVATE
      SGBLAS_EXPERIMENTAL_SM80_ASYNC=1
    )
  endif()
  if(SGBLAS_EXPERIMENTAL_SM80_MEDIUM)
    if(NOT SGBLAS_EXPERIMENTAL_SM80_ASYNC)
      message(FATAL_ERROR
        "SGBLAS_EXPERIMENTAL_SM80_MEDIUM requires "
        "SGBLAS_EXPERIMENTAL_SM80_ASYNC=ON")
    endif()
    target_sources(sgblas PRIVATE src/sgemm_async_sm80_medium.cu)
    target_compile_definitions(sgblas PRIVATE
      SGBLAS_EXPERIMENTAL_SM80_MEDIUM=1
      SGBLAS_SM80_MEDIUM_MIN_WIDE_CTAS=${SGBLAS_SM80_MEDIUM_MIN_WIDE_CTAS}
      SGBLAS_SM80_MEDIUM_MAX_WIDE_CTAS=${SGBLAS_SM80_MEDIUM_MAX_WIDE_CTAS}
      SGBLAS_SM80_MEDIUM_THREAD_ROWS=${SGBLAS_SM80_MEDIUM_THREAD_ROWS}
      SGBLAS_SM80_MEDIUM_L2_PREFETCH_BYTES=${SGBLAS_SM80_MEDIUM_L2_PREFETCH_BYTES}
      SGBLAS_SM80_MEDIUM_STAGES=${SGBLAS_SM80_MEDIUM_STAGES}
    )
    if(SGBLAS_SM80_MEDIUM_N_MAJOR_RASTER)
      target_compile_definitions(sgblas PRIVATE
        SGBLAS_SM80_MEDIUM_N_MAJOR_RASTER=1
      )
    endif()
  endif()
  if(SGBLAS_EXPERIMENTAL_SM80_SMALL)
    if(NOT SGBLAS_EXPERIMENTAL_SM80_ASYNC)
      message(FATAL_ERROR
        "SGBLAS_EXPERIMENTAL_SM80_SMALL requires "
        "SGBLAS_EXPERIMENTAL_SM80_ASYNC=ON")
    endif()
    target_sources(sgblas PRIVATE src/sgemm_async_sm80_small.cu)
    target_compile_definitions(sgblas PRIVATE
      SGBLAS_EXPERIMENTAL_SM80_SMALL=1
      SGBLAS_SM80_SMALL_MAX_WIDE_CTAS=${SGBLAS_SM80_SMALL_MAX_WIDE_CTAS}
      SGBLAS_SM80_SMALL_SECOND_MIN_WIDE_CTAS=${SGBLAS_SM80_SMALL_SECOND_MIN_WIDE_CTAS}
      SGBLAS_SM80_SMALL_SECOND_MAX_WIDE_CTAS=${SGBLAS_SM80_SMALL_SECOND_MAX_WIDE_CTAS}
      SGBLAS_SM80_SMALL_SECOND_MAX_M=${SGBLAS_SM80_SMALL_SECOND_MAX_M}
      SGBLAS_SM80_SMALL_SECOND_MAX_N=${SGBLAS_SM80_SMALL_SECOND_MAX_N}
      SGBLAS_SM80_SMALL_TILE_COLUMNS=${SGBLAS_SM80_SMALL_TILE_COLUMNS}
      SGBLAS_SM80_SMALL_THREAD_ROWS=${SGBLAS_SM80_SMALL_THREAD_ROWS}
      SGBLAS_SM80_SMALL_MIN_BLOCKS_PER_SM=${SGBLAS_SM80_SMALL_MIN_BLOCKS_PER_SM}
      SGBLAS_SM80_SMALL_MIN_K=${SGBLAS_SM80_SMALL_MIN_K}
    )
  endif()
  target_link_libraries(sgblas PRIVATE CUDA::cudart)
  target_compile_definitions(sgblas PUBLIC SGBLAS_HAS_CUDA=1)
  target_compile_features(sgblas PRIVATE cuda_std_17)
  target_compile_options(sgblas PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--ptxas-options=-v>
  )
  set_target_properties(sgblas PROPERTIES
    CUDA_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION OFF
  )
else()
  target_sources(sgblas PRIVATE src/sgemm_host.cpp)
endif()

if(SGBLAS_BUILD_TESTS)
  include(CTest)
endif()
if(BUILD_TESTING AND SGBLAS_BUILD_TESTS)
  if(SGBLAS_ENABLE_CUDA)
    add_executable(sgblas_cuda_correctness
      tests/cuda/correctness.cu
      tests/cuda/dispatch_probe.cpp
    )
    if(SGBLAS_EXPERIMENTAL_SM80_ASYNC
       AND NOT SGBLAS_EXPERIMENTAL_SM80_MEDIUM
       AND NOT SGBLAS_EXPERIMENTAL_SM80_SMALL)
      target_compile_definitions(sgblas_cuda_correctness PRIVATE
        SGBLAS_TEST_FROZEN_SM80_WIDE=1)
    endif()
    if(SGBLAS_EXPERIMENTAL_SM80_ASYNC
       AND SGBLAS_EXPERIMENTAL_SM80_MEDIUM
       AND SGBLAS_EXPERIMENTAL_SM80_SMALL
       AND "${SGBLAS_SM80_MEDIUM_MIN_WIDE_CTAS}" STREQUAL "128"
       AND "${SGBLAS_SM80_MEDIUM_MAX_WIDE_CTAS}" STREQUAL "2147483647"
       AND "${SGBLAS_SM80_MEDIUM_THREAD_ROWS}" STREQUAL "32"
       AND NOT SGBLAS_SM80_MEDIUM_N_MAJOR_RASTER
       AND "${SGBLAS_SM80_MEDIUM_L2_PREFETCH_BYTES}" STREQUAL "0"
       AND "${SGBLAS_SM80_MEDIUM_STAGES}" STREQUAL "2"
       AND "${SGBLAS_SM80_SMALL_MAX_WIDE_CTAS}" STREQUAL "128"
       AND "${SGBLAS_SM80_SMALL_SECOND_MIN_WIDE_CTAS}" STREQUAL "196"
       AND "${SGBLAS_SM80_SMALL_SECOND_MAX_WIDE_CTAS}" STREQUAL "256"
       AND "${SGBLAS_SM80_SMALL_SECOND_MAX_M}" STREQUAL "2048"
       AND "${SGBLAS_SM80_SMALL_SECOND_MAX_N}" STREQUAL "4096"
       AND "${SGBLAS_SM80_SMALL_TILE_COLUMNS}" STREQUAL "32"
       AND "${SGBLAS_SM80_SMALL_THREAD_ROWS}" STREQUAL "32"
       AND "${SGBLAS_SM80_SMALL_MIN_BLOCKS_PER_SM}" STREQUAL "5"
       AND "${SGBLAS_SM80_SMALL_MIN_K}" STREQUAL "128")
      target_compile_definitions(sgblas_cuda_correctness PRIVATE
        SGBLAS_TEST_FROZEN_SM80_HYBRID=1)
    endif()
    target_compile_features(sgblas_cuda_correctness PRIVATE cxx_std_17 cuda_std_17)
    target_link_libraries(sgblas_cuda_correctness PRIVATE sgblas CUDA::cudart)
    if(SGBLAS_EXPERIMENTAL_SM80_ASYNC)
      target_compile_definitions(sgblas_cuda_correctness PRIVATE
        SGBLAS_TEST_HAS_SM80_ASYNC=1)
    endif()
    if(SGBLAS_EXPERIMENTAL_SM80_MEDIUM)
      target_compile_definitions(sgblas_cuda_correctness PRIVATE
        SGBLAS_TEST_HAS_SM80_MEDIUM=1
        SGBLAS_TEST_SM80_MEDIUM_STAGES=${SGBLAS_SM80_MEDIUM_STAGES})
    endif()
    if(SGBLAS_EXPERIMENTAL_SM80_SMALL)
      target_compile_definitions(sgblas_cuda_correctness PRIVATE
        SGBLAS_TEST_HAS_SM80_SMALL=1
        SGBLAS_TEST_SM80_SMALL_TILE_COLUMNS=${SGBLAS_SM80_SMALL_TILE_COLUMNS})
    endif()
    set_target_properties(sgblas_cuda_correctness PROPERTIES CUDA_EXTENSIONS OFF)
    add_test(NAME sgblas.cuda.correctness COMMAND sgblas_cuda_correctness)
  else()
    add_executable(sgblas_host_tests tests/host/host_tests.cpp)
    target_compile_features(sgblas_host_tests PRIVATE cxx_std_17)
    target_link_libraries(sgblas_host_tests PRIVATE sgblas)
    add_test(NAME sgblas.host COMMAND sgblas_host_tests)
  endif()
endif()

if(SGBLAS_BUILD_BENCHMARKS)
  if(NOT SGBLAS_ENABLE_CUDA)
    message(FATAL_ERROR "SGBLAS_BUILD_BENCHMARKS requires SGBLAS_ENABLE_CUDA=ON")
  endif()
  add_executable(sgblas_benchmark bench/benchmark.cu)
  target_compile_features(sgblas_benchmark PRIVATE cxx_std_17 cuda_std_17)
  target_link_libraries(sgblas_benchmark PRIVATE
    sgblas
    CUDA::cudart
    CUDA::cublas
  )
  set_target_properties(sgblas_benchmark PROPERTIES CUDA_EXTENSIONS OFF)
endif()

install(TARGETS sgblas
  EXPORT sgblasTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

include(CMakePackageConfigHelpers)

set(SGBLAS_INSTALL_CMAKEDIR
    ${CMAKE_INSTALL_LIBDIR}/cmake/sgblas
    CACHE STRING "Install path for sgBLAS CMake package files")

get_target_property(SGBLAS_LIBRARY_TYPE sgblas TYPE)
set(SGBLAS_CONFIG_NEEDS_CUDATOOLKIT OFF)
if(SGBLAS_ENABLE_CUDA AND SGBLAS_LIBRARY_TYPE STREQUAL "STATIC_LIBRARY")
  set(SGBLAS_CONFIG_NEEDS_CUDATOOLKIT ON)
endif()

configure_package_config_file(
  cmake/sgblasConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/sgblasConfig.cmake
  INSTALL_DESTINATION ${SGBLAS_INSTALL_CMAKEDIR}
)

if(PROJECT_VERSION_MAJOR EQUAL 0)
  set(SGBLAS_VERSION_COMPATIBILITY SameMinorVersion)
else()
  set(SGBLAS_VERSION_COMPATIBILITY SameMajorVersion)
endif()
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/sgblasConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY ${SGBLAS_VERSION_COMPATIBILITY}
)

install(EXPORT sgblasTargets
  FILE sgblasTargets.cmake
  NAMESPACE sgblas::
  DESTINATION ${SGBLAS_INSTALL_CMAKEDIR}
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/sgblasConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/sgblasConfigVersion.cmake
  DESTINATION ${SGBLAS_INSTALL_CMAKEDIR}
)
install(FILES
  LICENSE
  THIRD_PARTY_NOTICES.md
  README.md
  PROVENANCE.md
  SECURITY.md
  DESTINATION ${CMAKE_INSTALL_DOCDIR}
)

if(PROJECT_IS_TOP_LEVEL)
  string(TOLOWER "${CMAKE_SYSTEM_NAME}" SGBLAS_PACKAGE_SYSTEM)
  set(SGBLAS_PACKAGE_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
  if(CMAKE_OSX_ARCHITECTURES)
    set(SGBLAS_PACKAGE_ARCHITECTURE ${CMAKE_OSX_ARCHITECTURES})
  endif()
  string(REPLACE ";" "_" SGBLAS_PACKAGE_ARCHITECTURE
                 "${SGBLAS_PACKAGE_ARCHITECTURE}")

  if(SGBLAS_ENABLE_CUDA)
    set(SGBLAS_PACKAGE_BACKEND cuda)
    string(REPLACE ";" "_sm" SGBLAS_PACKAGE_CUDA_ARCHITECTURES
                   "${CMAKE_CUDA_ARCHITECTURES}")
    string(APPEND SGBLAS_PACKAGE_BACKEND
           "-sm${SGBLAS_PACKAGE_CUDA_ARCHITECTURES}")
  else()
    set(SGBLAS_PACKAGE_BACKEND host)
  endif()

  if(SGBLAS_LIBRARY_TYPE STREQUAL "SHARED_LIBRARY")
    set(SGBLAS_PACKAGE_LINKAGE shared)
  else()
    set(SGBLAS_PACKAGE_LINKAGE static)
  endif()

  set(CPACK_PACKAGE_NAME sgblas)
  set(CPACK_PACKAGE_VENDOR sgBLAS)
  set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
      "A stream-aware, column-major CUDA SGEMM library")
  set(CPACK_RESOURCE_FILE_README ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
  set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)
  set(CPACK_PACKAGE_FILE_NAME
      "sgblas-${PROJECT_VERSION}-${SGBLAS_PACKAGE_SYSTEM}-${SGBLAS_PACKAGE_ARCHITECTURE}-${SGBLAS_PACKAGE_BACKEND}-${SGBLAS_PACKAGE_LINKAGE}")
  set(CPACK_SOURCE_PACKAGE_FILE_NAME "sgblas-${PROJECT_VERSION}")
  set(CPACK_GENERATOR TGZ)
  set(CPACK_SOURCE_GENERATOR TGZ)
  set(CPACK_SOURCE_IGNORE_FILES
    "/\\.git/"
    "/build[^/]*/"
    "/results/"
    "/\\.DS_Store$"
    "/CMakeUserPresets\\.json$"
    "/__pycache__/"
    "\\.py[cod]$"
    "\\.ncu-rep$"
    "\\.nsys-rep$"
  )
  set(CPACK_VERBATIM_VARIABLES YES)
  include(CPack)
endif()
