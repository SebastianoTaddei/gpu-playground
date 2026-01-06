cmake_minimum_required(VERSION 3.14)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/backends")

project(gpu_playground LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

if(NOT CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/lib")
  message(STATUS
    "BSplineX: "
    "CMAKE_INSTALL_PREFIX not specified, "
    "defaulting to ${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build." FORCE)
endif()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(THIRD_PARTY_DIR "${SRC_DIR}/third_party")

include(backend)

add_library(gpu_playground INTERFACE)
add_library(gpu_playground::gpu_playground ALIAS gpu_playground)

target_link_libraries(gpu_playground INTERFACE
  gpu_playground::backend
)

target_include_directories(gpu_playground INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  option(BUILD_TESTS "Build tests" OFF)
  option(BUILD_BENCHMARKS "Build benchmarks" OFF)

  if(BUILD_TESTS)
    string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}"
    )
    add_subdirectory(${SRC_DIR}/tests)
  endif()

  if(BUILD_BENCHMARKS)
    add_subdirectory(${SRC_DIR}/benchmarks)
  endif()
endif()

if(MSVC)
  target_compile_options(gpu_playground INTERFACE /W4 /WX)
else()
  target_compile_options(gpu_playground INTERFACE
    -Wall -Wextra -pedantic -Werror
  )
endif()
