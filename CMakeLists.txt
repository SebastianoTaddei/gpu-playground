cmake_minimum_required(VERSION 3.14)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/backends")

project(gpu_playground LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build." FORCE)
endif()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(THIRD_PARTY_DIR "${SRC_DIR}/third_party")

include(backend)

file(GLOB TEST_SOURCES
  "${SRC_DIR}/tests/*.cpp"
)

file(GLOB BENCHMARK_SOURCES
  "${SRC_DIR}/benchmarks/*.cpp"
)

foreach(TEST_SRC ${TEST_SOURCES})
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)
  set(TEST_NAME "test_${TEST_NAME}")

  add_executable(${TEST_NAME}
    "${TEST_SRC}"
  )

  set_target_properties(${TEST_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}/tests
  )

  target_include_directories(${TEST_NAME} PRIVATE
    "${SRC_DIR}/include"
  )

  target_link_libraries(${TEST_NAME} PRIVATE
    gpu_playground::backend
  )
endforeach()

foreach(BENCHMARK_SRC ${BENCHMARK_SOURCES})
  get_filename_component(BENCHMARK_NAME "${BENCHMARK_SRC}" NAME_WE)
  set(BENCHMARK_NAME "benchmark_${BENCHMARK_NAME}")

  add_executable(${BENCHMARK_NAME}
    "${BENCHMARK_SRC}"
  )

  set_target_properties(${BENCHMARK_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}/benchmarks
  )

  target_include_directories(${BENCHMARK_NAME} PRIVATE
    "${SRC_DIR}/include"
  )

  target_link_libraries(${BENCHMARK_NAME} PRIVATE
    gpu_playground::backend
  )
endforeach()
