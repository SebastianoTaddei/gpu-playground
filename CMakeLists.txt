cmake_minimum_required(VERSION 3.14)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/backends")

project(gpu_playground LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(THIRD_PARTY_DIR "${SRC_DIR}/third_party")

include(backend)

file(GLOB TEST_SOURCES
  "${SRC_DIR}/tests/*.cpp"
)

foreach(TEST_SRC "${TEST_SOURCES}")
  get_filename_component(TEST_NAME "${TEST_SRC}" NAME_WE)

  add_executable(${TEST_NAME}
    "${TEST_SRC}"
  )

  target_include_directories(${TEST_NAME} PRIVATE
    "${SRC_DIR}/include"
  )

  target_link_libraries(${TEST_NAME} PRIVATE
    gpu_playground::backend
  )
endforeach()
