cmake_minimum_required(VERSION 3.14)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(metal_tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_OSX_ARCHITECTURES "arm64")

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SHADERS_DIR "${SRC_DIR}/shaders")
set(METAL_TESTS_THIRD_PARTY_DIR "${SRC_DIR}/third_party")

include(Eigen3)

add_custom_command(
    OUTPUT ${BUILD_DIR}/ax_minus_b.air
    COMMAND xcrun -sdk macosx metal
            -std=metal3.0
            -c ${SHADERS_DIR}/ax_minus_b.metal
            -o ${BUILD_DIR}/ax_minus_b.air
    DEPENDS ${SHADERS_DIR}/ax_minus_b.metal
)

add_custom_command(
    OUTPUT ${BUILD_DIR}/ax_minus_b.metallib
    COMMAND xcrun -sdk macosx metallib
            ${BUILD_DIR}/ax_minus_b.air
            -o ${BUILD_DIR}/ax_minus_b.metallib
    DEPENDS ${BUILD_DIR}/ax_minus_b.air
)

add_executable(ax_minus_b
  ${SRC_DIR}/tests/ax_minus_b.cpp
  ${BUILD_DIR}/ax_minus_b.metallib
  ${SRC_DIR}/metal/metal.cpp
)

target_include_directories(ax_minus_b PUBLIC
  ${SRC_DIR}
)

target_link_libraries(ax_minus_b
    "-framework Metal"
    "-framework Foundation"
    Eigen3::Eigen
)
