include(Catch2)

file(GLOB_RECURSE VECTOR_BENCHMARKS
  "${CMAKE_CURRENT_SOURCE_DIR}/vector/benchmark_*.cpp"
)

add_executable(benchmark_vector ${VECTOR_BENCHMARKS})

target_include_directories(benchmark_vector PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(benchmark_vector PRIVATE
  gpu_playground Catch2::Catch2WithMain
)

catch_discover_tests(benchmark_vector)

file(GLOB_RECURSE MATRIX_BENCHMARKS
  "${CMAKE_CURRENT_SOURCE_DIR}/matrix/benchmark_*.cpp"
)

add_executable(benchmark_matrix ${MATRIX_BENCHMARKS})

target_include_directories(benchmark_matrix PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(benchmark_matrix PRIVATE
  gpu_playground Catch2::Catch2WithMain
)

catch_discover_tests(benchmark_matrix)

file(GLOB_RECURSE MATRIX_VECTOR_BENCHMARKS
  "${CMAKE_CURRENT_SOURCE_DIR}/matrix_vector/benchmark_*.cpp"
)

add_executable(benchmark_matrix_vector ${MATRIX_VECTOR_BENCHMARKS})

target_include_directories(benchmark_matrix_vector PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(benchmark_matrix_vector PRIVATE
  gpu_playground Catch2::Catch2WithMain
)

catch_discover_tests(benchmark_matrix_vector)
