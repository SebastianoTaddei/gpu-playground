include(Catch2)

file(GLOB_RECURSE VECTOR_TESTS
  "${CMAKE_CURRENT_SOURCE_DIR}/vector/test_*.cpp"
)

add_executable(test_vector ${VECTOR_TESTS})

target_include_directories(test_vector PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(test_vector PRIVATE gpu_playground Catch2::Catch2WithMain)

catch_discover_tests(test_vector)

file(GLOB_RECURSE MATRIX_TESTS
  "${CMAKE_CURRENT_SOURCE_DIR}/matrix/test_*.cpp"
)

add_executable(test_matrix ${MATRIX_TESTS})

target_include_directories(test_matrix PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(test_matrix PRIVATE gpu_playground Catch2::Catch2WithMain)

catch_discover_tests(test_matrix)

file(GLOB_RECURSE MATRIX_VECTOR_TESTS
  "${CMAKE_CURRENT_SOURCE_DIR}/matrix_vector/test_*.cpp"
)

add_executable(test_matrix_vector ${MATRIX_VECTOR_TESTS})

target_include_directories(test_matrix_vector PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(test_matrix_vector PRIVATE
  gpu_playground Catch2::Catch2WithMain
)

catch_discover_tests(test_matrix_vector)
