include(xsimd)

add_library(simd_backend STATIC
  "${SRC_DIR}/src/backends/simd/simd_device.cpp"
)

target_include_directories(simd_backend PRIVATE
  "${SRC_DIR}/include"
  "${SRC_DIR}/src/backends/simd"
)

target_link_libraries(simd_backend
    xsimd
)

target_compile_definitions(simd_backend PUBLIC
  GPU_PLAYGROUND_HAS_SIMD
)
