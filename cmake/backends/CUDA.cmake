enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75 86 89 120)

add_library(cuda_backend STATIC
  "${SRC_DIR}/src/backends/cuda/cuda_device.cu"
)

target_include_directories(cuda_backend PRIVATE
  "${SRC_DIR}/include"
  "${SRC_DIR}/src/backends/cuda"
  "${SRC_DIR}/src/backends/cuda/kernels"
)

set_target_properties(cuda_backend PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
)

target_compile_definitions(cuda_backend PUBLIC
  GPU_PLAYGROUND_HAS_CUDA
)
