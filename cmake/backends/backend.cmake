option(GPU_PLAYGROUND_ENABLE_EIGEN "Enable Eigen backend" OFF)
option(GPU_PLAYGROUND_ENABLE_SIMD "Enable SIMD backend" OFF)
option(GPU_PLAYGROUND_ENABLE_METAL "Enable Metal backend" OFF)

add_library(gpu_playground_backend INTERFACE)
add_library(gpu_playground::backend ALIAS gpu_playground_backend)

message(STATUS "GPU Playground: CPU backend always enabled")
include(CPU)
target_link_libraries(gpu_playground_backend INTERFACE
  cpu_backend
)


if(GPU_PLAYGROUND_ENABLE_EIGEN)
  message(STATUS "GPU Playground: Eigen backend enabled")
  include(Eigen)
  target_link_libraries(gpu_playground_backend INTERFACE
    eigen_backend
  )
endif()

if(GPU_PLAYGROUND_ENABLE_SIMD)
  message(STATUS "GPU Playground: SIMD backend enabled")
  include(SIMD)
  target_link_libraries(gpu_playground_backend INTERFACE
    simd_backend
  )
endif()

if(GPU_PLAYGROUND_ENABLE_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR
      "GPU Playground: Metal backend is only supported on macOS"
    )
  endif()
  message(STATUS "GPU Playground: Metal backend enabled")
  include(Metal)
  target_link_libraries(gpu_playground_backend INTERFACE
    metal_backend
  )
endif()
