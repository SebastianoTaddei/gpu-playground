option(GPU_PLAYGROUND_ENABLE_METAL "Enable Metal backend" OFF)

add_library(gpu_playground_backend INTERFACE)
add_library(gpu_playground::backend ALIAS gpu_playground_backend)

message(STATUS "GPU Playground: CPU backend always enabled")
include(CPU)
target_link_libraries(gpu_playground_backend INTERFACE
  cpu_backend
)

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
  target_compile_definitions(gpu_playground_backend INTERFACE
    GPU_PLAYGROUND_HAS_METAL
  )
endif()

