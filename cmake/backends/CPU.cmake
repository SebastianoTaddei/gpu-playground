include(Eigen3)

add_library(cpu_backend STATIC
  "${SRC_DIR}/src/backends/cpu/cpu_device.cpp"
)

target_include_directories(cpu_backend PRIVATE
  "${SRC_DIR}/include"
)

target_link_libraries(cpu_backend
    Eigen3::Eigen
)
