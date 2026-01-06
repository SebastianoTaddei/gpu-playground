include(Eigen3)

add_library(eigen_backend STATIC
  "${SRC_DIR}/src/backends/eigen/eigen_device.cpp"
)

target_include_directories(eigen_backend PRIVATE
  "${SRC_DIR}/include"
  "${SRC_DIR}/src/backends/eigen"
)

target_link_libraries(eigen_backend
    Eigen3::Eigen
)

target_compile_definitions(eigen_backend PUBLIC
  GPU_PLAYGROUND_HAS_EIGEN
)
if(MSVC)
  # Avoid compilation errors when compiling Eigen3
  target_compile_options(eigen_backend INTERFACE /wd4127 /wd4714)
endif()
