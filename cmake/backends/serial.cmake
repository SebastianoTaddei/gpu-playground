add_library(serial_backend STATIC
  "${SRC_DIR}/src/backends/serial/serial_device.cpp"
)

target_include_directories(serial_backend PRIVATE
  "${SRC_DIR}/include"
  "${SRC_DIR}/src/backends/serial"
)
