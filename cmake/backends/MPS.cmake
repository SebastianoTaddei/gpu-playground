set(SHADERS_DIR "${SRC_DIR}/src/backends/metal/shaders")
set(MPS_LIB_DIR "${BUILD_DIR}/src/backends/mps/shaders")

enable_language(OBJCXX)

function(add_metal_library TARGET SHADER_DIR OUTPUT_DIR METALLIB_NAME)
  file(GLOB METAL_SOURCES "${SHADER_DIR}/*.metal")

  set(AIR_FILES)

  foreach(SHADER ${METAL_SOURCES})
    get_filename_component(NAME ${SHADER} NAME_WE)
    set(AIR "${OUTPUT_DIR}/${NAME}.air")

    add_custom_command(
            OUTPUT ${AIR}
            COMMAND xcrun -sdk macosx metal
                    -std=metal3.0
                    -c ${SHADER}
                    -o ${AIR}
            DEPENDS ${SHADER}
        )

    list(APPEND AIR_FILES ${AIR})
  endforeach()

  set(METALLIB "${OUTPUT_DIR}/${METALLIB_NAME}.metallib")

  add_custom_command(
        OUTPUT ${METALLIB}
        COMMAND xcrun -sdk macosx metallib
                ${AIR_FILES}
                -o ${METALLIB}
        DEPENDS ${AIR_FILES}
    )

  add_custom_target(${TARGET}_metal_lib DEPENDS ${METALLIB})
  add_dependencies(${TARGET} ${TARGET}_metal_lib)

  set(${METALLIB_NAME}_PATH ${METALLIB} PARENT_SCOPE)
endfunction()

add_library(mps_backend STATIC
  "${SRC_DIR}/src/backends/mps/mps_device.mm"
)

target_include_directories(mps_backend PRIVATE
  "${SRC_DIR}/include"
  "${SRC_DIR}/src/backends/mps"
)

add_metal_library(mps_backend
  "${SHADERS_DIR}"
  "${MPS_LIB_DIR}"
  mps_backend
)

target_link_libraries(mps_backend
  "-framework Metal"
  "-framework MetalPerformanceShaders"
  "-framework Foundation"
)

target_compile_definitions(mps_backend PUBLIC
  MPS_LIB="${MPS_LIB_DIR}/mps_backend.metallib"
  GPU_PLAYGROUND_HAS_MPS
)
