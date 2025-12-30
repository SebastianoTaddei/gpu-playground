set(SHADERS_DIR "${SRC_DIR}/src/backends/metal/shaders")
set(MTL_LIB_DIR "${BUILD_DIR}/src/backends/metal/shaders")

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

add_library(metal_backend STATIC
  "${SRC_DIR}/src/backends/metal/metal_device.mm"
)

target_include_directories(metal_backend PRIVATE
  "${SRC_DIR}/include"
  "${SRC_DIR}/src/backends/metal"
)

add_metal_library(metal_backend
  "${SHADERS_DIR}"
  "${MTL_LIB_DIR}"
  metal_backend
)

target_link_libraries(metal_backend
  "-framework Metal"
  "-framework Foundation"
)

target_compile_definitions(metal_backend PUBLIC
  METAL_LIB="${MTL_LIB_DIR}/metal_backend.metallib"
  GPU_PLAYGROUND_HAS_METAL
)
