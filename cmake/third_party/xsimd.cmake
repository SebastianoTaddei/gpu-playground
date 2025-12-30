set(XSIMD_REQUIRED_VERSION 14.0.0)

list(APPEND CMAKE_PREFIX_PATH "${THIRD_PARTY_DIR}")
find_package(
  xsimd
  ${XSIMD_REQUIRED_VERSION}
  NO_MODULE
  QUIET
)

if(NOT TARGET xsimd)
  message(STATUS
    "GPU Playground: "
    "Did not find xsimd ${XSIMD_REQUIRED_VERSION} installed, "
    "downloading to ${THIRD_PARTY_DIR}"
  )
  include(FetchContent)

  set(FETCHCONTENT_BASE_DIR "${THIRD_PARTY_DIR}")
  fetchcontent_declare(
      xsimd
      GIT_REPOSITORY "https://github.com/xtensor-stack/xsimd"
      GIT_TAG ${XSIMD_REQUIRED_VERSION}
  )

  fetchcontent_makeavailable(xsimd)
else()
  get_target_property(XSIMD_INCLUDE_DIRS
    xsimd
    INTERFACE_INCLUDE_DIRECTORIES
  )
  message(STATUS
    "GPU Playground: Found xsimd installed in ${XSIMD_INCLUDE_DIRS}"
  )
endif()
