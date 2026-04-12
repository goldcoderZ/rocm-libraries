# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(rocisa)

if(TENSILELITE_ENABLE_AUTOBUILD)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(tensilelite_auto_build)
endif()

if(TENSILELITE_ENABLE_HOST)
    add_library(tensilelite-host OBJECT)
    add_library(tensilelite::tensilelite-host ALIAS tensilelite-host)

    set_target_properties(tensilelite-host
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
    )

    target_compile_features(tensilelite-host PUBLIC cxx_std_17)
    target_compile_definitions(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:TENSILE_DEFAULT_SERIALIZATION>
            $<BUILD_INTERFACE:TENSILE_USE_HIP>
            $<BUILD_INTERFACE:__HIP_HCC_COMPAT_MODE__=1> # if we don't have this we fail tests on 942_304cu
    )

    if(HIPBLASLT_ENABLE_YAML)
        target_link_libraries(tensilelite-host PRIVATE LLVMObjectYAML)
        target_include_directories(tensilelite-host PRIVATE ${LLVM_INCLUDE_DIRS})
        target_compile_definitions(tensilelite-host PUBLIC $<BUILD_INTERFACE:TENSILE_YAML>)
    else()
        target_link_libraries(tensilelite-host PRIVATE ${hipblaslt_msgpack_target})
        target_compile_definitions(tensilelite-host PUBLIC $<BUILD_INTERFACE:TENSILE_MSGPACK>)
    endif()

    target_include_directories(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/include"
    )

    target_link_libraries(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:rocisa::rocisa-cpp>
            roc::origami
        PRIVATE
            hip::device
    )

    add_subdirectory(src)
    add_subdirectory(include)

    if(TENSILELITE_ENABLE_CLIENT AND NOT WIN32)
        add_subdirectory(client)
    endif()

    if((TENSILELITE_BUILD_TESTING OR BUILD_TESTING) AND NOT WIN32)
        add_subdirectory(tests)
    endif()
endif()

# Stage sources for TheRock CI pytest execution via artifact system.
# Layout:
#   shared/origami/                      <- origami (for ../../shared/origami)
#   share/hipblaslt-test-src/            <- hipblaslt cmake root (HIPBLASLT_SOURCE_DIR)
#     CMakeLists.txt, CMakePresets.json, LICENSE.md, cmake/
#     tensilelite/                       <- TENSILELITE_DIR for uv/pytest
#       pyproject.toml, uv.lock, pytest.ini, Tensile/, rocisa/, scripts/
if((TENSILELITE_BUILD_TESTING OR BUILD_TESTING) AND NOT WIN32)
    set(_tl_src "${CMAKE_CURRENT_SOURCE_DIR}")
    cmake_path(SET _hipblaslt_src NORMALIZE "${_tl_src}/..")
    cmake_path(SET _origami_src NORMALIZE "${_hipblaslt_src}/../../shared/origami")
    set(_tl_hipblaslt_dest "${CMAKE_INSTALL_DATADIR}/hipblaslt-test-src")
    set(_tl_dest "${_tl_hipblaslt_dest}/tensilelite")

    # hipblaslt top-level files needed by cmake --preset rocisa
    install(FILES
        "${_hipblaslt_src}/CMakeLists.txt"
        "${_hipblaslt_src}/CMakePresets.json"
        "${_hipblaslt_src}/LICENSE.md"
        DESTINATION "${_tl_hipblaslt_dest}"
    )
    # hipblaslt cmake modules needed by the rocisa preset
    install(FILES
        "${_hipblaslt_src}/cmake/fetch_rocm_cmake.cmake"
        "${_hipblaslt_src}/cmake/hipblaslt_shim.cmake"
        "${_hipblaslt_src}/cmake/hipblaslt_python.cmake"
        "${_hipblaslt_src}/cmake/hipblaslt_target_configure_sanitizers.cmake"
        "${_hipblaslt_src}/cmake/tensilelite_supported_architectures.cmake"
        DESTINATION "${_tl_hipblaslt_dest}/cmake"
    )

    # origami at correct relative offset (../../shared/origami from hipblaslt root)
    install(DIRECTORY "${_origami_src}/"
        DESTINATION "shared/origami"
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "build" EXCLUDE
        PATTERN ".git*" EXCLUDE
        PATTERN "python" EXCLUDE
        PATTERN "tests" EXCLUDE
    )

    # TensileLite Python sources
    set(_tl_files "${_tl_src}/pyproject.toml" "${_tl_src}/pytest.ini")
    if(EXISTS "${_tl_src}/uv.lock")
        list(APPEND _tl_files "${_tl_src}/uv.lock")
    endif()
    install(FILES ${_tl_files} DESTINATION "${_tl_dest}")

    install(DIRECTORY "${_tl_src}/Tensile/"
        DESTINATION "${_tl_dest}/Tensile"
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN ".tox" EXCLUDE
    )
    install(DIRECTORY "${_tl_src}/rocisa/"
        DESTINATION "${_tl_dest}/rocisa"
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "build" EXCLUDE
    )
    install(DIRECTORY "${_tl_src}/scripts/"
        DESTINATION "${_tl_dest}/scripts"
        PATTERN "__pycache__" EXCLUDE
    )
endif()
