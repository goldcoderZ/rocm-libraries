# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# StaticOrdering Heuristic Plugin
# RFC 0007: Well-known policy that wraps utilities::sortEngineIds

project(hipdnn_heuristic_static_ordering VERSION 1.0.0 LANGUAGES CXX)

# Create the plugin as a shared library
add_library(hipdnn_heuristic_static_ordering SHARED
    StaticOrderingPlugin.cpp
)

# Set output name to match expected plugin naming convention
set_target_properties(hipdnn_heuristic_static_ordering PROPERTIES
    OUTPUT_NAME "hipdnn_heuristic_static_ordering"
    PREFIX "lib"
    SUFFIX ".so"
    CXX_VISIBILITY_PRESET hidden
)

# Include directories
target_include_directories(hipdnn_heuristic_static_ordering PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against hipdnn_plugin_sdk (for C ABI types) and hipdnn_data_sdk (for utilities)
target_link_libraries(hipdnn_heuristic_static_ordering PRIVATE
    hipdnn_plugin_sdk
    hipdnn_data_sdk
)

# Compiler options
target_compile_options(hipdnn_heuristic_static_ordering PRIVATE
    ${HIPDNN_WARNING_COMPILE_OPTIONS}
)

# Export symbols for the C ABI
if(NOT WIN32)
    # On Linux, export only the C ABI symbols
    target_link_options(hipdnn_heuristic_static_ordering PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
    )
endif()

# Installation
install(TARGETS hipdnn_heuristic_static_ordering
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/hipdnn_plugins/heuristics
    COMPONENT Runtime
)

# Optionally install to build directory for testing
if(NOT CMAKE_INSTALL_PREFIX STREQUAL CMAKE_BINARY_DIR)
    add_custom_command(TARGET hipdnn_heuristic_static_ordering POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/hipdnn_plugins/heuristics
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:hipdnn_heuristic_static_ordering>
            ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/hipdnn_plugins/heuristics/
        COMMENT "Copying StaticOrdering plugin to build directory for testing"
    )
endif()
