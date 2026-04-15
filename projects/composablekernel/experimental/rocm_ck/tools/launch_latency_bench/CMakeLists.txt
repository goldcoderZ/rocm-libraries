# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Kpack launch latency benchmark. Measures runtime kernel loading and
# launch overhead via hipModuleLaunchKernel.
#
# Usage:
#   cmake -B build -S . -G Ninja \
#       -DCMAKE_PREFIX_PATH=/opt/rocm
#   ninja -C build
#   build/launch_latency_bench <path-to-gemm.kpack> [kernel_name]

cmake_minimum_required(VERSION 3.21)
project(launch_latency_bench LANGUAGES CXX)

find_package(hip REQUIRED)

if(NOT TARGET rocm_kpack)
    add_subdirectory(../../rocm_kpack rocm_kpack_build)
endif()
if(NOT TARGET rocm_ck)
    add_subdirectory(../../include rocm_ck_build)
endif()

add_executable(launch_latency_bench bench.cpp)
target_compile_features(launch_latency_bench PRIVATE cxx_std_20)
target_link_libraries(launch_latency_bench PRIVATE rocm_ck rocm_kpack hip::host)

add_executable(minimal_test minimal_test.cpp)
target_compile_features(minimal_test PRIVATE cxx_std_20)
target_link_libraries(minimal_test PRIVATE rocm_ck rocm_kpack hip::host)
