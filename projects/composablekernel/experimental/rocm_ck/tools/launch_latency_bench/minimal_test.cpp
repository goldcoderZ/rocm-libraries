// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Minimal test: just load and launch a kpack kernel N times, print timing.
#include <rocm_ck/args.hpp>
#include <rocm_ck/hip_check.hpp>
#include <rocm_ck/kpack_module.hpp>
#include <rocm_ck/kpack_spec_reader.hpp>
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::fprintf(stderr, "Usage: %s <kpack>\n", argv[0]);
        return 1;
    }

    rocm_ck::KpackArchive archive;
    if(!archive.open(argv[1]))
        return 1;

    rocm_ck::KpackKernel kernel;
    if(!kernel.load(archive, "gemm_fp16"))
        return 1;

    auto variants                 = rocm_ck::readVariantSpecs(argv[1]);
    const rocm_ck::GemmSpec* spec = nullptr;
    for(auto& vi : variants)
        if(vi.name == "gemm_fp16")
        {
            spec = &vi.gemm_spec;
            break;
        }
    if(!spec)
        return 1;

    int M = 256, N = 256, K = 256;
    int grid = ((M + spec->block_tile.m - 1) / spec->block_tile.m) *
               ((N + spec->block_tile.n - 1) / spec->block_tile.n);

    void *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, static_cast<size_t>(M) * K * 2));
    HIP_CHECK(hipMalloc(&d_b, static_cast<size_t>(K) * N * 2));
    HIP_CHECK(hipMalloc(&d_c, static_cast<size_t>(M) * N * 2));
    HIP_CHECK(hipMemset(d_a, 0, static_cast<size_t>(M) * K * 2));
    HIP_CHECK(hipMemset(d_b, 0, static_cast<size_t>(K) * N * 2));
    HIP_CHECK(hipMemset(d_c, 0, static_cast<size_t>(M) * N * 2));

    auto [a_sm, a_sk] = rocm_ck::layoutStrides(spec->lhs().layout, M, K);
    auto [b_sk, b_sn] = rocm_ck::layoutStrides(spec->rhs().layout, K, N);
    auto [c_sm, c_sn] = rocm_ck::layoutStrides(spec->output().layout, M, N);

    rocm_ck::Args args{};
    args.tensors[spec->lhs().args_slot] = {
        d_a, rocm_ck::makeShape(M, K), rocm_ck::makeStrides(a_sm, a_sk)};
    args.tensors[spec->rhs().args_slot] = {
        d_b, rocm_ck::makeShape(K, N), rocm_ck::makeStrides(b_sk, b_sn)};
    args.tensors[spec->output().args_slot] = {
        d_c, rocm_ck::makeShape(M, N), rocm_ck::makeStrides(c_sm, c_sn)};

    void* args_ptr   = &args;
    size_t args_size = sizeof(args);
    void* config[]   = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                        args_ptr,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE,
                        &args_size,
                        HIP_LAUNCH_PARAM_END};

    std::printf("Grid: %d  Block: %d\n", grid, spec->workgroup_size);

    // First launch (cold)
    auto t0 = Clock::now();
    HIP_CHECK(hipModuleLaunchKernel(kernel.function(),
                                    grid,
                                    1,
                                    spec->k_batch,
                                    spec->workgroup_size,
                                    1,
                                    1,
                                    0,
                                    nullptr,
                                    nullptr,
                                    config));
    HIP_CHECK(hipDeviceSynchronize());
    auto t1         = Clock::now();
    double first_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::printf("First launch: %.0f us (%.1f ms)\n", first_us, first_us / 1000);

    // 10 iterations
    for(int i = 0; i < 10; ++i)
    {
        auto a = Clock::now();
        HIP_CHECK(hipModuleLaunchKernel(kernel.function(),
                                        grid,
                                        1,
                                        spec->k_batch,
                                        spec->workgroup_size,
                                        1,
                                        1,
                                        0,
                                        nullptr,
                                        nullptr,
                                        config));
        HIP_CHECK(hipDeviceSynchronize());
        auto b    = Clock::now();
        double dt = std::chrono::duration<double, std::micro>(b - a).count();
        std::printf("  iter %d: %.0f us\n", i, dt);
    }

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    return 0;
}
