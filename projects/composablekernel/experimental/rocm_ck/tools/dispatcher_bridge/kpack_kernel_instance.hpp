// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// KpackKernelInstance — CK dispatcher backend that loads kernels from kpack archives.
//
// Implements the dispatcher's KernelInstance interface. Translates between the
// dispatcher's (a_ptr, b_ptr, c_ptr, Problem) calling convention and rocm_ck's
// slot-based Args struct, using the GemmSpec's physical tensor table for mapping.

#pragma once

#include <rocm_ck/args.hpp>
#include <rocm_ck/datatype_convert.hpp>
#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/gemm_spec.hpp>
#include <rocm_ck/hip_check.hpp>
#include <rocm_ck/kpack_module.hpp>
#include <rocm_ck/layout.hpp>
#include <rocm_ck/projection.hpp>
#include <rocm_ck/verify.hpp>

#include <ck_tile/dispatcher/kernel_instance.hpp>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace rocm_ck {

/// Dispatcher backend that loads GEMM kernels from self-describing kpack archives.
///
/// Each instance wraps a single kernel variant. The archive is shared across
/// all variants from the same kpack file.
class KpackKernelInstance : public ck_tile::dispatcher::KernelInstance
{
    public:
    KpackKernelInstance(ck_tile::dispatcher::KernelKey key,
                        std::string name,
                        GemmSpec spec,
                        std::shared_ptr<KpackArchive> archive)
        : key_(std::move(key)), name_(std::move(name)), spec_(spec), archive_(std::move(archive))
    {
    }

    const ck_tile::dispatcher::KernelKey& get_key() const override { return key_; }

    std::string get_name() const override { return name_; }

    bool supports(const ck_tile::dispatcher::Problem& problem) const override
    {
        if(!spec_.pad_m && problem.M % spec_.block_tile.m != 0)
            return false;
        if(!spec_.pad_n && problem.N % spec_.block_tile.n != 0)
            return false;
        // K must always be tile-aligned (CK Tile constraint — pad_k only
        // controls vector load width, does not mask the K-tail)
        if(problem.K % spec_.block_tile.k != 0)
            return false;
        // Split-K: kernel's k_batch must match problem's k_batch
        if(problem.k_batch != spec_.k_batch)
            return false;
        return true;
    }

    float run(const void* a_ptr,
              const void* b_ptr,
              void* c_ptr,
              const void** d_ptrs,
              const ck_tile::dispatcher::Problem& problem,
              void* stream) const override
    {
        // D tensor fusion not wired through Args yet — reject any kernel that
        // expects D tensors, regardless of whether the caller provided pointers.
        if(spec_.numDTensors() > 0)
            throw std::runtime_error("KpackKernelInstance: kernel '" + name_ + "' expects " +
                                     std::to_string(spec_.numDTensors()) +
                                     " D tensor(s) — D tensor fusion not implemented yet");
        (void)d_ptrs;

        // Thread-safe lazy kernel loading
        std::call_once(load_flag_, [this]() {
            kernel_.emplace();
            if(!kernel_->load(*archive_, name_.c_str()))
                throw std::runtime_error("Failed to load kernel '" + name_ + "' from kpack");
        });

        hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);

        // Build rocm_ck Args from dispatcher arguments + GemmSpec tensor table
        Args kernel_args{};

        auto [a_sm, a_sk] = layoutStrides(spec_.lhs().layout, problem.M, problem.K);
        auto [b_sk, b_sn] = layoutStrides(spec_.rhs().layout, problem.K, problem.N);
        auto [c_sm, c_sn] = layoutStrides(spec_.output().layout, problem.M, problem.N);

        kernel_args.tensors[spec_.lhs().args_slot] = {
            a_ptr, makeShape(problem.M, problem.K), makeStrides(a_sm, a_sk)};
        kernel_args.tensors[spec_.rhs().args_slot] = {
            b_ptr, makeShape(problem.K, problem.N), makeStrides(b_sk, b_sn)};
        kernel_args.tensors[spec_.output().args_slot] = {
            c_ptr, makeShape(problem.M, problem.N), makeStrides(c_sm, c_sn)};

        // Grid dimensions — partitioner-dependent
        int grid_m = (problem.M + spec_.block_tile.m - 1) / spec_.block_tile.m;
        int grid_n = (problem.N + spec_.block_tile.n - 1) / spec_.block_tile.n;

        int grid_x = grid_m * grid_n; // Linear / SpatiallyLocal
        int grid_y = 1;
        int grid_z = spec_.k_batch;

        if(spec_.tile_partitioner == TilePartitioner::Direct)
        {
            grid_x = grid_m;
            grid_y = grid_n;
        }

        // HIP event timing
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start, hip_stream));

        // Launch via hipModuleLaunchKernel
        void* args_ptr        = &kernel_args;
        size_t args_size      = sizeof(kernel_args);
        void* launch_config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                 args_ptr,
                                 HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                 &args_size,
                                 HIP_LAUNCH_PARAM_END};

        HIP_CHECK(hipModuleLaunchKernel(kernel_->function(),
                                        grid_x,
                                        grid_y,
                                        grid_z,
                                        spec_.workgroup_size,
                                        1,
                                        1,
                                        0,
                                        hip_stream,
                                        nullptr,
                                        launch_config));

        HIP_CHECK(hipEventRecord(stop, hip_stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        return elapsed_ms;
    }

    bool validate(const void* a_ptr,
                  const void* b_ptr,
                  const void* c_ptr,
                  const void** d_ptrs,
                  const ck_tile::dispatcher::Problem& problem,
                  float tolerance) const override
    {
        (void)d_ptrs; // D tensors not wired — validation is for basic GEMM only

        int M = static_cast<int>(problem.M);
        int N = static_cast<int>(problem.N);
        int K = static_cast<int>(problem.K);

        // Download A, B, C from device to host as float
        auto downloadAsFloat = [](const void* dev_ptr, DataType dtype, int count) {
            int elem_bytes = dataTypeBits(dtype) / 8;
            std::vector<char> raw(static_cast<size_t>(count) * elem_bytes);
            HIP_CHECK(hipMemcpy(raw.data(), dev_ptr, raw.size(), hipMemcpyDeviceToHost));
            std::vector<float> out(count);
            for(int i = 0; i < count; ++i)
                out[i] = typedToFloat(dtype, raw.data() + i * elem_bytes);
            return out;
        };

        auto h_a = downloadAsFloat(a_ptr, spec_.lhs().dtype, M * K);
        auto h_b = downloadAsFloat(b_ptr, spec_.rhs().dtype, K * N);
        auto h_c = downloadAsFloat(c_ptr, spec_.output().dtype, M * N);

        // CPU reference GEMM with layout-aware strides
        auto [a_sm, a_sk] = layoutStrides(spec_.lhs().layout, M, K);
        auto [b_sk, b_sn] = layoutStrides(spec_.rhs().layout, K, N);
        auto [c_sm, c_sn] = layoutStrides(spec_.output().layout, M, N);

        std::vector<float> ref(M * N, 0.0f);
        for(int m = 0; m < M; ++m)
            for(int n = 0; n < N; ++n)
            {
                float sum = 0.0f;
                for(int k = 0; k < K; ++k)
                    sum += h_a[m * a_sm + k * a_sk] * h_b[k * b_sk + n * b_sn];
                ref[m * c_sm + n * c_sn] = sum;
            }

        auto vr = verify(h_c.data(), ref.data(), M * N, tolerance);
        if(!vr.passed)
        {
            std::fprintf(stderr,
                         "%s: validation FAILED at [%d] got=%f expected=%f\n",
                         name_.c_str(),
                         vr.first_mismatch,
                         vr.got,
                         vr.expected);
        }
        return vr.passed;
    }

    private:
    ck_tile::dispatcher::KernelKey key_;
    std::string name_;
    GemmSpec spec_;
    std::shared_ptr<KpackArchive> archive_;
    mutable std::once_flag load_flag_;
    mutable std::optional<KpackKernel> kernel_;
};

} // namespace rocm_ck
