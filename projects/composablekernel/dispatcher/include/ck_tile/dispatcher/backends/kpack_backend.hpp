// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Kpack backend — runtime kernel loading from self-describing kpack archives.
//
// This backend implements the KernelInstance interface using hipModule to load
// pre-compiled kernels at runtime, rather than requiring static compilation.
// Kernels are loaded lazily on first use and cached for subsequent launches.
//
// Usage:
//   Registry registry;
//   int n = registerKpackKernels("gemm.kpack", registry);
//   Dispatcher dispatcher(&registry);
//   dispatcher.run(a, b, c, problem, stream);

#pragma once

#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/dispatcher/registry.hpp"

#include <rocm_ck/args.hpp>
#include <rocm_ck/datatype_convert.hpp>
#include <rocm_ck/gemm_spec.hpp>
#include <rocm_ck/gpu_arch.hpp>
#include <rocm_ck/hip_check.hpp>
#include <rocm_ck/kpack_module.hpp>
#include <rocm_ck/kpack_spec_reader.hpp>
#include <rocm_ck/layout.hpp>
#include <rocm_ck/projection.hpp>
#include <rocm_ck/verify.hpp>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

// Throwing variant of HIP_CHECK for use in run() — allows callers to catch
// and continue rather than terminating the process on GPU errors.
#define HIP_CHECK_THROW(call)                                                                  \
    do                                                                                         \
    {                                                                                          \
        hipError_t err = (call);                                                               \
        if(err != hipSuccess)                                                                  \
        {                                                                                      \
            throw std::runtime_error(                                                          \
                std::string("HIP error ") + std::to_string(static_cast<int>(err)) + " (" +     \
                hipGetErrorString(err) + ") at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                                      \
    } while(0)

namespace ck_tile {
namespace dispatcher {
namespace backends {

/// GEMM kernel instance loaded from a kpack archive at runtime.
///
/// Each instance wraps a single kernel variant. The archive is shared across
/// all variants from the same kpack file via shared_ptr.
///
/// Kernel loading is lazy and thread-safe — the hipModule is loaded on first
/// run() call via std::call_once.
class KpackGemmKernelInstance : public KernelInstance
{
    public:
    KpackGemmKernelInstance(KernelKey key,
                            std::string name,
                            rocm_ck::GemmSpec spec,
                            std::shared_ptr<rocm_ck::KpackArchive> archive)
        : key_(std::move(key)), name_(std::move(name)), spec_(spec), archive_(std::move(archive))
    {
    }

    const KernelKey& get_key() const override { return key_; }

    std::string get_name() const override { return name_; }

    bool supports(const Problem& problem) const override
    {
        if(!spec_.pad_m && problem.M % spec_.block_tile.m != 0)
            return false;
        if(!spec_.pad_n && problem.N % spec_.block_tile.n != 0)
            return false;
        // K must always be tile-aligned (CK Tile constraint — pad_k only
        // controls vector load width, does not mask the K-tail)
        if(problem.K % spec_.block_tile.k != 0)
            return false;
        if(problem.k_batch != spec_.k_batch)
            return false;
        return true;
    }

    float run(const void* a_ptr,
              const void* b_ptr,
              void* c_ptr,
              const void** d_ptrs,
              const Problem& problem,
              void* stream) const override
    {
        ensureLoaded();

        hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);

        // Build rocm_ck Args from dispatcher arguments + GemmSpec tensor table
        rocm_ck::Args kernel_args{};

        auto [a_sm, a_sk] = rocm_ck::layoutStrides(spec_.lhs().layout, problem.M, problem.K);
        auto [b_sk, b_sn] = rocm_ck::layoutStrides(spec_.rhs().layout, problem.K, problem.N);
        auto [c_sm, c_sn] = rocm_ck::layoutStrides(spec_.output().layout, problem.M, problem.N);

        kernel_args.tensors[spec_.lhs().args_slot] = {
            a_ptr, rocm_ck::makeShape(problem.M, problem.K), rocm_ck::makeStrides(a_sm, a_sk)};
        kernel_args.tensors[spec_.rhs().args_slot] = {
            b_ptr, rocm_ck::makeShape(problem.K, problem.N), rocm_ck::makeStrides(b_sk, b_sn)};
        kernel_args.tensors[spec_.output().args_slot] = {
            c_ptr, rocm_ck::makeShape(problem.M, problem.N), rocm_ck::makeStrides(c_sm, c_sn)};

        // D tensors (bias, scale, residual) — same [M, N] shape as output
        int num_d = spec_.numDTensors();
        if(num_d > 0)
        {
            if(d_ptrs == nullptr)
                throw std::runtime_error("KpackGemmKernelInstance: kernel '" + name_ +
                                         "' expects " + std::to_string(num_d) +
                                         " D tensor(s) but d_ptrs is null");

            auto [d_sm, d_sn] = rocm_ck::layoutStrides(spec_.output().layout, problem.M, problem.N);

            for(int i = 0; i < num_d; ++i)
            {
                if(d_ptrs[i] == nullptr)
                    throw std::runtime_error("KpackGemmKernelInstance: d_ptrs[" +
                                             std::to_string(i) + "] is null for kernel '" + name_ +
                                             "'");

                auto d_tensor                           = spec_.physical_tensors[3 + i];
                kernel_args.tensors[d_tensor.args_slot] = {d_ptrs[i],
                                                           rocm_ck::makeShape(problem.M, problem.N),
                                                           rocm_ck::makeStrides(d_sm, d_sn)};
            }
        }

        // Grid dimensions — partitioner-dependent
        int grid_m = (problem.M + spec_.block_tile.m - 1) / spec_.block_tile.m;
        int grid_n = (problem.N + spec_.block_tile.n - 1) / spec_.block_tile.n;

        int grid_x = grid_m * grid_n; // Linear / SpatiallyLocal
        int grid_y = 1;
        int grid_z = spec_.k_batch;

        if(spec_.tile_partitioner == rocm_ck::TilePartitioner::Direct)
        {
            grid_x = grid_m;
            grid_y = grid_n;
        }

        // HIP event timing
        hipEvent_t start, stop;
        HIP_CHECK_THROW(hipEventCreate(&start));
        HIP_CHECK_THROW(hipEventCreate(&stop));
        HIP_CHECK_THROW(hipEventRecord(start, hip_stream));

        void* args_ptr        = &kernel_args;
        size_t args_size      = sizeof(kernel_args);
        void* launch_config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                 args_ptr,
                                 HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                 &args_size,
                                 HIP_LAUNCH_PARAM_END};

        HIP_CHECK_THROW(hipModuleLaunchKernel(kernel_->function(),
                                              grid_x,
                                              grid_y,
                                              grid_z,
                                              spec_.workgroup_size,
                                              1,
                                              1,
                                              shared_mem_bytes_,
                                              hip_stream,
                                              nullptr,
                                              launch_config));

        HIP_CHECK_THROW(hipEventRecord(stop, hip_stream));
        HIP_CHECK_THROW(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK_THROW(hipEventElapsedTime(&elapsed_ms, start, stop));

        HIP_CHECK_THROW(hipEventDestroy(start));
        HIP_CHECK_THROW(hipEventDestroy(stop));

        return elapsed_ms;
    }

    bool validate(const void* a_ptr,
                  const void* b_ptr,
                  const void* c_ptr,
                  const void** d_ptrs,
                  const Problem& problem,
                  float tolerance) const override
    {
        (void)d_ptrs; // D tensors not validated — reference is basic GEMM only

        int M = static_cast<int>(problem.M);
        int N = static_cast<int>(problem.N);
        int K = static_cast<int>(problem.K);

        auto downloadAsFloat = [](const void* dev_ptr, rocm_ck::DataType dtype, int count) {
            int elem_bytes = rocm_ck::dataTypeBits(dtype) / 8;
            std::vector<char> raw(static_cast<size_t>(count) * elem_bytes);
            HIP_CHECK(hipMemcpy(raw.data(), dev_ptr, raw.size(), hipMemcpyDeviceToHost));
            std::vector<float> out(count);
            for(int i = 0; i < count; ++i)
                out[i] = rocm_ck::typedToFloat(dtype, raw.data() + i * elem_bytes);
            return out;
        };

        auto h_a = downloadAsFloat(a_ptr, spec_.lhs().dtype, M * K);
        auto h_b = downloadAsFloat(b_ptr, spec_.rhs().dtype, K * N);
        auto h_c = downloadAsFloat(c_ptr, spec_.output().dtype, M * N);

        auto [a_sm, a_sk] = rocm_ck::layoutStrides(spec_.lhs().layout, M, K);
        auto [b_sk, b_sn] = rocm_ck::layoutStrides(spec_.rhs().layout, K, N);
        auto [c_sm, c_sn] = rocm_ck::layoutStrides(spec_.output().layout, M, N);

        std::vector<float> ref(M * N, 0.0f);
        for(int m = 0; m < M; ++m)
            for(int n = 0; n < N; ++n)
            {
                float sum = 0.0f;
                for(int k = 0; k < K; ++k)
                    sum += h_a[m * a_sm + k * a_sk] * h_b[k * b_sk + n * b_sn];
                ref[m * c_sm + n * c_sn] = sum;
            }

        auto vr = rocm_ck::verify(h_c.data(), ref.data(), M * N, tolerance);
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
    void ensureLoaded() const
    {
        // call_once retries on exception (once_flag not set until success)
        std::call_once(load_flag_, [this]() {
            kernel_.emplace();
            if(!kernel_->load(*archive_, name_.c_str()))
            {
                kernel_.reset();
                throw std::runtime_error("Failed to load kernel '" + name_ + "' from kpack");
            }

            int smem = 0;
            HIP_CHECK_THROW(hipFuncGetAttribute(
                &smem, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel_->function()));
            shared_mem_bytes_ = static_cast<unsigned int>(smem);
        });
    }

    KernelKey key_;
    std::string name_;
    rocm_ck::GemmSpec spec_;
    std::shared_ptr<rocm_ck::KpackArchive> archive_;
    mutable std::once_flag load_flag_;
    mutable std::optional<rocm_ck::KpackKernel> kernel_;
    mutable unsigned int shared_mem_bytes_ = 0;
};

// =============================================================================
// Registration
// =============================================================================

/// Load a kpack archive and register all compatible GEMM kernels.
///
/// Reads variant_specs from the kpack TOC, filters by current GPU and
/// supported features, projects each GemmSpec to a KernelKey, and
/// registers a KpackGemmKernelInstance in the registry.
///
/// Returns the number of kernels successfully registered.
inline int registerKpackKernels(const char* kpack_path,
                                Registry& registry,
                                Registry::Priority priority = Registry::Priority::Normal)
{
    auto archive = std::make_shared<rocm_ck::KpackArchive>();
    if(!archive->open(kpack_path))
        return 0;

    auto variants = rocm_ck::readVariantSpecs(kpack_path);
    if(variants.empty())
    {
        std::fprintf(stderr,
                     "No variant specs in kpack TOC — archive may predate "
                     "self-describing format\n");
        return 0;
    }

    auto detected_target = rocm_ck::detectGpuTarget();
    if(!detected_target)
    {
        std::fprintf(stderr, "Unsupported GPU for kpack registration\n");
        return 0;
    }
    std::string arch = rocm_ck::getGpuArch();

    int registered = 0;
    int skipped    = 0;

    for(const auto& vi : variants)
    {
        if(vi.spec_type != "GemmSpec")
            continue;

        if(!vi.targets.contains(*detected_target))
        {
            ++skipped;
            continue;
        }

        if(vi.gemm_spec.pipeline == rocm_ck::Pipeline::Preshuffle)
        {
            std::fprintf(stderr, "  skip %s: preshuffle not supported\n", vi.name.c_str());
            ++skipped;
            continue;
        }

        if(vi.gemm_spec.group_size > 0)
        {
            std::fprintf(stderr, "  skip %s: quantized kernels not supported\n", vi.name.c_str());
            ++skipped;
            continue;
        }

        if(vi.gemm_spec.tile_partitioner == rocm_ck::TilePartitioner::StreamK)
        {
            std::fprintf(stderr, "  skip %s: StreamK not supported\n", vi.name.c_str());
            ++skipped;
            continue;
        }

        auto key = rocm_ck::projectToDispatcher(vi.gemm_spec, arch);
        auto instance =
            std::make_shared<KpackGemmKernelInstance>(key, vi.name, vi.gemm_spec, archive);

        if(registry.register_kernel(instance, priority))
            ++registered;
    }

    if(registered == 0 && skipped > 0)
        std::fprintf(stderr,
                     "  No kernels registered — %d variants skipped "
                     "(arch mismatch or unsupported features)\n",
                     skipped);
    else if(skipped > 0)
        std::fprintf(stderr, "  (%d variants skipped)\n", skipped);

    return registered;
}

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile
