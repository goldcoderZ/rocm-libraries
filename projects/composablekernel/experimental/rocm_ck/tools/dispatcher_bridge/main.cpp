// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Dispatcher Bridge — load a kpack archive, register its kernels with
// the CK dispatcher, run each GEMM through the dispatcher API, and verify.

#include <ck_tile/dispatcher/backends/kpack_backend.hpp>

#include <rocm_ck/hip_check.hpp>
#include <rocm_ck/kpack_spec_reader.hpp>
#include <rocm_ck/layout.hpp>
#include <rocm_ck/typed_buffer.hpp>

#include <ck_tile/dispatcher/dispatcher.hpp>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <vector>

// ============================================================================
// CPU reference
// ============================================================================

static void cpuGemm(const float* a,
                    const float* b,
                    float* c,
                    int M,
                    int N,
                    int K,
                    int a_stride_m,
                    int a_stride_k,
                    int b_stride_k,
                    int b_stride_n,
                    int c_stride_m,
                    int c_stride_n)
{
    for(int m = 0; m < M; ++m)
        for(int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for(int k = 0; k < K; ++k)
                sum += a[m * a_stride_m + k * a_stride_k] * b[k * b_stride_k + n * b_stride_n];
            c[m * c_stride_m + n * c_stride_n] = sum;
        }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::fprintf(stderr, "Usage: %s <path-to-gemm.kpack>\n", argv[0]);
        return 1;
    }

    // --- Register kpack kernels with the dispatcher registry ---
    ck_tile::dispatcher::Registry registry;
    int n_registered = ck_tile::dispatcher::backends::registerKpackKernels(argv[1], registry);
    std::printf("\nRegistered %d kernels from kpack\n\n", n_registered);

    if(n_registered == 0)
    {
        std::fprintf(stderr, "No kernels registered — nothing to dispatch\n");
        return 1;
    }

    // --- List registered kernels ---
    auto all_kernels = registry.get_all();
    std::printf("Registered kernels:\n");
    for(size_t i = 0; i < all_kernels.size(); ++i)
        std::printf("  [%zu] %s\n", i + 1, all_kernels[i]->get_name().c_str());
    std::printf("\n");

    // --- Read variant specs for dtype/layout info ---
    auto variants = rocm_ck::readVariantSpecs(argv[1]);

    std::unordered_map<std::string, rocm_ck::GemmSpec> spec_map;
    for(const auto& vi : variants)
        spec_map[vi.name] = vi.gemm_spec;

    // --- Setup test problem ---
    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 256;

    ck_tile::dispatcher::Problem problem(M, N, K);

    // --- Run and verify each kernel ---
    int total   = 0;
    int pass    = 0;
    int fail    = 0;
    int skipped = 0;

    for(size_t ki = 0; ki < all_kernels.size(); ++ki)
    {
        const auto& kernel = all_kernels[ki];
        int num            = static_cast<int>(ki + 1);

        auto it = spec_map.find(kernel->get_name());
        if(it == spec_map.end())
        {
            std::printf("[%d] %s\n    skip: no spec in TOC\n", num, kernel->get_name().c_str());
            ++skipped;
            continue;
        }
        const auto& spec = it->second;

        if(!kernel->supports(problem))
        {
            std::printf("[%d] %s\n    skip: does not support M=%d N=%d K=%d\n",
                        num,
                        kernel->get_name().c_str(),
                        M,
                        N,
                        K);
            ++skipped;
            continue;
        }

        std::printf("[%d] %s\n", num, kernel->get_name().c_str());
        std::fflush(stdout);

        try
        {
            // Generate input data (values 0-7 for exact fp16 accumulation)
            std::vector<float> h_a(M * K);
            std::vector<float> h_b(K * N);
            for(int i = 0; i < M * K; ++i)
                h_a[i] = static_cast<float>(i % 8);
            for(int i = 0; i < K * N; ++i)
                h_b[i] = static_cast<float>(i % 8);

            // D tensor test data (small values for exact accumulation)
            int num_d = spec.numDTensors();
            std::vector<float> h_d0(num_d >= 1 ? M * N : 0);
            std::vector<float> h_d1(num_d >= 2 ? M * N : 0);
            for(int i = 0; i < static_cast<int>(h_d0.size()); ++i)
                h_d0[i] = static_cast<float>(i % 4);
            for(int i = 0; i < static_cast<int>(h_d1.size()); ++i)
                h_d1[i] = static_cast<float>(i % 3);

            // CPU reference (layout-aware)
            auto [a_sm, a_sk] = rocm_ck::layoutStrides(spec.lhs().layout, M, K);
            auto [b_sk, b_sn] = rocm_ck::layoutStrides(spec.rhs().layout, K, N);
            auto [c_sm, c_sn] = rocm_ck::layoutStrides(spec.output().layout, M, N);

            std::vector<float> ref_c(M * N, 0.0f);
            cpuGemm(
                h_a.data(), h_b.data(), ref_c.data(), M, N, K, a_sm, a_sk, b_sk, b_sn, c_sm, c_sn);

            // Apply fused epilogue to CPU reference
            if(spec.hasEpilogueOp(rocm_ck::EpilogueOp::Add) &&
               spec.hasEpilogueOp(rocm_ck::EpilogueOp::Relu))
            {
                for(int i = 0; i < M * N; ++i)
                    ref_c[i] = std::fmax(0.0f, ref_c[i] + h_d0[i]);
            }
            else if(spec.hasEpilogueOp(rocm_ck::EpilogueOp::Add) && num_d >= 2)
            {
                for(int i = 0; i < M * N; ++i)
                    ref_c[i] = ref_c[i] + h_d0[i] + h_d1[i];
            }
            else if(spec.hasEpilogueOp(rocm_ck::EpilogueOp::Add))
            {
                for(int i = 0; i < M * N; ++i)
                    ref_c[i] = ref_c[i] + h_d0[i];
            }

            // Allocate and upload GPU buffers
            rocm_ck::TypedBuffer buf_a(spec.lhs().dtype, M * K);
            rocm_ck::TypedBuffer buf_b(spec.rhs().dtype, K * N);
            rocm_ck::TypedBuffer buf_c(spec.output().dtype, M * N);

            buf_a.upload(h_a.data());
            buf_b.upload(h_b.data());
            buf_c.zero();

            // D tensor GPU buffers
            std::unique_ptr<rocm_ck::TypedBuffer> buf_d0;
            std::unique_ptr<rocm_ck::TypedBuffer> buf_d1;
            std::vector<const void*> d_ptr_vec;

            if(num_d >= 1)
            {
                buf_d0 = std::make_unique<rocm_ck::TypedBuffer>(spec.d0().dtype, M * N);
                buf_d0->upload(h_d0.data());
                d_ptr_vec.push_back(buf_d0->ptr());
            }
            if(num_d >= 2)
            {
                buf_d1 = std::make_unique<rocm_ck::TypedBuffer>(spec.d1().dtype, M * N);
                buf_d1->upload(h_d1.data());
                d_ptr_vec.push_back(buf_d1->ptr());
            }

            const void** d_ptrs = d_ptr_vec.empty() ? nullptr : d_ptr_vec.data();

            // Run
            float time_ms =
                kernel->run(buf_a.ptr(), buf_b.ptr(), buf_c.ptr(), d_ptrs, problem, nullptr);

            // Verify
            std::vector<float> result(M * N);
            buf_c.download(result.data());

            float tolerance = 1e-5f;
            if(spec.output().dtype == rocm_ck::DataType::FP16 ||
               spec.output().dtype == rocm_ck::DataType::BF16)
                tolerance = 1e-2f;
            else if(spec.output().dtype == rocm_ck::DataType::FP8_FNUZ ||
                    spec.output().dtype == rocm_ck::DataType::FP8_OCP ||
                    spec.output().dtype == rocm_ck::DataType::BF8_FNUZ ||
                    spec.output().dtype == rocm_ck::DataType::BF8_OCP)
                tolerance = 5e-2f;

            int errors        = 0;
            float max_rel_err = 0.0f;
            for(int i = 0; i < M * N; ++i)
            {
                float expected = ref_c[i];
                float actual   = result[i];
                float rel_err  = (expected != 0.0f) ? std::fabs((actual - expected) / expected)
                                                    : std::fabs(actual);
                max_rel_err    = std::fmax(max_rel_err, rel_err);
                if(rel_err > tolerance)
                    ++errors;
            }

            bool kernel_passed = (errors == 0);
            double tflops      = (2.0 * M * N * K) / (time_ms * 1e9);

            std::printf("    %s  (%.3f ms, %.2f TFLOPS, max_rel_err=%.2e, errors=%d/%d)\n",
                        kernel_passed ? "PASSED" : "FAILED",
                        time_ms,
                        tflops,
                        max_rel_err,
                        errors,
                        M * N);

            if(!kernel_passed)
            {
                int shown = 0;
                for(int i = 0; i < M * N && shown < 5; ++i)
                {
                    float expected = ref_c[i];
                    float actual   = result[i];
                    float rel_err  = (expected != 0.0f) ? std::fabs((actual - expected) / expected)
                                                        : std::fabs(actual);
                    if(rel_err > tolerance)
                    {
                        std::printf("      [%d] expected=%.4f actual=%.4f rel_err=%.2e\n",
                                    i,
                                    expected,
                                    actual,
                                    rel_err);
                        ++shown;
                    }
                }
                ++fail;
            }
            else
            {
                ++pass;
            }
        }
        catch(const std::exception& e)
        {
            std::fflush(stderr);
            std::printf("\nERROR RUNNING KERNEL [%d]\n%s\n", num, e.what());
            ++fail;

            // Check if the GPU is still usable. hipGetLastError returns the
            // sticky error state — if it's non-zero, the device context is
            // unrecoverable on ROCm and we must stop.
            hipError_t gpu_err = hipGetLastError();
            if(gpu_err != hipSuccess)
            {
                ++total;
                std::printf("GPU fault (%s) — remaining kernels skipped.\n",
                            hipGetErrorString(gpu_err));
                break;
            }
            // Host-side error (validation, load failure) — safe to continue
        }
        ++total;
    }

    // --- Summary ---
    std::printf("\n%d/%d passed", pass, total);
    if(skipped > 0)
        std::printf(", %d skipped", skipped);
    if(fail > 0)
        std::printf(", %d FAILED", fail);
    std::printf("\n");

    return (fail == 0) ? 0 : 1;
}
