// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Dispatcher Bridge PoC — load a kpack archive, register its kernels with
// the CK dispatcher, run a GEMM through the dispatcher API, and verify.

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
    for(const auto& k : all_kernels)
    {
        std::printf(
            "  %-30s  %s\n", k->get_name().c_str(), k->get_key().encode_identifier().c_str());
    }
    std::printf("\n");

    // --- Find a basic FP16 kernel to test ---
    // registry.lookup() matches by encode_identifier(), not name.
    // Search by name instead.
    ck_tile::dispatcher::KernelInstancePtr kernel;
    for(const auto& k : all_kernels)
        if(k->get_name() == "gemm_fp16")
        {
            kernel = k;
            break;
        }
    if(!kernel)
    {
        kernel = all_kernels[0];
        std::printf("gemm_fp16 not found, using: %s\n", kernel->get_name().c_str());
    }

    // --- Setup test problem (same as example 04) ---
    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 256;

    ck_tile::dispatcher::Problem problem(M, N, K);

    if(!kernel->supports(problem))
    {
        std::fprintf(
            stderr, "%s does not support M=%d N=%d K=%d\n", kernel->get_name().c_str(), M, N, K);
        return 1;
    }

    // --- Generate input data (values 0-7 for exact fp16 accumulation) ---
    std::vector<float> h_a(M * K);
    std::vector<float> h_b(K * N);
    for(int i = 0; i < M * K; ++i)
        h_a[i] = static_cast<float>(i % 8);
    for(int i = 0; i < K * N; ++i)
        h_b[i] = static_cast<float>(i % 8);

    // --- Read the selected kernel's spec from kpack ---
    // We need the GemmSpec for layout info and dtype — read from the archive TOC.
    auto variants = rocm_ck::readVariantSpecs(argv[1]);
    rocm_ck::GemmSpec spec{};
    for(const auto& vi : variants)
        if(vi.name == kernel->get_name())
        {
            spec = vi.gemm_spec;
            break;
        }

    // --- CPU reference (layout-aware) ---
    auto [a_sm, a_sk] = rocm_ck::layoutStrides(spec.lhs().layout, M, K);
    auto [b_sk, b_sn] = rocm_ck::layoutStrides(spec.rhs().layout, K, N);
    auto [c_sm, c_sn] = rocm_ck::layoutStrides(spec.output().layout, M, N);

    std::vector<float> ref_c(M * N, 0.0f);
    cpuGemm(h_a.data(), h_b.data(), ref_c.data(), M, N, K, a_sm, a_sk, b_sk, b_sn, c_sm, c_sn);

    // --- Allocate and upload GPU buffers ---
    rocm_ck::TypedBuffer buf_a(spec.lhs().dtype, M * K);
    rocm_ck::TypedBuffer buf_b(spec.rhs().dtype, K * N);
    rocm_ck::TypedBuffer buf_c(spec.output().dtype, M * N);

    buf_a.upload(h_a.data());
    buf_b.upload(h_b.data());
    buf_c.zero();

    // --- Run through the dispatcher KernelInstance interface ---
    std::printf("Running %s: M=%d, N=%d, K=%d, block=%d\n",
                kernel->get_name().c_str(),
                M,
                N,
                K,
                kernel->get_key().algorithm.block_size);

    float time_ms = kernel->run(buf_a.ptr(), buf_b.ptr(), buf_c.ptr(), nullptr, problem, nullptr);

    // --- Verify ---
    std::vector<float> result(M * N);
    buf_c.download(result.data());

    // Tolerance depends on output dtype — fp16/bf16 have ~0.1% relative error
    // from accumulation over K elements, fp32/fp64 are near-exact.
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
        float rel_err =
            (expected != 0.0f) ? std::fabs((actual - expected) / expected) : std::fabs(actual);
        max_rel_err = std::fmax(max_rel_err, rel_err);
        if(rel_err > tolerance)
            ++errors;
    }

    bool passed = (errors == 0);
    std::printf("%s: %s  (%.3f ms, max_rel_err=%.2e, errors=%d/%d)\n",
                kernel->get_name().c_str(),
                passed ? "PASSED" : "FAILED",
                time_ms,
                max_rel_err,
                errors,
                M * N);

    if(!passed)
    {
        // Print first few errors
        int shown = 0;
        for(int i = 0; i < M * N && shown < 5; ++i)
        {
            float expected = ref_c[i];
            float actual   = result[i];
            float rel_err =
                (expected != 0.0f) ? std::fabs((actual - expected) / expected) : std::fabs(actual);
            if(rel_err > tolerance)
            {
                std::printf("  [%d] expected=%.4f actual=%.4f rel_err=%.2e\n",
                            i,
                            expected,
                            actual,
                            rel_err);
                ++shown;
            }
        }
    }

    // --- Validate through KernelInstance::validate() ---
    bool validated =
        kernel->validate(buf_a.ptr(), buf_b.ptr(), buf_c.ptr(), nullptr, problem, tolerance);
    std::printf("validate(): %s\n", validated ? "PASSED" : "FAILED");

    // --- Performance ---
    double tflops = (2.0 * M * N * K) / (time_ms * 1e9);
    std::printf("Performance: %.2f TFLOPS\n", tflops);

    return (passed && validated) ? 0 : 1;
}
