// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Kpack launch latency benchmark. Measures the full cost breakdown of
// runtime kernel loading via kpack + hipModule.
//
// Reports:
//   1. Archive open time (file I/O + GPU detection)
//   2. Kernel load time (kpack extraction + hipModuleLoadData + hipModuleGetFunction)
//   3. First-call launch latency (possible driver lazy init)
//   4. Steady-state per-launch overhead at multiple problem sizes
//   5. Pipelined throughput (no sync between launches)
//
// Usage: launch_latency_bench <path-to-gemm.kpack> [kernel_name [iters]]
//   kernel_name default: gemm_fp16
//   iters default: 20

#include <rocm_ck/args.hpp>
#include <rocm_ck/hip_check.hpp>
#include <rocm_ck/kpack_module.hpp>
#include <rocm_ck/kpack_spec_reader.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Clock = std::chrono::steady_clock;

// ============================================================================
// Statistics
// ============================================================================

struct Stats
{
    double min_us;
    double median_us;
    double p99_us;
    double mean_us;
};

Stats computeStats(std::vector<double>& v)
{
    std::sort(v.begin(), v.end());
    size_t n   = v.size();
    double sum = 0;
    for(double x : v)
        sum += x;
    return {v[0], v[n / 2], v[std::min(n - 1, static_cast<size_t>(n * 0.99))], sum / n};
}

void printStats(const char* label, const Stats& s)
{
    std::printf("  %-20s min=%9.1f  median=%9.1f  p99=%9.1f  mean=%9.1f us\n",
                label,
                s.min_us,
                s.median_us,
                s.p99_us,
                s.mean_us);
}

double us(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::micro>(b - a).count();
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    if(argc < 2)
    {
        std::fprintf(stderr,
                     "Usage: %s <kpack> [kernel_name [iters]]\n"
                     "  kernel_name: default gemm_fp16\n"
                     "  iters:       default 20\n",
                     argv[0]);
        return 1;
    }

    const char* kpack_path  = argv[1];
    const char* kernel_name = argc >= 3 ? argv[2] : "gemm_fp16";
    int iters               = argc >= 4 ? std::atoi(argv[3]) : 20;
    if(iters < 1)
        iters = 1;

    // ======================================================================
    // Phase 1: Archive open (timed)
    // ======================================================================
    auto t0 = Clock::now();
    rocm_ck::KpackArchive archive;
    if(!archive.open(kpack_path))
        return 1;
    auto t1 = Clock::now();

    std::printf("\n=== Kpack Launch Latency Benchmark ===\n");
    std::printf("Kernel: %s   Iters: %d\n", kernel_name, iters);
    std::printf("Archive open:  %7.0f us (%.1f ms)\n", us(t0, t1), us(t0, t1) / 1000);

    // ======================================================================
    // Phase 2: Kernel load (timed)
    // ======================================================================
    t0 = Clock::now();
    rocm_ck::KpackKernel kernel;
    if(!kernel.load(archive, kernel_name))
        return 1;
    t1 = Clock::now();
    std::printf("Kernel load:   %7.0f us (%.1f ms)\n", us(t0, t1), us(t0, t1) / 1000);

    int smem_static = 0;
    HIP_CHECK(
        hipFuncGetAttribute(&smem_static, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel.function()));
    std::printf("Shared memory: %d bytes (static)\n", smem_static);

    // ======================================================================
    // Phase 3: Read spec for grid config
    // ======================================================================
    auto variants                 = rocm_ck::readVariantSpecs(kpack_path);
    const rocm_ck::GemmSpec* spec = nullptr;
    for(const auto& vi : variants)
    {
        if(vi.name == kernel_name)
        {
            spec = &vi.gemm_spec;
            break;
        }
    }
    if(!spec)
    {
        std::fprintf(stderr, "Spec '%s' not in TOC\n", kernel_name);
        return 1;
    }

    std::printf("Tile: %dx%dx%d  WG: %d  k_batch: %d\n\n",
                spec->block_tile.m,
                spec->block_tile.n,
                spec->block_tile.k,
                spec->workgroup_size,
                spec->k_batch);

    // ======================================================================
    // Phase 4: Benchmark at each problem size
    // ======================================================================
    struct Size
    {
        int M, N, K;
        const char* tag;
    };
    Size sizes[] = {
        {512, 512, 256, " 512x512x256 "},
        {2048, 2048, 2048, "  2Kx2Kx2K   "},
        {4096, 4096, 4096, "  4Kx4Kx4K   "},
    };

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0));
    HIP_CHECK(hipEventCreate(&ev1));

    bool first_launch = true;

    for(const auto& sz : sizes)
    {
        int M = sz.M, N = sz.N, K = sz.K;
        int grid_m = (M + spec->block_tile.m - 1) / spec->block_tile.m;
        int grid_n = (N + spec->block_tile.n - 1) / spec->block_tile.n;
        int grid   = grid_m * grid_n;

        // Allocate device buffers (fp16 = 2 bytes per element)
        void *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        HIP_CHECK(hipMalloc(&d_a, static_cast<size_t>(M) * K * 2));
        HIP_CHECK(hipMalloc(&d_b, static_cast<size_t>(K) * N * 2));
        HIP_CHECK(hipMalloc(&d_c, static_cast<size_t>(M) * N * 2));
        HIP_CHECK(hipMemset(d_a, 0, static_cast<size_t>(M) * K * 2));
        HIP_CHECK(hipMemset(d_b, 0, static_cast<size_t>(K) * N * 2));
        HIP_CHECK(hipMemset(d_c, 0, static_cast<size_t>(M) * N * 2));

        // Build Args
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

        // --- First-ever launch (cold path) ---
        if(first_launch)
        {
            auto fa = Clock::now();
            HIP_CHECK(hipEventRecord(ev0, stream));
            HIP_CHECK(hipModuleLaunchKernel(kernel.function(),
                                            grid,
                                            1,
                                            spec->k_batch,
                                            spec->workgroup_size,
                                            1,
                                            1,
                                            0,
                                            stream,
                                            nullptr,
                                            config));
            auto fb = Clock::now();
            HIP_CHECK(hipEventRecord(ev1, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            auto fc = Clock::now();

            float gpu_ms;
            HIP_CHECK(hipEventElapsedTime(&gpu_ms, ev0, ev1));

            std::printf("--- First-call (%s) ---\n", sz.tag);
            std::printf("  Host submit:   %9.1f us\n", us(fa, fb));
            std::printf("  GPU execution: %9.1f us (%.3f ms)\n", gpu_ms * 1000, gpu_ms);
            std::printf("  Total wall:    %9.1f us (%.3f ms)\n\n", us(fa, fc), us(fa, fc) / 1000);
            first_launch = false;
        }

        // --- Warmup (2 iterations) ---
        for(int i = 0; i < 2; ++i)
        {
            HIP_CHECK(hipModuleLaunchKernel(kernel.function(),
                                            grid,
                                            1,
                                            spec->k_batch,
                                            spec->workgroup_size,
                                            1,
                                            1,
                                            0,
                                            stream,
                                            nullptr,
                                            config));
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        // --- Steady-state: serialized (sync per launch) ---
        std::vector<double> host_times(iters);
        std::vector<double> gpu_times(iters);
        std::vector<double> wall_times(iters);

        for(int i = 0; i < iters; ++i)
        {
            auto a = Clock::now();
            HIP_CHECK(hipEventRecord(ev0, stream));
            HIP_CHECK(hipModuleLaunchKernel(kernel.function(),
                                            grid,
                                            1,
                                            spec->k_batch,
                                            spec->workgroup_size,
                                            1,
                                            1,
                                            0,
                                            stream,
                                            nullptr,
                                            config));
            auto b = Clock::now();
            HIP_CHECK(hipEventRecord(ev1, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            auto c = Clock::now();

            float gms;
            HIP_CHECK(hipEventElapsedTime(&gms, ev0, ev1));

            host_times[i] = us(a, b);
            gpu_times[i]  = gms * 1000.0;
            wall_times[i] = us(a, c);
        }

        auto hs = computeStats(host_times);
        auto gs = computeStats(gpu_times);
        auto ws = computeStats(wall_times);

        double flops  = 2.0 * M * N * K;
        double tflops = flops / (gs.median_us * 1e6);

        std::printf("--- %s (grid=%d, %d iters) ---\n", sz.tag, grid, iters);
        printStats("Host submit:", hs);
        printStats("GPU execution:", gs);
        printStats("Total wall:", ws);
        std::printf("  Launch %% of wall:  %.2f%%\n", (hs.median_us / ws.median_us) * 100);
        std::printf("  TFLOPS:            %.2f\n\n", tflops);

        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));
    }

    // ======================================================================
    // Phase 5: Pipelined throughput (no sync between launches)
    // ======================================================================
    {
        int M = 2048, N = 2048, K = 2048;
        int grid_m = (M + spec->block_tile.m - 1) / spec->block_tile.m;
        int grid_n = (N + spec->block_tile.n - 1) / spec->block_tile.n;
        int grid   = grid_m * grid_n;

        void *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
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

        // warmup
        for(int i = 0; i < 2; ++i)
        {
            HIP_CHECK(hipModuleLaunchKernel(kernel.function(),
                                            grid,
                                            1,
                                            spec->k_batch,
                                            spec->workgroup_size,
                                            1,
                                            1,
                                            0,
                                            stream,
                                            nullptr,
                                            config));
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        auto a = Clock::now();
        for(int i = 0; i < iters; ++i)
        {
            HIP_CHECK(hipModuleLaunchKernel(kernel.function(),
                                            grid,
                                            1,
                                            spec->k_batch,
                                            spec->workgroup_size,
                                            1,
                                            1,
                                            0,
                                            stream,
                                            nullptr,
                                            config));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        auto b = Clock::now();

        double total = us(a, b);
        std::printf("--- Pipelined throughput (2Kx2Kx2K, %d launches) ---\n", iters);
        std::printf("  Total:      %.0f us (%.1f ms)\n", total, total / 1000);
        std::printf("  Per launch: %.1f us (amortized)\n\n", total / iters);

        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));
    }

    HIP_CHECK(hipEventDestroy(ev0));
    HIP_CHECK(hipEventDestroy(ev1));
    HIP_CHECK(hipStreamDestroy(stream));

    std::printf("=== Interpretation Guide ===\n");
    std::printf("- Archive open + kernel load: one-time cost at initialization\n");
    std::printf("- Host submit: hipModuleLaunchKernel async call return time\n");
    std::printf("  hipLaunchKernel (static) baseline: typically 5-10 us on AMD GPUs\n");
    std::printf("  If similar, kpack adds no per-launch overhead\n");
    std::printf("- Launch %% of wall: when << 100%%, GPU compute dominates\n");
    std::printf("- Pipelined: real-world throughput with command buffering\n\n");

    return 0;
}
