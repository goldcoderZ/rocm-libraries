/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#include "ProgramOptions.hpp"
#include "Reference.hpp"
#include "rocisa/include/enum.hpp"
#include <Tensile/Activation.hpp>

/*
 * CPU GEMM Driver and Validator
 *
 * This tool acts as a test harness for the TensileLite CPU GEMM implementation.
 * It allows for command-line verification of matrix multiplication kernels across
 * different data types (f32, f16, bf16) and geometries. It can also be used for
 * benchmarking different CPU GEMM implementations.
 *
 * The driver performs the following steps:
 * 1. Sets up a contraction problem based on user arguments (M, N, K, Transpose, etc).
 * 2. Initializes input matrices (A, B) with random data.
 * 3. Executes the "Device Under Test" (the optimized CPU solve).
 * 4. Optionally validates the result against a simple, golden reference implementation.
 *
 * Usage Examples:
 * # Standard f32 run
 * ./cpu_gemm_driver --M 1024 --N 1024 --K 1024
 *
 * # BF16 run with validation enabled
 * ./cpu_gemm_driver --type bf16 --M 512 --N 512 --K 256 --validate 1
 *
 * # Benchmark mode (validation disabled)
 * ./cpu_gemm_driver --M 2048 --N 2048 --K 2048 --validate 0 --tryFastPath 1
 *
 * # Help messnage
 * ./cpu_gemm_driver --help
 */

namespace
{
    using namespace TensileLite;
    using namespace TensileLite::Client;

    // Helper traits to map C++ storage types to rocisa data type enums.
    template <typename T>
    struct TypeTraits;

    template <>
    struct TypeTraits<float>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::Float;
    };

    template <>
    struct TypeTraits<TensileLite::Half>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::Half;
    };

    template <>
    struct TypeTraits<TensileLite::BFloat16>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::BFloat16;
    };

#ifndef _WIN32
    template <>
    struct TypeTraits<TensileLite::Float4x2>
    {
        static constexpr rocisa::DataType value = rocisa::DataType::Float4;
    };
#endif

    // A naive, slow, golden reference implementation of GEMM.
    // Used strictly for validating the correctness of the optimized path.
    // Calculates D = activation(alpha * scaleA[i] * scaleB[j] * scaleAlphaVec[d] * (A * B) + beta * C + bias[i])
    // where d = i (row/M) when factorDim==0, or d = j (col/N) when factorDim==1.
    //
    // When mxBlock > 0 and mxScaleA/mxScaleB are non-null, the K-reduction is
    // block-structured: accumulate mxBlock products, then scale by the MX
    // scale factors before adding to the running sum.
    void columnMajorGemm(const float*   a,
                         const float*   b,
                         const float*   c,
                         float*         d,
                         size_t         m,
                         size_t         n,
                         size_t         k,
                         bool           transA,
                         bool           transB,
                         float          alpha,
                         float          beta,
                         const float*   biasVec       = nullptr,
                         const float*   scaleAlphaVec = nullptr,
                         ActivationType activation    = ActivationType::None,
                         const float*   scaleAVec     = nullptr,
                         const float*   scaleBVec     = nullptr,
                         int            factorDim     = 0,
#ifndef _WIN32
                         const MXScale* mxScaleA      = nullptr,
                         const MXScale* mxScaleB      = nullptr,
#else
                         const void*    mxScaleA      = nullptr,
                         const void*    mxScaleB      = nullptr,
#endif
                         int            mxBlock       = 0)
    {
        size_t strideAK = transA ? 1 : m;
        size_t strideAM = transA ? k : 1;
        size_t strideBK = transB ? n : 1;
        size_t strideBN = transB ? 1 : k;

        // MX scale tensor layout (column-major, tight strides):
        //   !transA: mxsa = {m, k/mxBlock}, idx = row + block * m
        //    transA: mxsa = {k/mxBlock, m}, idx = block + row * (k/mxBlock)
        //   !transB: mxsb = {k/mxBlock, n}, idx = block + col * (k/mxBlock)
        //    transB: mxsb = {n, k/mxBlock}, idx = col + block * n
        size_t kBlocks = (mxBlock > 0) ? k / static_cast<size_t>(mxBlock) : 0;

        for(size_t i = 0; i < m; i++)
        {
            for(size_t j = 0; j < n; j++)
            {
                float sum = 0.0f;

#ifndef _WIN32
                if(mxBlock > 0 && mxScaleA && mxScaleB)
                {
                    for(size_t blk = 0; blk < kBlocks; blk++)
                    {
                        float blockSum = 0.0f;
                        size_t lBase   = blk * static_cast<size_t>(mxBlock);
                        for(size_t t = 0; t < static_cast<size_t>(mxBlock); t++)
                        {
                            size_t l    = lBase + t;
                            float  aVal = a[i * strideAM + l * strideAK];
                            float  bVal = b[l * strideBK + j * strideBN];
                            blockSum += aVal * bVal;
                        }

                        size_t mxsaIdx = transA ? (blk + i * kBlocks)
                                                : (i + blk * m);
                        size_t mxsbIdx = transB ? (j + blk * n)
                                                : (blk + j * kBlocks);

                        float mxScale = static_cast<float>(mxScaleA[mxsaIdx])
                                      * static_cast<float>(mxScaleB[mxsbIdx]);
                        sum += blockSum * mxScale;
                    }
                }
                else
#endif
                {
                    for(size_t l = 0; l < k; l++)
                    {
                        float aVal = a[i * strideAM + l * strideAK];
                        float bVal = b[l * strideBK + j * strideBN];
                        sum += aVal * bVal;
                    }
                }

                float effectiveAlpha = alpha;
                if(scaleAVec)
                    effectiveAlpha *= scaleAVec[i];
                if(scaleBVec)
                    effectiveAlpha *= scaleBVec[j];
                if(scaleAlphaVec)
                    effectiveAlpha *= scaleAlphaVec[factorDim == 0 ? i : j];

                float result = effectiveAlpha * sum + beta * c[i + j * m];

                if(biasVec)
                    result += biasVec[i];

                if(activation == ActivationType::Relu)
                {
                    result = std::max(0.0f, result);
                }
                else
                {
                    assert(activation == ActivationType::None);
                }

                d[i + j * m] = result;
            }
        }
    }
}

/*
 * Main templated runner.
 * Handles memory allocation, data initialization, execution, and validation.
 *
 * InputT: The C++ type used for storage of A and B matrices (e.g. float, half).
 * AccumulateT: The type used for accumulation (currently restricted to float).
 */
template <typename InputT, typename AccumulateT = float>
int runGemm(size_t         m,
            size_t         n,
            size_t         k,
            bool           transA,
            bool           transB,
            float          alpha,
            float          beta,
            bool           validate,
            bool           tryFastPath,
            bool           useBias,
            ActivationType activation,
            bool               useScaleAlphaVec,
            const std::string& useScaleAB,
            int                factorDim,
            int                mxBlock = 0)
{
    constexpr rocisa::DataType dtypeEnum = TypeTraits<InputT>::value;
    static_assert(std::is_same<AccumulateT, float>::value,
                  "Currently only float accumulation is supported");

#ifndef _WIN32
    constexpr bool isFP4 = std::is_same_v<InputT, Float4x2>;
#else
    constexpr bool isFP4 = false;
#endif

    if constexpr(!isFP4)
        mxBlock = 0;

    if constexpr(isFP4)
    {
        if(mxBlock > 0)
        {
            if(k < static_cast<size_t>(mxBlock))
            {
                std::cerr << "Error: K (" << k << ") must be >= mxBlock (" << mxBlock << ")"
                          << std::endl;
                return 1;
            }
            if(k % static_cast<size_t>(mxBlock) != 0)
            {
                std::cerr << "Error: K (" << k << ") must be a multiple of mxBlock (" << mxBlock
                          << ")" << std::endl;
                return 1;
            }
        }
        tryFastPath = false;
    }

    // Calculate strides assuming standard column-major packed storage
    size_t lda        = transA ? k : m;
    size_t ldb        = transB ? n : k;
    size_t ldc        = m;
    size_t batchCount = 1;

    // Define the contraction problem (geometry, strides, types)
    ContractionProblemGemm contraction
        = ContractionProblemGemm::GEMM_Strides(transA,
                                               transB,
                                               dtypeEnum,
                                               dtypeEnum,
                                               rocisa::DataType::Float,
                                               rocisa::DataType::Float,
                                               m,
                                               n,
                                               k,
                                               batchCount,
                                               lda,
                                               -1,
                                               ldb,
                                               -1,
                                               ldc,
                                               -1,
                                               ldc,
                                               -1,
                                               static_cast<double>(beta));

    contraction.setComputeInputTypeA(dtypeEnum);
    contraction.setComputeInputTypeB(dtypeEnum);
    contraction.setAlphaType(rocisa::DataType::Float);
    contraction.setBetaType(rocisa::DataType::Float);

    // Allocate host memory for inputs and outputs
    size_t numA = m * k;
    size_t numB = k * n;

    size_t storageA, storageB;
    if constexpr(isFP4)
    {
        storageA = (numA + 1) / 2;
        storageB = (numB + 1) / 2;
    }
    else
    {
        storageA = numA;
        storageB = numB;
    }

    std::vector<InputT> a(storageA);
    std::vector<InputT> b(storageB);
    std::vector<float>  c(m * n);
    std::vector<float>  d(m * n);

    // Initialize inputs with random values
    size_t                          seed = 42;
    std::mt19937                    gen(seed);
    std::uniform_int_distribution<> binary_distribution(0, 1);

    auto randomGen = [&]() { return binary_distribution(gen) ? 1.0f : -1.0f; };

    if constexpr(isFP4)
    {
        // E2M1-representable values for diverse test coverage
        constexpr float fp4Values[] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
        std::uniform_int_distribution<> fp4Dist(0, 6);
        auto randomFp4 = [&]() { return fp4Values[fp4Dist(gen)]; };

        for(size_t i = 0; i < storageA; i++)
        {
            float v0 = randomFp4();
            float v1 = (i == storageA - 1 && numA % 2 != 0) ? 0.0f : randomFp4();
            a[i]     = Float4x2(v0, v1);
        }
        for(size_t i = 0; i < storageB; i++)
        {
            float v0 = randomFp4();
            float v1 = (i == storageB - 1 && numB % 2 != 0) ? 0.0f : randomFp4();
            b[i]     = Float4x2(v0, v1);
        }
    }
    else
    {
        std::generate(a.begin(), a.end(), [&]() { return static_cast<InputT>(randomGen()); });
        std::generate(b.begin(), b.end(), [&]() { return static_cast<InputT>(randomGen()); });
    }
    std::generate(c.begin(), c.end(), [&]() { return static_cast<float>(randomGen()); });

    // Optional feature buffers
    std::vector<float> biasVec;
    std::vector<float> scaleAlphaVecBuf;

    if(useBias)
    {
        biasVec.resize(m);
        std::generate(biasVec.begin(), biasVec.end(), randomGen);
        contraction.setUseBias(1);
        contraction.setBias(rocisa::DataType::Float, m, m);
    }

    if(useScaleAlphaVec)
    {
        size_t scaleAlphaVecLen = (factorDim == 0) ? m : n;
        scaleAlphaVecBuf.resize(scaleAlphaVecLen);
        std::generate(scaleAlphaVecBuf.begin(), scaleAlphaVecBuf.end(), randomGen);
        contraction.setUseScaleAlphaVec(1);
        contraction.setScaleAlphaVec(rocisa::DataType::Float, scaleAlphaVecLen, factorDim);
    }

    // Random scale generator: magnitude in (1, 100], integer values to avoid rounding issues, sign random.
    // Excludes 0 and ±1 so missing/incorrect scaling is never masked.
    std::uniform_int_distribution<int> scaleDis(2, 100);
    auto scaleGen = [&]() {
        float sign = binary_distribution(gen) ? 1.0f : -1.0f;
        int mag    = scaleDis(gen);
        return sign * static_cast<float>(mag);
    };

    std::vector<float> scaleABuf;
    std::vector<float> scaleBBuf;

    if(useScaleAB == "Scalar")
    {
        scaleABuf = {scaleGen()};
        scaleBBuf = {scaleGen()};
        // setUseScaleAB must be called before setScaleA/setScaleB,
        // because setScaleA/B silently skips tensor registration when
        // m_useScaleAB is still empty.
        contraction.setUseScaleAB("Scalar");
        contraction.setScaleA(rocisa::DataType::Float, 1);
        contraction.setScaleB(rocisa::DataType::Float, 1);
    }
    else if(useScaleAB == "Vector")
    {
        scaleABuf.resize(m);
        scaleBBuf.resize(n);
        std::generate(scaleABuf.begin(), scaleABuf.end(), scaleGen);
        std::generate(scaleBBuf.begin(), scaleBBuf.end(), scaleGen);
        contraction.setUseScaleAB("Vector");
        contraction.setScaleA(rocisa::DataType::Float, m);
        contraction.setScaleB(rocisa::DataType::Float, n);
    }

    if(activation != ActivationType::None)
    {
        contraction.setActivationType(ActivationType::All);
        contraction.setParams().setActivationEnum(activation);
    }

    // MX scale setup (FP4 with mxBlock > 0 only)
    [[maybe_unused]] std::vector<MXScale> mxsa, mxsb;

    if constexpr(isFP4)
    {
        if(mxBlock > 0)
        {
            contraction.setMXScaleA(mxBlock);
            contraction.setMXScaleB(mxBlock);

            size_t nmxsa = contraction.mxsa().totalLogicalElements();
            size_t nmxsb = contraction.mxsb().totalLogicalElements();

            if(nmxsa == 0 || nmxsb == 0)
            {
                std::cerr << "Error: MX scale tensor has zero elements (nmxsa=" << nmxsa
                          << ", nmxsb=" << nmxsb << ")" << std::endl;
                return 1;
            }

            mxsa.resize(nmxsa);
            mxsb.resize(nmxsb);

            // Distinct exponents in [0..7] so wrong indexing breaks validation
            std::uniform_int_distribution<> expDist(0, 7);
            for(size_t i = 0; i < nmxsa; i++)
                mxsa[i] = MXScale(std::ldexp(1.0f, expDist(gen)));
            for(size_t i = 0; i < nmxsb; i++)
                mxsb[i] = MXScale(std::ldexp(1.0f, expDist(gen)));
        }
    }

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), alpha, beta);
    inputs.bias          = useBias ? biasVec.data() : nullptr;
    inputs.scaleAlphaVec = useScaleAlphaVec ? scaleAlphaVecBuf.data() : nullptr;
    inputs.scaleA        = (useScaleAB != "none") ? scaleABuf.data() : nullptr;
    inputs.scaleB        = (useScaleAB != "none") ? scaleBBuf.data() : nullptr;

    if constexpr(isFP4)
    {
        inputs.mxsa = (mxBlock > 0) ? mxsa.data() : nullptr;
        inputs.mxsb = (mxBlock > 0) ? mxsb.data() : nullptr;
    }

    auto start = std::chrono::high_resolution_clock::now();

    if(tryFastPath && !TensileLite::Client::isFastPathEligible(contraction))
    {
        throw std::runtime_error(
            "--tryFastPath was requested but the problem is not eligible "
            "for the fast CPU GEMM path.");
    }

    // Execute the 'device under test'.
    // passing -1 for elementsToValidate ensures that the 'fast path' which we
    // currently want to test is maybe taken.
    int elementsToValidate = -1;
    TensileLite::Client::SolveGemmCPU(contraction, inputs, elementsToValidate, tryFastPath);

    auto                                      end      = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution Time: " << duration.count() << " ms" << std::endl;

    if(validate)
    {
        std::cout << "Validating..." << std::endl;

        // Convert inputs to f32 for the golden reference comparison
        std::vector<float> aF32, bF32;

        if constexpr(isFP4)
        {
            aF32.resize(numA);
            for(size_t i = 0; i < numA; i++)
                aF32[i] = a[i / 2].getElement(i % 2);
            bF32.resize(numB);
            for(size_t i = 0; i < numB; i++)
                bF32[i] = b[i / 2].getElement(i % 2);
        }
        else
        {
            aF32.resize(numA);
            for(size_t i = 0; i < numA; i++)
                aF32[i] = static_cast<float>(a[i]);
            bF32.resize(numB);
            for(size_t i = 0; i < numB; i++)
                bF32[i] = static_cast<float>(b[i]);
        }

        std::vector<float> cF32(c.begin(), c.end());
        std::vector<float> dRef(d.size());

        // Run the golden reference
        columnMajorGemm(aF32.data(),
                        bF32.data(),
                        cF32.data(),
                        dRef.data(),
                        m,
                        n,
                        k,
                        transA,
                        transB,
                        (useScaleAB == "Scalar") ? alpha * scaleABuf[0] * scaleBBuf[0] : alpha,
                        beta,
                        useBias ? biasVec.data() : nullptr,
                        useScaleAlphaVec ? scaleAlphaVecBuf.data() : nullptr,
                        activation,
                        (useScaleAB == "Vector") ? scaleABuf.data() : nullptr,
                        (useScaleAB == "Vector") ? scaleBBuf.data() : nullptr,
                        factorDim,
#ifndef _WIN32
                        (isFP4 && mxBlock > 0) ? mxsa.data() : nullptr,
                        (isFP4 && mxBlock > 0) ? mxsb.data() : nullptr,
#else
                        nullptr,
                        nullptr,
#endif
                        mxBlock);

        // Compare results — FP4 with MX scales needs wider tolerance
        float tolerance = isFP4 ? 0.5f : 0.05f;

        bool  allClose = true;
        float maxDiff  = 0.0f;

        for(size_t i = 0; i < m * n; i++)
        {
            float valDut = static_cast<float>(d[i]);
            float valRef = dRef[i];
            float diff   = std::abs(valDut - valRef);

            if(diff > tolerance)
            {
                allClose = false;
                maxDiff  = std::max(maxDiff, diff);
                if(i < 10)
                {
                    std::cout << "Mismatch at " << i << ": observed=" << valDut
                              << " expected=" << valRef << " diff=" << diff << std::endl;
                }
            }
        }

        if(allClose)
        {
            std::cout << "PASSED! (max diff: " << maxDiff << ")" << std::endl;
        }
        else
        {
            std::cout << "FAILED! (max diff: " << maxDiff << ")" << std::endl;
            return 1;
        }
    }
    return 0;
}

int main(int argc, char* argv[])
{
    using namespace TensileLite;

    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "Produce help message")(
        "M", po::value<size_t>()->default_value(128), "Matrix M dimension")(
        "N", po::value<size_t>()->default_value(128), "Matrix N dimension")(
        "K", po::value<size_t>()->default_value(128), "Matrix K dimension")(
        "transA", po::value<bool>()->default_value(false), "Transpose A")(
        "transB", po::value<bool>()->default_value(false), "Transpose B")(
        "alpha", po::value<float>()->default_value(1.0f), "Alpha scalar")(
        "beta", po::value<float>()->default_value(0.0f), "Beta scalar")(
        "type", po::value<std::string>()->default_value("f32"), "Data type (f32, f16, bf16, f4)")(
        "validate", po::value<bool>()->default_value(true), "Run validation against ref")(
        "tryFastPath", po::value<bool>()->default_value(false), "Use optimized path")(
        "bias", po::value<bool>()->default_value(false), "Enable bias vector")(
        "activation", po::value<std::string>()->default_value("none"), "Activation (none, relu)")(
        "scaleAlphaVec", po::value<bool>()->default_value(false), "Enable per-row alpha scaling")(
        "factorDim", po::value<int>()->default_value(0), "ScaleAlphaVec dimension: 0=row(M), 1=col(N)")(
        "useScaleAB", po::value<std::string>()->default_value("none"), "ScaleAB mode (none, Scalar, Vector)")(
        "mxBlock", po::value<int>()->default_value(0), "MX block size for FP4 (0=no MX, must be power of 2)");

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error parsing options: " << ex.what() << std::endl;
        return 1;
    }

    if(vm.count("help"))
    {
        std::cout << desc << "\n";
        return 0;
    }

    size_t      m                = vm["M"].as<size_t>();
    size_t      n                = vm["N"].as<size_t>();
    size_t      k                = vm["K"].as<size_t>();
    bool        transA           = vm["transA"].as<bool>();
    bool        transB           = vm["transB"].as<bool>();
    float       alpha            = vm["alpha"].as<float>();
    float       beta             = vm["beta"].as<float>();
    std::string typeStr          = vm["type"].as<std::string>();
    bool        validate         = vm["validate"].as<bool>();
    bool        tryFastPath      = vm["tryFastPath"].as<bool>();
    bool        useBias          = vm["bias"].as<bool>();
    std::string activationStr    = vm["activation"].as<std::string>();
    bool        useScaleAlphaVec = vm["scaleAlphaVec"].as<bool>();
    int         factorDim        = vm["factorDim"].as<int>();
    std::string useScaleAB       = vm["useScaleAB"].as<std::string>();
    int         mxBlock          = vm["mxBlock"].as<int>();

    if(useScaleAB != "none" && useScaleAB != "Scalar" && useScaleAB != "Vector")
    {
        std::cerr << "Unknown useScaleAB mode: " << useScaleAB << std::endl;
        return 1;
    }

    if(factorDim != 0 && factorDim != 1)
    {
        std::cerr << "Invalid factorDim: " << factorDim << " (must be 0 or 1)" << std::endl;
        return 1;
    }

    ActivationType activation = ActivationType::None;
    if(activationStr == "relu")
        activation = ActivationType::Relu;
    else if(activationStr != "none")
    {
        std::cerr << "Unknown activation: " << activationStr << std::endl;
        return 1;
    }

    std::cout << "Running GEMM with: M=" << m << " N=" << n << " K=" << k << " Type=" << typeStr
              << " FastPath=" << tryFastPath << std::endl;

    try
    {
        if(typeStr == "f32")
        {
            return runGemm<float>(
                m, n, k, transA, transB, alpha, beta, validate, tryFastPath,
                useBias, activation, useScaleAlphaVec, useScaleAB, factorDim);
        }
        else if(typeStr == "bf16")
        {
            return runGemm<BFloat16>(
                m, n, k, transA, transB, alpha, beta, validate, tryFastPath,
                useBias, activation, useScaleAlphaVec, useScaleAB, factorDim);
        }
        else if(typeStr == "f16")
        {
            return runGemm<Half>(
                m, n, k, transA, transB, alpha, beta, validate, tryFastPath,
                useBias, activation, useScaleAlphaVec, useScaleAB, factorDim);
        }
#ifndef _WIN32
        else if(typeStr == "f4")
        {
            return runGemm<Float4x2>(
                m, n, k, transA, transB, alpha, beta, validate, tryFastPath,
                useBias, activation, useScaleAlphaVec, useScaleAB, factorDim, mxBlock);
        }
#endif
        else
        {
            std::cerr << "Unknown type: " << typeStr << std::endl;
            return 1;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
