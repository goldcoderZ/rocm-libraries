# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(GTest REQUIRED)

add_executable(tensilelite-tests)
add_subdirectory(configs)

set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${TEST_DATA_DIR}")

foreach(FILE ${SOLUTION_LIBRARY_FILES})
    if(EXISTS "${FILE}.gz")
        execute_process(COMMAND gzip --keep --decompress --force "${FILE}.gz")
    endif()
    file(COPY ${FILE} DESTINATION "${TEST_DATA_DIR}")
endforeach()

target_sources(tensilelite-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AlmostEqual_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ContractionSelectionLibrary_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataTypes_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Predicates_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TestData.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/GEMMKernelTest.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/GEMMKernelTest_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/RunGEMMKernelTest.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/RunGEMMKernelTest_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/TestData.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/TestUtils.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RangeLibrary_test.cpp"
)

target_include_directories(tensilelite-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(tensilelite-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../client/include")

target_link_libraries(tensilelite-tests
    PUBLIC
        tensilelite::tensilelite-host
    PRIVATE
        hip::device
        ${CMAKE_DL_LIBS}
)

if(TARGET roc::mxDataGenerator)
    target_sources(tensilelite-tests
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/MXDataGen_test.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../clients/common/src/mxDataGen.cpp"
    )
    target_include_directories(tensilelite-tests
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../../clients/common/include"
    )
    target_link_libraries(tensilelite-tests PRIVATE roc::mxDataGenerator)
    target_compile_definitions(tensilelite-tests PRIVATE HIPBLASLT_ENABLE_MXDATAGENERATOR)
    set_target_properties(tensilelite-tests
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
endif()

gtest_discover_tests(tensilelite-tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} TIMEOUT 60)

target_link_libraries(tensilelite-tests PUBLIC GTest::gtest)

if(HIPBLASLT_ENABLE_OPENMP)
    target_link_libraries(tensilelite-tests PRIVATE OpenMP::OpenMP_CXX)
endif()


enable_testing()

#################################################
#        Tests for cpu-gemm-driver              #
#################################################

function(add_cpu_gemm_test TEST_NAME)
    cmake_parse_arguments(ARG "" "" "ARGS" ${ARGN})

    add_test(
        NAME "CPUGemm.${TEST_NAME}"
        COMMAND cpu-gemm-driver ${ARG_ARGS}
    )
endfunction()

# Default test sizes. Non-tile-aligned to stress remainder handling.
# f32 and f16/bf16 use different sizes to increase coverage diversity.
set(F32_SIZES -M 133 -N 147 -K 165)
set(F16_SIZES -M 131 -N 141 -K 151)
set(ADDITIONAL_SIZES -M 102 -N 103 -K 105)

macro(get_test_sizes TYPE OUT_SIZES)
    if("${TYPE}" STREQUAL "f32")
        set(${OUT_SIZES} ${F32_SIZES})
    else()
        set(${OUT_SIZES} ${F16_SIZES})
    endif()
endmacro()

# Helper: convert transpose tag (NN/TN/NT/TT) to flags
macro(get_trans_flags TRANS OUT_FLAGS)
    set(${OUT_FLAGS} "")
    if("${TRANS}" MATCHES "^T")
        list(APPEND ${OUT_FLAGS} --transA)
    endif()
    if("${TRANS}" MATCHES "T$")
        list(APPEND ${OUT_FLAGS} --transB)
    endif()
endmacro()

# Run all tests against both fast and slow paths
foreach(PATH fast slow)
    if("${PATH}" STREQUAL "fast")
        set(PATH_FLAGS --tryFastPath)
    else()
        set(PATH_FLAGS "")
    endif()

    # Basic sanity, transpose combos, odd sizes, type coverage
    add_cpu_gemm_test(${PATH}_F32_Sanity
        ARGS ${F32_SIZES} --type f32 --validate ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_F32_ADDITIONAL_SIZES
        ARGS ${ADDITIONAL_SIZES} --type f32 --validate ${PATH_FLAGS}
    )
    foreach(TYPE f32 f16 bf16)
        get_test_sizes(${TYPE} SIZES)
        foreach(TRANS NN TN NT TT)
            get_trans_flags(${TRANS} TFLAGS)
            add_cpu_gemm_test(${PATH}_${TYPE}_${TRANS}
                ARGS ${SIZES} ${TFLAGS} --type ${TYPE} --validate ${PATH_FLAGS}
            )
        endforeach()
    endforeach()

    # Cross-validate: features per type
    foreach(TYPE f32 f16 bf16)
        get_test_sizes(${TYPE} SIZES)
        add_cpu_gemm_test(${PATH}_${TYPE}_Beta
            ARGS ${SIZES} --type ${TYPE} --beta 1.0 --validate ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_${TYPE}_Bias
            ARGS ${SIZES} --type ${TYPE} --validate --bias ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_${TYPE}_AllFeatures
            ARGS ${SIZES} --type ${TYPE} --validate --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_${TYPE}_TN_AllFeatures
            ARGS ${SIZES} --transA --type ${TYPE} --validate --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
        )
    endforeach()

    # F32-only individual feature tests
    add_cpu_gemm_test(${PATH}_f32_Relu
        ARGS ${F32_SIZES} --type f32 --validate --activation relu ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_f32_ScaleAlphaVec
        ARGS ${F32_SIZES} --type f32 --validate --scaleAlphaVec ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_f32_Beta_AllFeatures
        ARGS ${F32_SIZES} --type f32 --beta 0.5 --validate --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_f32_ScaleAlphaVec_ColN
        ARGS ${F32_SIZES} --type f32 --validate --scaleAlphaVec --factorDim 1 ${PATH_FLAGS}
    )

    # Edge cases: dimensions at or below tile size (BLOCK_M=32, BLOCK_N=32, BLOCK_K=8)
    foreach(CASE "M1;1;147;165" "N1;133;1;165" "K1;133;147;1" "1x1;1;1;1" "ExactBlock;32;32;8")
        list(GET CASE 0 NAME)
        list(GET CASE 1 EM)
        list(GET CASE 2 EN)
        list(GET CASE 3 EK)
        add_cpu_gemm_test(${PATH}_f32_${NAME}
            ARGS -M ${EM} -N ${EN} -K ${EK} --type f32 --validate ${PATH_FLAGS}
        )
    endforeach()

    # ScaleAB tests
    foreach(MODE Scalar Vector)
        foreach(TYPE f32 f16 bf16)
            get_test_sizes(${TYPE} SIZES)
            add_cpu_gemm_test(${PATH}_${TYPE}_ScaleAB_${MODE}
                ARGS ${SIZES} --type ${TYPE} --validate --useScaleAB ${MODE} ${PATH_FLAGS}
            )
        endforeach()
        add_cpu_gemm_test(${PATH}_f32_ScaleAB_${MODE}_AllFeatures
            ARGS ${F32_SIZES} --type f32 --beta 0.5 --validate --useScaleAB ${MODE} --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
        )
        # ScaleAB with factorDim=1 (column-dimension ScaleAlphaVec)
        add_cpu_gemm_test(${PATH}_f32_ScaleAB_${MODE}_ColN
            ARGS ${F32_SIZES} --type f32 --validate --useScaleAB ${MODE} --scaleAlphaVec --factorDim 1 ${PATH_FLAGS}
        )
    endforeach()
endforeach()

#################################################
#        FP4 / MX Scale tests (fast & slow)     #
#################################################
if(NOT WIN32)
    # M/N similar to f32 tests; K must be a multiple of mxBlock (32).
    set(F4_SIZES_32 -M 133 -N 147 -K 160)
    set(F4_SIZES_64 -M 133 -N 147 -K 192)

    foreach(PATH fast slow)
        if("${PATH}" STREQUAL "fast")
            set(PATH_FLAGS --tryFastPath)
        else()
            set(PATH_FLAGS "")
        endif()

        # Basic F4 with MX scaling, all transpose combos
        foreach(TRANS NN TN NT TT)
            get_trans_flags(${TRANS} TFLAGS)
            add_cpu_gemm_test(${PATH}_f4_MX32_${TRANS}
                ARGS ${F4_SIZES_32} ${TFLAGS} --type f4 --mxBlock 32 --validate ${PATH_FLAGS}
            )
            add_cpu_gemm_test(${PATH}_f4_MX32_K192_${TRANS}
                ARGS ${F4_SIZES_64} ${TFLAGS} --type f4 --mxBlock 32 --validate ${PATH_FLAGS}
            )
        endforeach()

        # F4 without MX (mxBlock=0, the default)
        add_cpu_gemm_test(${PATH}_f4_NoMX_NN
            ARGS -M 16 -N 16 -K 16 --type f4 --validate ${PATH_FLAGS}
        )

        # F4 + MX + epilogue features
        add_cpu_gemm_test(${PATH}_f4_MX32_Bias
            ARGS ${F4_SIZES_32} --type f4 --mxBlock 32 --validate --bias ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_f4_MX32_Relu
            ARGS ${F4_SIZES_32} --type f4 --mxBlock 32 --validate --activation relu ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_f4_MX32_ScaleAlphaVec
            ARGS ${F4_SIZES_32} --type f4 --mxBlock 32 --validate --scaleAlphaVec ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_f4_MX32_AllFeatures
            ARGS ${F4_SIZES_32} --type f4 --mxBlock 32 --validate --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_f4_MX32_Beta
            ARGS ${F4_SIZES_32} --type f4 --mxBlock 32 --beta 1.0 --validate ${PATH_FLAGS}
        )

        # F4 + MX + ScaleAB
        foreach(MODE Scalar Vector)
            add_cpu_gemm_test(${PATH}_f4_MX32_ScaleAB_${MODE}
                ARGS ${F4_SIZES_32} --type f4 --mxBlock 32 --validate --useScaleAB ${MODE} ${PATH_FLAGS}
            )
        endforeach()
    endforeach()

    # Fast-path-only: mxBlock=16 (tests non-hardcoded mxBlock; 16 % BLOCK_K=8 == 0)
    add_cpu_gemm_test(fast_f4_MX16_NN
        ARGS ${F4_SIZES_32} --type f4 --mxBlock 16 --validate --tryFastPath
    )

    # Negative tests: invalid K/mxBlock combos should fail with clear error
    add_test(
        NAME "CPUGemm.slow_f4_MX32_BadK_Unaligned"
        COMMAND cpu-gemm-driver -M 16 -N 16 -K 33 --type f4 --mxBlock 32 --validate
    )
    set_tests_properties("CPUGemm.slow_f4_MX32_BadK_Unaligned" PROPERTIES WILL_FAIL TRUE)

    add_test(
        NAME "CPUGemm.slow_f4_MX32_BadK_TooSmall"
        COMMAND cpu-gemm-driver -M 16 -N 16 -K 16 --type f4 --mxBlock 32 --validate
    )
    set_tests_properties("CPUGemm.slow_f4_MX32_BadK_TooSmall" PROPERTIES WILL_FAIL TRUE)
endif()
