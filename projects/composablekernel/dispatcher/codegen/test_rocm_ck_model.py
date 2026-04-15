# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for rocm_ck_model — mapping tables, translation, and serialization.

Run: pytest test_rocm_ck_model.py -v
"""

from rocm_ck_model import (
    DISPATCHER_ACC_DTYPE,
    DISPATCHER_DTYPE,
    DISPATCHER_LAYOUT,
    DISPATCHER_OUTPUT_DTYPE,
    DISPATCHER_PIPELINE,
    DISPATCHER_SCHEDULER,
    DISPATCHER_STORE_STRATEGY,
    DISPATCHER_TARGET_SET,
    AddOp,
    DataType,
    Dim3,
    EpilogueOp,
    GemmAlgorithm,
    GemmOp,
    GemmSpec,
    Layout,
    PhysicalTensor,
    Pipeline,
    PipelineScheduler,
    ReluOp,
    Signature,
    StoreStrategy,
    Tensor,
    TilePartitioner,
    to_hip_source,
    translate_kernel_config,
)

# Import dispatcher types for translation tests
from unified_gemm_codegen import (
    GemmVariant,
    KernelConfig,
    TileConfig,
    TraitConfig,
)


# ============================================================================
# Helpers — build dispatcher KernelConfig fixtures
# ============================================================================


def _tile(m=128, n=128, k=32, wm=2, wn=2, wk=1, wtm=32, wtn=32, wtk=16):
    return TileConfig(
        tile_m=m,
        tile_n=n,
        tile_k=k,
        warp_m=wm,
        warp_n=wn,
        warp_k=wk,
        warp_tile_m=wtm,
        warp_tile_n=wtn,
        warp_tile_k=wtk,
    )


def _trait(
    pipeline="compv3",
    epilogue="cshuffle",
    scheduler="intrawave",
    pad_m=False,
    pad_n=False,
    pad_k=False,
    persistent=False,
):
    return TraitConfig(
        pipeline=pipeline,
        epilogue=epilogue,
        scheduler=scheduler,
        pad_m=pad_m,
        pad_n=pad_n,
        pad_k=pad_k,
        persistent=persistent,
    )


def _config(tile=None, trait=None, variant=GemmVariant.STANDARD, **kwargs):
    return KernelConfig(
        tile=tile or _tile(), trait=trait or _trait(), variant=variant, **kwargs
    )


# ============================================================================
# Mapping table tests
# ============================================================================


class TestMappingTables:
    """Every dispatcher string key maps to a valid enum value."""

    def test_dtype_keys(self):
        for key, val in DISPATCHER_DTYPE.items():
            assert isinstance(val, DataType), f"DTYPE[{key}] is not DataType"

    def test_acc_dtype_keys(self):
        for key, val in DISPATCHER_ACC_DTYPE.items():
            assert isinstance(val, DataType), f"ACC_DTYPE[{key}] is not DataType"

    def test_output_dtype_keys(self):
        for key, val in DISPATCHER_OUTPUT_DTYPE.items():
            assert isinstance(val, DataType), f"OUTPUT_DTYPE[{key}] is not DataType"

    def test_layout_keys(self):
        for key, val in DISPATCHER_LAYOUT.items():
            assert isinstance(val, Layout), f"LAYOUT[{key}] is not Layout"

    def test_pipeline_keys(self):
        for key, val in DISPATCHER_PIPELINE.items():
            assert isinstance(val, Pipeline), f"PIPELINE[{key}] is not Pipeline"

    def test_scheduler_keys(self):
        for key, val in DISPATCHER_SCHEDULER.items():
            assert isinstance(val, PipelineScheduler)

    def test_store_strategy_keys(self):
        for key, val in DISPATCHER_STORE_STRATEGY.items():
            assert isinstance(val, StoreStrategy)

    def test_dtype_and_acc_dtype_have_same_keys(self):
        assert set(DISPATCHER_DTYPE.keys()) == set(DISPATCHER_ACC_DTYPE.keys())

    def test_dtype_and_output_dtype_have_same_keys(self):
        assert set(DISPATCHER_DTYPE.keys()) == set(DISPATCHER_OUTPUT_DTYPE.keys())

    def test_dtype_and_target_set_have_same_keys(self):
        assert set(DISPATCHER_DTYPE.keys()) == set(DISPATCHER_TARGET_SET.keys())

    # Spot-check specific mappings
    def test_fp16_maps(self):
        assert DISPATCHER_DTYPE["fp16"] == DataType.FP16
        assert DISPATCHER_ACC_DTYPE["fp16"] == DataType.FP32
        assert DISPATCHER_OUTPUT_DTYPE["fp16"] == DataType.FP16

    def test_int8_maps(self):
        assert DISPATCHER_DTYPE["int8"] == DataType.I8
        assert DISPATCHER_ACC_DTYPE["int8"] == DataType.I32
        assert DISPATCHER_OUTPUT_DTYPE["int8"] == DataType.I32

    def test_fp8_maps(self):
        assert DISPATCHER_DTYPE["fp8"] == DataType.FP8_FNUZ
        assert DISPATCHER_OUTPUT_DTYPE["fp8"] == DataType.FP16

    def test_pipeline_mem(self):
        assert DISPATCHER_PIPELINE["mem"] == Pipeline.Memory

    def test_store_strategy_default_is_direct2d(self):
        assert DISPATCHER_STORE_STRATEGY["default"] == StoreStrategy.Direct2D


# ============================================================================
# Translation tests
# ============================================================================


class TestTranslation:
    """translate_kernel_config produces correct dataclass values."""

    def test_fp16_rcr(self):
        config = _config(tile=_tile(256, 128, 32))
        sig, algo, spec = translate_kernel_config(config, "fp16", "rcr", ["gfx90a"])

        assert sig.dtype == DataType.FP16
        assert len(sig.tensors) == 0  # output dtype == input dtype
        assert len(sig.ops) == 1
        assert isinstance(sig.ops[0], GemmOp)
        assert sig.ops[0].acc_dtype == DataType.FP32

        assert algo.pipeline == Pipeline.V3
        assert algo.block_tile == Dim3(256, 128, 32)
        assert algo.store_strategy == StoreStrategy.CShuffle

        assert len(spec.physical_tensors) == 3
        assert spec.physical_tensors[0].name == "A"
        assert spec.physical_tensors[0].layout == Layout.Row
        assert spec.physical_tensors[1].name == "B"
        assert spec.physical_tensors[1].layout == Layout.Col
        assert spec.physical_tensors[2].name == "C"
        assert spec.physical_tensors[2].layout == Layout.Row
        assert spec.acc_dtype == DataType.FP32
        assert spec.workgroup_size == 256

    def test_int8_output_dtype(self):
        config = _config()
        sig, algo, spec = translate_kernel_config(config, "int8", "rcr", ["gfx942"])

        assert sig.dtype == DataType.I8
        assert len(sig.tensors) == 1
        assert sig.tensors[0].name == "C"
        assert sig.tensors[0].dtype == DataType.I32
        assert sig.ops[0].acc_dtype == DataType.I32
        assert spec.physical_tensors[2].dtype == DataType.I32

    def test_fp8_output_dtype(self):
        config = _config()
        sig, algo, spec = translate_kernel_config(config, "fp8", "rcr", ["gfx942"])

        assert sig.dtype == DataType.FP8_FNUZ
        assert sig.tensors[0].dtype == DataType.FP16
        assert spec.physical_tensors[2].dtype == DataType.FP16

    def test_multi_d_add_relu(self):
        config = _config(
            variant=GemmVariant.MULTI_D,
            elementwise_op="Relu",
            num_d_tensors=1,
            d_layout="r",
        )
        sig, algo, spec = translate_kernel_config(config, "fp16", "rcr", [])

        # Op chain: GemmOp -> AddOp -> ReluOp
        assert len(sig.ops) == 3
        assert isinstance(sig.ops[0], GemmOp)
        assert isinstance(sig.ops[1], AddOp)
        assert sig.ops[1].rhs == "D0"
        assert isinstance(sig.ops[2], ReluOp)
        assert sig.ops[2].in_ == "D"
        assert sig.ops[2].out == "E"

        # Physical tensors: A, B, E (output), D0
        assert len(spec.physical_tensors) == 4
        assert spec.physical_tensors[2].name == "E"  # final output
        assert spec.physical_tensors[3].name == "D0"

        # Epilogue ops
        assert spec.epilogue_ops == [EpilogueOp.Add, EpilogueOp.Relu]

    def test_preshuffle_pipeline(self):
        config = _config(
            trait=_trait(pipeline="preshufflev2"),
            preshuffle=True,
        )
        sig, algo, spec = translate_kernel_config(config, "fp16", "rcr", [])
        assert algo.pipeline == Pipeline.Preshuffle
        assert spec.pipeline == Pipeline.Preshuffle

    def test_memory_interwave(self):
        config = _config(trait=_trait(pipeline="mem", scheduler="interwave"))
        sig, algo, spec = translate_kernel_config(config, "fp16", "rcr", [])
        assert algo.pipeline == Pipeline.Memory
        assert algo.pipeline_scheduler == PipelineScheduler.Interwave

    def test_direct2d(self):
        config = _config(trait=_trait(epilogue="default"))
        sig, algo, spec = translate_kernel_config(config, "fp16", "rcr", [])
        assert algo.store_strategy == StoreStrategy.Direct2D

    def test_pad_flags(self):
        config = _config(trait=_trait(pad_m=True, pad_n=True))
        sig, algo, spec = translate_kernel_config(config, "fp16", "rcr", [])
        assert algo.pad_m is True
        assert algo.pad_n is True
        assert spec.pad_m is True
        assert spec.pad_n is True


# ============================================================================
# Serialization tests — field-level assertions
# ============================================================================


class TestSpecJsonSerialization:
    """GemmSpec.to_spec_json produces correct structure."""

    def test_basic_structure(self):
        spec = GemmSpec(
            physical_tensors=[
                PhysicalTensor("A", DataType.FP16, Layout.Row, 0),
                PhysicalTensor("B", DataType.FP16, Layout.Col, 1),
                PhysicalTensor("C", DataType.FP16, Layout.Row, 2),
            ],
            acc_dtype=DataType.FP32,
            block_tile=Dim3(256, 128, 32),
            block_waves=Dim3(2, 2, 1),
            wave_tile=Dim3(32, 32, 16),
            workgroup_size=256,
            k_batch=1,
            pipeline=Pipeline.V3,
            pipeline_scheduler=PipelineScheduler.Intrawave,
            tile_partitioner=TilePartitioner.Linear,
            epilogue_ops=[],
            store_strategy=StoreStrategy.CShuffle,
            pad_m=False,
            pad_n=False,
        )
        result = spec.to_spec_json("test_kernel", ["gfx90a"])

        assert result["name"] == "test_kernel"
        assert result["spec_type"] == "GemmSpec"
        assert result["targets"] == ["gfx90a"]

        s = result["spec"]
        assert s["acc_dtype"] == "FP32"
        assert s["pipeline"] == "V3"
        assert s["pipeline_scheduler"] == "Intrawave"
        assert s["tile_partitioner"] == "Linear"
        assert s["store_strategy"] == "CShuffle"
        assert s["workgroup_size"] == 256
        assert s["epilogue_ops"] == []
        assert s["pad_m"] is False
        assert s["group_size"] == 0

    def test_physical_tensors(self):
        spec = GemmSpec(
            physical_tensors=[
                PhysicalTensor("A", DataType.FP16, Layout.Row, 0),
                PhysicalTensor("B", DataType.FP16, Layout.Col, 1),
                PhysicalTensor("C", DataType.FP16, Layout.Row, 2),
            ],
            acc_dtype=DataType.FP32,
            block_tile=Dim3(128, 128, 32),
            block_waves=Dim3(2, 2, 1),
            wave_tile=Dim3(32, 32, 16),
            workgroup_size=256,
            k_batch=1,
            pipeline=Pipeline.V3,
            pipeline_scheduler=PipelineScheduler.Intrawave,
            tile_partitioner=TilePartitioner.Linear,
            epilogue_ops=[],
            store_strategy=StoreStrategy.CShuffle,
            pad_m=False,
            pad_n=False,
        )
        tensors = spec.to_spec_json("k", [])["spec"]["physical_tensors"]
        assert len(tensors) == 3
        assert tensors[0] == {
            "name": "A",
            "dtype": "FP16",
            "layout": "Row",
            "args_slot": 0,
        }
        assert tensors[1] == {
            "name": "B",
            "dtype": "FP16",
            "layout": "Col",
            "args_slot": 1,
        }

    def test_epilogue_ops(self):
        spec = GemmSpec(
            physical_tensors=[
                PhysicalTensor("A", DataType.FP16, Layout.Row, 0),
                PhysicalTensor("B", DataType.FP16, Layout.Col, 1),
                PhysicalTensor("E", DataType.FP16, Layout.Row, 2),
                PhysicalTensor("D0", DataType.FP16, Layout.Row, 3),
            ],
            acc_dtype=DataType.FP32,
            block_tile=Dim3(128, 128, 32),
            block_waves=Dim3(2, 2, 1),
            wave_tile=Dim3(32, 32, 16),
            workgroup_size=256,
            k_batch=1,
            pipeline=Pipeline.V3,
            pipeline_scheduler=PipelineScheduler.Intrawave,
            tile_partitioner=TilePartitioner.Linear,
            epilogue_ops=[EpilogueOp.Add, EpilogueOp.Relu],
            store_strategy=StoreStrategy.CShuffle,
            pad_m=False,
            pad_n=False,
        )
        result = spec.to_spec_json("k", [])["spec"]
        assert result["epilogue_ops"] == ["Add", "Relu"]


class TestHipSerialization:
    """to_hip_source produces correct C++ code."""

    def test_contains_signature(self):
        sig = Signature(dtype=DataType.FP16, ops=[GemmOp("A", "B", "C")])
        algo = GemmAlgorithm(
            block_tile=Dim3(128, 128, 32),
            block_waves=Dim3(2, 2, 1),
            wave_tile=Dim3(32, 32, 16),
            pipeline=Pipeline.V3,
        )
        hip = to_hip_source("test_k", sig, algo, "cdna()")
        assert "rocm_ck::Signature{" in hip
        assert "rocm_ck::GemmAlgorithm{" in hip
        assert "rocm_ck::makeSpec(" in hip

    def test_pipeline_field(self):
        sig = Signature(dtype=DataType.FP16, ops=[GemmOp("A", "B", "C")])
        algo = GemmAlgorithm(
            block_tile=Dim3(128, 128, 32),
            block_waves=Dim3(2, 2, 1),
            wave_tile=Dim3(32, 32, 16),
            pipeline=Pipeline.V3,
        )
        hip = to_hip_source("test_k", sig, algo, "cdna()")
        assert ".pipeline    = rocm_ck::Pipeline::V3" in hip

    def test_default_pipeline_omitted(self):
        sig = Signature(dtype=DataType.FP16, ops=[GemmOp("A", "B", "C")])
        algo = GemmAlgorithm(
            block_tile=Dim3(128, 128, 32),
            block_waves=Dim3(2, 2, 1),
            wave_tile=Dim3(32, 32, 16),
            pipeline=Pipeline.V1,
        )
        hip = to_hip_source("test_k", sig, algo, "cdna()")
        assert "Pipeline::" not in hip

    def test_interwave_scheduler(self):
        sig = Signature(dtype=DataType.FP16, ops=[GemmOp("A", "B", "C")])
        algo = GemmAlgorithm(
            block_tile=Dim3(128, 128, 32),
            block_waves=Dim3(2, 2, 1),
            wave_tile=Dim3(32, 32, 16),
            pipeline=Pipeline.Memory,
            pipeline_scheduler=PipelineScheduler.Interwave,
        )
        hip = to_hip_source("test_k", sig, algo, "cdna()")
        assert "PipelineScheduler::Interwave" in hip

    def test_tensor_override_for_int8(self):
        sig = Signature(
            dtype=DataType.I8,
            tensors=[Tensor(name="C", dtype=DataType.I32)],
            ops=[GemmOp("A", "B", "C", acc_dtype=DataType.I32)],
        )
        algo = GemmAlgorithm(
            block_tile=Dim3(128, 128, 32),
            block_waves=Dim3(2, 2, 1),
            wave_tile=Dim3(32, 32, 16),
            pipeline=Pipeline.V3,
        )
        hip = to_hip_source("test_k", sig, algo, "family_gfx94()")
        assert (
            '.tensors = {rocm_ck::Tensor{.name = "C", .dtype = rocm_ck::DataType::I32}}'
            in hip
        )
        assert ".acc_dtype = rocm_ck::DataType::I32" in hip

    def test_kernel_function(self):
        sig = Signature(dtype=DataType.FP16, ops=[GemmOp("A", "B", "C")])
        algo = GemmAlgorithm(
            block_tile=Dim3(128, 128, 32),
            block_waves=Dim3(2, 2, 1),
            wave_tile=Dim3(32, 32, 16),
        )
        hip = to_hip_source("my_kernel", sig, algo, "cdna()")
        assert 'extern "C" __global__ void my_kernel(rocm_ck::Args args)' in hip

    def test_multi_d_ops(self):
        sig = Signature(
            dtype=DataType.FP16,
            ops=[
                GemmOp("A", "B", "C"),
                AddOp("C", "D0", "D"),
                ReluOp(in_="D", out="E"),
            ],
        )
        algo = GemmAlgorithm(
            block_tile=Dim3(128, 128, 32),
            block_waves=Dim3(2, 2, 1),
            wave_tile=Dim3(32, 32, 16),
            pipeline=Pipeline.V3,
            pad_m=True,
            pad_n=True,
        )
        hip = to_hip_source("test_k", sig, algo, "cdna()")
        assert 'rocm_ck::AddOp{.lhs = "C", .rhs = "D0", .out = "D"}' in hip
        assert 'rocm_ck::ReluOp{.in = "D", .out = "E"}' in hip
        assert ".pad_m = true" in hip
        assert ".pad_n = true" in hip


# ============================================================================
# Round-trip tests: translate → serialize → verify structure
# ============================================================================


class TestRoundTrip:
    """End-to-end: translate + serialize produces correct output."""

    def _check_hip(self, config, datatype, layout, kernel_name):
        """Verify .hip output matches expected C++ structure."""
        sig, algo, spec = translate_kernel_config(config, datatype, layout, [])
        target_set = DISPATCHER_TARGET_SET[datatype]
        hip = to_hip_source(kernel_name, sig, algo, target_set)

        assert "rocm_ck::makeSpec(" in hip
        assert f'extern "C" __global__ void {kernel_name}(' in hip
        assert f'static constexpr const char* kName = "{kernel_name}"' in hip
        return hip

    def _check_spec(self, config, datatype, layout, kernel_name, targets):
        """Verify .spec.json output matches expected structure."""
        sig, algo, spec = translate_kernel_config(config, datatype, layout, targets)
        result = spec.to_spec_json(kernel_name, targets)

        assert result["name"] == kernel_name
        assert result["spec_type"] == "GemmSpec"
        assert result["targets"] == targets
        return result

    def test_fp16_rcr_hip(self):
        hip = self._check_hip(_config(tile=_tile(256, 128, 32)), "fp16", "rcr", "k")
        assert "rocm_ck::DataType::FP16" in hip
        assert ".block_tile  = {256, 128, 32}" in hip
        assert "Pipeline::V3" in hip

    def test_fp16_rcr_spec(self):
        result = self._check_spec(
            _config(tile=_tile(256, 128, 32)), "fp16", "rcr", "k", ["gfx90a", "gfx942"]
        )
        s = result["spec"]
        assert s["physical_tensors"][0]["name"] == "A"
        assert s["physical_tensors"][2]["name"] == "C"
        assert s["acc_dtype"] == "FP32"
        assert s["pipeline"] == "V3"

    def test_int8_hip(self):
        hip = self._check_hip(_config(tile=_tile(256, 128, 64)), "int8", "rcr", "k")
        assert "DataType::I8" in hip
        assert "DataType::I32" in hip  # acc_dtype and output tensor override

    def test_int8_spec(self):
        result = self._check_spec(
            _config(tile=_tile(256, 128, 64)), "int8", "rcr", "k", ["gfx942"]
        )
        s = result["spec"]
        assert s["physical_tensors"][2]["dtype"] == "I32"
        assert s["acc_dtype"] == "I32"

    def test_fp8_hip(self):
        hip = self._check_hip(_config(), "fp8", "rcr", "k")
        assert "DataType::FP8_FNUZ" in hip
        assert "DataType::FP16" in hip  # output tensor override

    def test_fp8_spec(self):
        result = self._check_spec(_config(), "fp8", "rcr", "k", ["gfx942"])
        s = result["spec"]
        assert s["physical_tensors"][0]["dtype"] == "FP8_FNUZ"
        assert s["physical_tensors"][2]["dtype"] == "FP16"

    def test_multi_d_add_relu_hip(self):
        config = _config(
            variant=GemmVariant.MULTI_D,
            elementwise_op="Relu",
            num_d_tensors=1,
            d_layout="r",
            trait=_trait(pad_m=True, pad_n=True),
        )
        hip = self._check_hip(config, "fp16", "rcr", "k")
        assert 'AddOp{.lhs = "C", .rhs = "D0", .out = "D"}' in hip
        assert 'ReluOp{.in = "D", .out = "E"}' in hip
        assert ".pad_m = true" in hip

    def test_multi_d_add_relu_spec(self):
        config = _config(
            variant=GemmVariant.MULTI_D,
            elementwise_op="Relu",
            num_d_tensors=1,
            d_layout="r",
            trait=_trait(pad_m=True, pad_n=True),
        )
        result = self._check_spec(config, "fp16", "rcr", "k", ["gfx90a"])
        s = result["spec"]
        # Output is "E" — chain-derived name, matching C++ makeSpec
        assert s["physical_tensors"][2]["name"] == "E"
        assert s["physical_tensors"][3]["name"] == "D0"
        assert s["epilogue_ops"] == ["Add", "Relu"]

    def test_memory_interwave_hip(self):
        config = _config(trait=_trait(pipeline="mem", scheduler="interwave"))
        hip = self._check_hip(config, "fp16", "rcr", "k")
        assert "Pipeline::Memory" in hip
        assert "PipelineScheduler::Interwave" in hip

    def test_direct2d_hip(self):
        config = _config(trait=_trait(epilogue="default"))
        hip = self._check_hip(config, "fp16", "rcr", "k")
        assert "StoreStrategy::Direct2D" in hip

    def test_bf16_rrr_spec(self):
        result = self._check_spec(_config(), "bf16", "rrr", "k", ["gfx90a"])
        s = result["spec"]
        assert s["physical_tensors"][0]["layout"] == "Row"
        assert s["physical_tensors"][1]["layout"] == "Row"
        assert s["physical_tensors"][2]["layout"] == "Row"

    def test_v4_pipeline_hip(self):
        config = _config(trait=_trait(pipeline="compv4"))
        hip = self._check_hip(config, "fp16", "rcr", "k")
        assert "Pipeline::V4" in hip
