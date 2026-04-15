# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Python model of rocm_ck structural types.

Mirrors the C++ types in rocm_ck/include/rocm_ck/ — Signature, GemmAlgorithm,
GemmSpec, and the operator/enum types they depend on. Field names match C++.

Three sections:
  1. Dataclasses — dumb data, no validation (validation is in C++ makeSpec)
  2. Mapping tables — dispatcher codegen strings → rocm_ck enum values
  3. Translation — dispatcher KernelConfig → rocm_ck typed model
  4. Serialization — typed model → .hip C++ source / .spec.json dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


# ============================================================================
# Enums (mirror C++ enums)
# ============================================================================

# gemm_spec.hpp, datatype_utils.hpp, ops.hpp, layout.hpp


class DataType(Enum):
    """Mirror of rocm_ck::DataType (datatype_utils.hpp)."""

    FP64 = "FP64"
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    FP8_FNUZ = "FP8_FNUZ"
    BF8_FNUZ = "BF8_FNUZ"
    FP8_OCP = "FP8_OCP"
    BF8_OCP = "BF8_OCP"
    I4 = "I4"
    I8 = "I8"
    I16 = "I16"
    I32 = "I32"
    I64 = "I64"
    U8 = "U8"
    U16 = "U16"
    U32 = "U32"
    U64 = "U64"


class Layout(Enum):
    """Mirror of rocm_ck::Layout (layout.hpp)."""

    Row = "Row"
    Col = "Col"


class Pipeline(Enum):
    """Mirror of rocm_ck::Pipeline (gemm_spec.hpp)."""

    V1 = "V1"
    V3 = "V3"
    V4 = "V4"
    Memory = "Memory"
    Preshuffle = "Preshuffle"


class PipelineScheduler(Enum):
    """Mirror of rocm_ck::PipelineScheduler (gemm_spec.hpp)."""

    Intrawave = "Intrawave"
    Interwave = "Interwave"


class TilePartitioner(Enum):
    """Mirror of rocm_ck::TilePartitioner (gemm_spec.hpp)."""

    Direct = "Direct"
    Linear = "Linear"
    SpatiallyLocal = "SpatiallyLocal"
    StreamK = "StreamK"


class StoreStrategy(Enum):
    """Mirror of rocm_ck::StoreStrategy (gemm_spec.hpp)."""

    CShuffle = "CShuffle"
    Direct2D = "Direct2D"


class EpilogueOp(Enum):
    """Mirror of rocm_ck::EpilogueOp (gemm_spec.hpp)."""

    Add = "Add"
    Mul = "Mul"
    Relu = "Relu"
    FastGelu = "FastGelu"
    Gelu = "Gelu"
    Silu = "Silu"
    Sigmoid = "Sigmoid"


# ============================================================================
# Structural types (mirror C++ structs)
# ============================================================================


@dataclass
class Dim3:
    """Mirror of rocm_ck::Dim3 (gemm_spec.hpp)."""

    m: int
    n: int
    k: int


@dataclass
class PhysicalTensor:
    """Mirror of rocm_ck::PhysicalTensor (physical_tensor.hpp)."""

    name: str
    dtype: DataType
    layout: Layout
    args_slot: int


# ============================================================================
# Op types (mirror C++ ops.hpp)
# ============================================================================


@dataclass
class GemmOp:
    """Mirror of rocm_ck::GemmOp."""

    lhs: str
    rhs: str
    out: str
    acc_dtype: DataType = DataType.FP32


@dataclass
class AddOp:
    """Mirror of rocm_ck::AddOp."""

    lhs: str
    rhs: str
    out: str


@dataclass
class MulOp:
    """Mirror of rocm_ck::MulOp."""

    lhs: str
    rhs: str
    out: str


@dataclass
class ReluOp:
    """Mirror of rocm_ck::ReluOp."""

    in_: str  # `in` is a Python keyword
    out: str


@dataclass
class FastGeluOp:
    """Mirror of rocm_ck::FastGeluOp."""

    in_: str
    out: str


@dataclass
class GeluOp:
    """Mirror of rocm_ck::GeluOp."""

    in_: str
    out: str


@dataclass
class SiluOp:
    """Mirror of rocm_ck::SiluOp."""

    in_: str
    out: str


@dataclass
class SigmoidOp:
    """Mirror of rocm_ck::SigmoidOp."""

    in_: str
    out: str


Op = Union[GemmOp, AddOp, MulOp, ReluOp, FastGeluOp, GeluOp, SiluOp, SigmoidOp]

# Map from Op type to the EpilogueOp enum value (for ops that appear in epilogue chain)
_EPILOGUE_OP_MAP: dict[type, EpilogueOp] = {
    AddOp: EpilogueOp.Add,
    MulOp: EpilogueOp.Mul,
    ReluOp: EpilogueOp.Relu,
    FastGeluOp: EpilogueOp.FastGelu,
    GeluOp: EpilogueOp.Gelu,
    SiluOp: EpilogueOp.Silu,
    SigmoidOp: EpilogueOp.Sigmoid,
}


# ============================================================================
# Signature and Algorithm (mirror C++ structs)
# ============================================================================


@dataclass
class Tensor:
    """Mirror of rocm_ck::Tensor (signature.hpp).

    Tensors in the signature graph. Only tensors that differ from the
    default dtype need to be listed explicitly.
    """

    name: str
    dtype: DataType | None = None


@dataclass
class Signature:
    """Mirror of rocm_ck::Signature (signature.hpp).

    Describes WHAT a kernel computes — dtype, explicit tensors, op graph.
    """

    dtype: DataType
    tensors: list[Tensor] = field(default_factory=list)
    ops: list[Op] = field(default_factory=list)


@dataclass
class GemmAlgorithm:
    """Mirror of rocm_ck::GemmAlgorithm (gemm_spec.hpp).

    Describes HOW a GEMM executes — tile geometry, pipeline, partitioning.
    """

    block_tile: Dim3
    block_waves: Dim3
    wave_tile: Dim3
    k_batch: int = 1
    pipeline: Pipeline = Pipeline.V1
    pipeline_scheduler: PipelineScheduler = PipelineScheduler.Intrawave
    tile_partitioner: TilePartitioner = TilePartitioner.Linear
    store_strategy: StoreStrategy = StoreStrategy.CShuffle
    pad_m: bool = False
    pad_n: bool = False


@dataclass
class GemmSpec:
    """Mirror of rocm_ck::GemmSpec (gemm_spec.hpp).

    Fully resolved kernel descriptor — the output of makeSpec().
    """

    physical_tensors: list[PhysicalTensor]
    acc_dtype: DataType
    block_tile: Dim3
    block_waves: Dim3
    wave_tile: Dim3
    workgroup_size: int
    k_batch: int
    pipeline: Pipeline
    pipeline_scheduler: PipelineScheduler
    tile_partitioner: TilePartitioner
    epilogue_ops: list[EpilogueOp]
    store_strategy: StoreStrategy
    pad_m: bool
    pad_n: bool
    group_size: int = 0

    def to_spec_json(self, name: str, targets: list[str]) -> dict:
        """Serialize to the JSON structure pack.py expects."""
        return {
            "name": name,
            "spec_type": "GemmSpec",
            "targets": targets,
            "spec": {
                "physical_tensors": [
                    {
                        "name": pt.name,
                        "dtype": pt.dtype.value,
                        "layout": pt.layout.value,
                        "args_slot": pt.args_slot,
                    }
                    for pt in self.physical_tensors
                ],
                "acc_dtype": self.acc_dtype.value,
                "block_tile": {
                    "m": self.block_tile.m,
                    "n": self.block_tile.n,
                    "k": self.block_tile.k,
                },
                "block_waves": {
                    "m": self.block_waves.m,
                    "n": self.block_waves.n,
                    "k": self.block_waves.k,
                },
                "wave_tile": {
                    "m": self.wave_tile.m,
                    "n": self.wave_tile.n,
                    "k": self.wave_tile.k,
                },
                "workgroup_size": self.workgroup_size,
                "k_batch": self.k_batch,
                "pipeline": self.pipeline.value,
                "pipeline_scheduler": self.pipeline_scheduler.value,
                "tile_partitioner": self.tile_partitioner.value,
                "epilogue_ops": [op.value for op in self.epilogue_ops],
                "store_strategy": self.store_strategy.value,
                "pad_m": self.pad_m,
                "pad_n": self.pad_n,
                "group_size": self.group_size,
            },
        }


# ============================================================================
# Mapping tables: dispatcher codegen strings → rocm_ck enum values
# ============================================================================

DISPATCHER_DTYPE = {
    "fp16": DataType.FP16,
    "bf16": DataType.BF16,
    "fp32": DataType.FP32,
    "fp64": DataType.FP64,
    "fp8": DataType.FP8_FNUZ,
    "bf8": DataType.BF8_FNUZ,
    "int8": DataType.I8,
    "int4": DataType.I4,
}

DISPATCHER_ACC_DTYPE = {
    "fp16": DataType.FP32,
    "bf16": DataType.FP32,
    "fp32": DataType.FP32,
    "fp64": DataType.FP64,
    "fp8": DataType.FP32,
    "bf8": DataType.FP32,
    "int8": DataType.I32,
    "int4": DataType.I32,
}

DISPATCHER_OUTPUT_DTYPE = {
    "fp16": DataType.FP16,
    "bf16": DataType.BF16,
    "fp32": DataType.FP32,
    "fp64": DataType.FP64,
    "fp8": DataType.FP16,
    "bf8": DataType.FP16,
    "int8": DataType.I32,
    "int4": DataType.I32,
}

DISPATCHER_LAYOUT = {
    "r": Layout.Row,
    "c": Layout.Col,
}

DISPATCHER_PIPELINE = {
    "compv1": Pipeline.V1,
    "compv3": Pipeline.V3,
    "compv4": Pipeline.V4,
    "mem": Pipeline.Memory,
    "preshufflev2": Pipeline.Preshuffle,
}

DISPATCHER_SCHEDULER = {
    "intrawave": PipelineScheduler.Intrawave,
    "interwave": PipelineScheduler.Interwave,
}

DISPATCHER_STORE_STRATEGY = {
    "cshuffle": StoreStrategy.CShuffle,
    "default": StoreStrategy.Direct2D,
}

DISPATCHER_TARGET_SET = {
    "fp16": "cdna()",
    "bf16": "cdna()",
    "fp32": "cdna()",
    "fp64": "cdna()",
    "fp8": "family_gfx94()",
    "bf8": "family_gfx94()",
    "int8": "family_gfx94()",
    "int4": "family_gfx94()",
}

# Dispatcher elementwise_op values → unary activation Op constructors
_UNARY_ACTIVATIONS: dict[str, type] = {
    "Relu": ReluOp,
    "Gelu": GeluOp,
    "FastGelu": FastGeluOp,
    "Silu": SiluOp,
    "Sigmoid": SigmoidOp,
}

# Dispatcher elementwise_op values → binary D-tensor Op constructors
_BINARY_D_OPS: dict[str, type] = {
    "MultiDAdd": AddOp,
    "MultiDMultiply": MulOp,
}


# ============================================================================
# Translation: dispatcher KernelConfig → rocm_ck typed model
# ============================================================================


def _build_ops(
    elementwise_op: str, num_d_tensors: int, acc_dtype: DataType
) -> list[Op]:
    """Build the typed op chain from dispatcher config.

    Maps elementwise_op + num_d_tensors to the rocm_ck op graph:
    - GemmOp is always first (A x B -> C)
    - Binary D-tensor ops (Add/Mul) fold D0, D1, ... into the chain
    - Unary activations (Relu, Gelu, etc.) apply after any D-tensor ops
    - Tensor names advance: C -> D -> E -> F -> ...
    """
    gemm = GemmOp(lhs="A", rhs="B", out="C", acc_dtype=acc_dtype)
    ops: list[Op] = [gemm]
    cur = "C"

    if elementwise_op in _BINARY_D_OPS:
        op_cls = _BINARY_D_OPS[elementwise_op]
        for i in range(num_d_tensors):
            nxt = chr(ord(cur) + 1)
            ops.append(op_cls(lhs=cur, rhs=f"D{i}", out=nxt))
            cur = nxt
    elif elementwise_op in _UNARY_ACTIVATIONS:
        # D tensor adds come first, then the activation
        for i in range(num_d_tensors):
            nxt = chr(ord(cur) + 1)
            ops.append(AddOp(lhs=cur, rhs=f"D{i}", out=nxt))
            cur = nxt
        nxt = chr(ord(cur) + 1)
        act_cls = _UNARY_ACTIVATIONS[elementwise_op]
        ops.append(act_cls(in_=cur, out=nxt))
        cur = nxt

    return ops


def _build_epilogue_ops(ops: list[Op]) -> list[EpilogueOp]:
    """Extract EpilogueOp sequence from the typed op chain (skip GemmOp)."""
    result = []
    for op in ops:
        if isinstance(op, GemmOp):
            continue
        epi = _EPILOGUE_OP_MAP.get(type(op))
        if epi is not None:
            result.append(epi)
    return result


def _find_output_name(ops: list[Op]) -> str:
    """Find the final output tensor name from the op chain."""
    last = ops[-1]
    if hasattr(last, "out"):
        return last.out
    raise ValueError(f"Last op {type(last).__name__} has no 'out' field")


def translate_kernel_config(config, datatype: str, layout: str, targets: list[str]):
    """Translate dispatcher KernelConfig into rocm_ck typed model.

    datatype and layout are external to KernelConfig — they come from
    the cartesian product in the generation loop, not from the config itself.

    Returns (Signature, GemmAlgorithm, GemmSpec).
    """
    input_dtype = DISPATCHER_DTYPE[datatype]
    acc_dtype = DISPATCHER_ACC_DTYPE[datatype]
    output_dtype = DISPATCHER_OUTPUT_DTYPE[datatype]

    # Build typed op chain
    ops = _build_ops(config.elementwise_op, config.num_d_tensors, acc_dtype)

    # Build explicit tensors list (only when output dtype differs from input)
    tensors: list[Tensor] = []
    if output_dtype != input_dtype:
        # Output tensor name comes from the op chain
        out_name = _find_output_name(ops)
        tensors.append(Tensor(name=out_name, dtype=output_dtype))

    sig = Signature(dtype=input_dtype, tensors=tensors, ops=ops)

    # GemmAlgorithm
    t = config.tile
    pipeline = DISPATCHER_PIPELINE[config.trait.pipeline]
    scheduler = DISPATCHER_SCHEDULER.get(
        config.trait.scheduler, PipelineScheduler.Intrawave
    )
    store_strategy = DISPATCHER_STORE_STRATEGY[config.trait.epilogue]

    algo = GemmAlgorithm(
        block_tile=Dim3(t.tile_m, t.tile_n, t.tile_k),
        block_waves=Dim3(t.warp_m, t.warp_n, t.warp_k),
        wave_tile=Dim3(t.warp_tile_m, t.warp_tile_n, t.warp_tile_k),
        pipeline=pipeline,
        pipeline_scheduler=scheduler,
        store_strategy=store_strategy,
        pad_m=config.trait.pad_m,
        pad_n=config.trait.pad_n,
    )

    # Build physical tensor table
    tensor_layouts = [DISPATCHER_LAYOUT[c] for c in layout[:3]]
    physical_tensors = [
        PhysicalTensor(
            name="A", dtype=input_dtype, layout=tensor_layouts[0], args_slot=0
        ),
        PhysicalTensor(
            name="B", dtype=input_dtype, layout=tensor_layouts[1], args_slot=1
        ),
    ]

    # Output tensor — name from op chain, dtype may differ from input
    out_name = _find_output_name(ops)
    physical_tensors.append(
        PhysicalTensor(
            name=out_name, dtype=output_dtype, layout=tensor_layouts[2], args_slot=2
        )
    )

    # D tensors for multi-D variants
    d_layout = DISPATCHER_LAYOUT.get(config.d_layout, Layout.Row)
    for i in range(config.num_d_tensors):
        physical_tensors.append(
            PhysicalTensor(
                name=f"D{i}",
                dtype=output_dtype,
                layout=d_layout,
                args_slot=3 + i,
            )
        )

    # Derive workgroup_size
    wavefront_size = 64  # CDNA
    workgroup_size = t.warp_m * t.warp_n * t.warp_k * wavefront_size

    epilogue_ops = _build_epilogue_ops(ops)

    spec = GemmSpec(
        physical_tensors=physical_tensors,
        acc_dtype=acc_dtype,
        block_tile=Dim3(t.tile_m, t.tile_n, t.tile_k),
        block_waves=Dim3(t.warp_m, t.warp_n, t.warp_k),
        wave_tile=Dim3(t.warp_tile_m, t.warp_tile_n, t.warp_tile_k),
        workgroup_size=workgroup_size,
        k_batch=1,
        pipeline=pipeline,
        pipeline_scheduler=scheduler,
        tile_partitioner=TilePartitioner.Linear,
        epilogue_ops=epilogue_ops,
        store_strategy=store_strategy,
        pad_m=config.trait.pad_m,
        pad_n=config.trait.pad_n,
    )

    return sig, algo, spec


# ============================================================================
# Serialization: typed model → .hip C++ source
# ============================================================================


def _ops_to_cpp(ops: list[Op]) -> str:
    """Serialize the op chain to C++ designated-initializer literals."""
    parts = []
    for op in ops:
        if isinstance(op, GemmOp):
            if op.acc_dtype != DataType.FP32:
                parts.append(
                    f'rocm_ck::GemmOp{{.lhs = "{op.lhs}", .rhs = "{op.rhs}", '
                    f'.out = "{op.out}", '
                    f".acc_dtype = rocm_ck::DataType::{op.acc_dtype.value}}}"
                )
            else:
                parts.append(
                    f'rocm_ck::GemmOp{{.lhs = "{op.lhs}", .rhs = "{op.rhs}", '
                    f'.out = "{op.out}"}}'
                )
        elif isinstance(op, AddOp):
            parts.append(
                f'rocm_ck::AddOp{{.lhs = "{op.lhs}", .rhs = "{op.rhs}", '
                f'.out = "{op.out}"}}'
            )
        elif isinstance(op, MulOp):
            parts.append(
                f'rocm_ck::MulOp{{.lhs = "{op.lhs}", .rhs = "{op.rhs}", '
                f'.out = "{op.out}"}}'
            )
        elif isinstance(op, (ReluOp, FastGeluOp, GeluOp, SiluOp, SigmoidOp)):
            type_name = type(op).__name__
            parts.append(f'rocm_ck::{type_name}{{.in = "{op.in_}", .out = "{op.out}"}}')

    indent = "                                 "
    return (",\n" + indent).join(parts)


def _algo_to_cpp(algo: GemmAlgorithm) -> str:
    """Serialize GemmAlgorithm to C++ designated-initializer fields."""
    fields = [
        f".block_tile  = {{{algo.block_tile.m}, {algo.block_tile.n}, {algo.block_tile.k}}}",
        f".block_waves = {{{algo.block_waves.m}, {algo.block_waves.n}, {algo.block_waves.k}}}",
        f".wave_tile   = {{{algo.wave_tile.m}, {algo.wave_tile.n}, {algo.wave_tile.k}}}",
    ]
    if algo.pipeline != Pipeline.V1:
        fields.append(f".pipeline    = rocm_ck::Pipeline::{algo.pipeline.value}")
    if algo.pipeline_scheduler != PipelineScheduler.Intrawave:
        fields.append(
            f".pipeline_scheduler = rocm_ck::PipelineScheduler::{algo.pipeline_scheduler.value}"
        )
    if algo.store_strategy != StoreStrategy.CShuffle:
        fields.append(
            f".store_strategy = rocm_ck::StoreStrategy::{algo.store_strategy.value}"
        )
    if algo.pad_m:
        fields.append(".pad_m = true")
    if algo.pad_n:
        fields.append(".pad_n = true")

    return ",\n                           ".join(fields)


def _sig_to_cpp(sig: Signature, ops_cpp: str) -> str:
    """Serialize Signature to C++ designated-initializer block."""
    if sig.tensors:
        tensors_cpp = ", ".join(
            f'rocm_ck::Tensor{{.name = "{t.name}", '
            f".dtype = rocm_ck::DataType::{t.dtype.value}}}"
            for t in sig.tensors
            if t.dtype is not None
        )
        return (
            f".dtype   = rocm_ck::DataType::{sig.dtype.value},\n"
            f"                       .tensors = {{{tensors_cpp}}},\n"
            f"                       .ops     = {{{ops_cpp}}}"
        )
    else:
        return (
            f".dtype = rocm_ck::DataType::{sig.dtype.value},\n"
            f"                       .ops   = {{{ops_cpp}}}"
        )


def to_hip_source(
    kernel_name: str,
    sig: Signature,
    algo: GemmAlgorithm,
    target_set_expr: str,
) -> str:
    """Serialize Signature + GemmAlgorithm to a .hip file.

    target_set_expr is the C++ TargetSet expression (e.g., "cdna()" or
    "family_gfx94()"), not a list of arch strings.
    """
    ops_cpp = _ops_to_cpp(sig.ops)
    sig_cpp = _sig_to_cpp(sig, ops_cpp)
    algo_cpp = _algo_to_cpp(algo)

    return f"""\
// Auto-generated by unified_gemm_codegen.py --output-format kpack
// SPDX-License-Identifier: MIT

#pragma once

#include <rocm_ck/gemm_spec.hpp>

static constexpr const char* kName = "{kernel_name}";

static constexpr auto kTargets = rocm_ck::TargetSet::{target_set_expr};

static constexpr auto kSpec = rocm_ck::makeSpec(
    rocm_ck::Signature{{{sig_cpp}}},
    rocm_ck::GemmAlgorithm{{{algo_cpp}}},
    kTargets);

#ifdef __HIP_DEVICE_COMPILE__
#include <rocm_ck/gemm_dev.hpp>

extern "C" __global__ void {kernel_name}(rocm_ck::Args args)
{{
    rocm_ck::run<kSpec>(args);
}}
#endif
"""
