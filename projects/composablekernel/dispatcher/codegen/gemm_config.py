# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared GEMM kernel configuration types.

This module exists to solve a Python import problem. When a script is run
directly (``python3 foo.py``), Python loads it as the ``__main__`` module.
If another module then does ``from foo import Bar``, Python loads ``foo.py``
a *second* time as module ``foo``. Now ``Bar`` exists as two different
classes — ``__main__.Bar`` and ``foo.Bar``. Enum ``==`` uses class identity,
so ``__main__.Bar.X == foo.Bar.X`` evaluates to ``False``.

By placing shared types here — in a module that is never run as ``__main__``
— every importer gets the same class, and normal ``==`` works everywhere.

Types defined here:
    GemmVariant   — STANDARD, PRESHUFFLE, MULTI_D
    TileConfig    — block/warp/wave tile dimensions
    TraitConfig   — pipeline, epilogue, scheduler, padding
    KernelConfig  — complete kernel configuration (tile + trait + variant)
"""

from dataclasses import asdict, dataclass
from enum import Enum


class GemmVariant(Enum):
    """GEMM kernel variants"""

    STANDARD = "standard"
    PRESHUFFLE = "preshuffle"
    MULTI_D = "multi_d"


@dataclass
class TileConfig:
    """Tile configuration parameters"""

    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    def is_valid(self) -> bool:
        """Validate tile configuration"""
        return (
            self.tile_m % (self.warp_m * self.warp_tile_m) == 0
            and self.tile_n % (self.warp_n * self.warp_tile_n) == 0
            and self.tile_k % (self.warp_k * self.warp_tile_k) == 0
            and self.tile_m > 0
            and self.tile_n > 0
            and self.tile_k > 0
        )


@dataclass
class TraitConfig:
    """Kernel trait configuration"""

    pipeline: str  # mem, compv3, compv4
    epilogue: str  # default, cshuffle
    scheduler: str  # intrawave, interwave
    pad_m: bool
    pad_n: bool
    pad_k: bool
    persistent: bool

    def is_valid(self) -> bool:
        """Check if trait combination is valid"""
        # Only 'mem' pipeline supports interwave scheduler.
        # All compute pipelines (compv3/v4/v5/v6/async) only support intrawave.
        unsupported = {
            ("compv3", "cshuffle", "interwave"),
            ("compv3", "default", "interwave"),
            ("compv4", "cshuffle", "interwave"),
            ("compv4", "default", "interwave"),
            ("compv5", "cshuffle", "interwave"),
            ("compv5", "default", "interwave"),
            ("compv6", "cshuffle", "interwave"),
            ("compv6", "default", "interwave"),
            ("comp_async", "cshuffle", "interwave"),
            ("comp_async", "default", "interwave"),
        }
        return (self.pipeline, self.epilogue, self.scheduler) not in unsupported


@dataclass
class KernelConfig:
    """Complete kernel configuration"""

    tile: TileConfig
    trait: TraitConfig
    variant: GemmVariant = GemmVariant.STANDARD

    # Variant-specific
    preshuffle: bool = False
    elementwise_op: str = "PassThrough"
    num_d_tensors: int = 0
    d_layout: str = "r"  # Layout for D tensors (r=row, c=col) - same for all D tensors

    # Fixed parameters
    block_size: int = 256
    k_block_per_cu: int = 1
    num_wave_groups: int = 1

    def name(self, datatype: str, layout: str) -> str:
        """C++ alias for template instance"""
        return f"ck_tile_gemm_{self.key_name(datatype, layout)}"

    def key_name(self, datatype: str, layout: str) -> str:
        """
        Unique identifier for this kernel configuration.

        All parameters that affect kernel behavior MUST be included to ensure
        unique names for unique configurations:
        - Data type and layout (signature)
        - Tile, warp, warp_tile dimensions (algorithm)
        - Pipeline, epilogue, scheduler (traits)
        - Padding flags (affects divisibility requirements)
        - Persistent mode
        - Preshuffle variant
        - Multi-D: elementwise op, num D tensors, D layout
        - Occupancy: wave groups, k_block_per_cu (if non-default)
        """
        parts = []
        # Signature
        parts.append(f"dt_{datatype}")
        parts.append(f"ly_{layout}")

        # Tile configuration
        parts.append(f"tile_{self.tile.tile_m}x{self.tile.tile_n}x{self.tile.tile_k}")
        parts.append(f"warp_{self.tile.warp_m}x{self.tile.warp_n}x{self.tile.warp_k}")
        parts.append(
            f"wtile_{self.tile.warp_tile_m}x{self.tile.warp_tile_n}x{self.tile.warp_tile_k}"
        )

        # Traits
        parts.append(f"pipe_{self.trait.pipeline}")
        parts.append(f"epi_{self.trait.epilogue}")
        parts.append(f"sched_{self.trait.scheduler}")

        # Padding flags (only if not all True - the common case)
        if not (self.trait.pad_m and self.trait.pad_n and self.trait.pad_k):
            parts.append(
                f"pad{int(self.trait.pad_m)}{int(self.trait.pad_n)}{int(self.trait.pad_k)}"
            )

        # Persistent mode
        if self.trait.persistent:
            parts.append("persist")

        # Preshuffle variant
        if self.preshuffle:
            parts.append("preshuffle")

        # Multi-D variant: include elementwise op, num tensors, and D layout
        if self.variant == GemmVariant.MULTI_D:
            parts.append(f"ew_{self.elementwise_op}")
            parts.append(f"nd{self.num_d_tensors}")
            parts.append(f"dly_{self.d_layout}")

        # Occupancy parameters (only if non-default)
        if self.num_wave_groups != 1:
            parts.append(f"wg{self.num_wave_groups}")
        if self.k_block_per_cu != 1:
            parts.append(f"kbpc{self.k_block_per_cu}")

        return "_".join(parts)

    def dict_items(self):
        """Iterator over (field, value) pairs"""
        return asdict(self).items()
