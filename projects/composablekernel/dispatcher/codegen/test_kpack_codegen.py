# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for kpack codegen pipeline — naming, generation, and packing.

Run: pytest test_kpack_codegen.py -v
"""

import json
from pathlib import Path

from gemm_config import GemmVariant, KernelConfig, TileConfig, TraitConfig
from preselected_kernels import get_preselected_set
from unified_gemm_codegen import KernelNaming, UnifiedGemmCodegen


# ============================================================================
# Helpers
# ============================================================================


def _tile(m=128, n=128, k=32, wm=2, wn=2, wk=1, wtm=32, wtn=32, wtk=16):
    return TileConfig(m, n, k, wm, wn, wk, wtm, wtn, wtk)


def _trait(
    pipeline="compv4",
    epilogue="cshuffle",
    scheduler="intrawave",
    pad_m=True,
    pad_n=True,
    pad_k=True,
    persistent=False,
):
    return TraitConfig(pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, persistent)


def _config(tile=None, trait=None, variant=GemmVariant.STANDARD, **kwargs):
    return KernelConfig(
        tile=tile or _tile(), trait=trait or _trait(), variant=variant, **kwargs
    )


def _essential_configs():
    return get_preselected_set("fp16_rcr_essential")


def _essential_names():
    return [KernelNaming.generate(c, "fp16", "rcr") for c in _essential_configs()]


def _make_codegen(tmp_path, **kwargs):
    """Create a UnifiedGemmCodegen pointed at a temp directory."""
    defaults = dict(
        output_dir=str(tmp_path),
        datatype="fp16",
        layout="rcr",
        gpu_target="gfx942",
        use_preselected="fp16_rcr_essential",
        output_format="kpack",
        clean=True,
    )
    defaults.update(kwargs)
    return defaults


# ============================================================================
# TestMainModuleSafety — enum identity across module boundaries
# ============================================================================


class TestMainModuleSafety:
    """GemmVariant must be the same class everywhere.

    Shared types live in gemm_config.py — a module that is never run as
    __main__. This guarantees every importer gets the same enum class,
    so normal == works. See gemm_config.py docstring for the full story.
    """

    def test_preselected_configs_use_same_enum_class(self):
        """Configs from preselected_kernels use the same GemmVariant as us."""
        configs = _essential_configs()
        for c in configs:
            assert type(c.variant) is GemmVariant, (
                f"Config variant type is {type(c.variant).__module__}.{type(c.variant).__qualname__}, "
                f"expected gemm_config.GemmVariant"
            )

    def test_normal_enum_comparison_works(self):
        """Normal == works — no .value workaround needed."""
        configs = _essential_configs()
        multi_d = [c for c in configs if c.variant == GemmVariant.MULTI_D]
        assert len(multi_d) == 2


# ============================================================================
# TestKernelNaming — pure logic, no I/O
# ============================================================================


class TestKernelNaming:
    """Kernel names must be unique across all configs in a preselected set."""

    def test_standard_names_are_unique(self):
        configs = [c for c in _essential_configs() if c.variant == GemmVariant.STANDARD]
        names = [KernelNaming.generate(c, "fp16", "rcr") for c in configs]
        assert len(names) == len(set(names)), f"Duplicate standard names: {names}"

    def test_multi_d_names_differ_from_standard(self):
        configs = _essential_configs()
        standard_names = {
            KernelNaming.generate(c, "fp16", "rcr")
            for c in configs
            if c.variant == GemmVariant.STANDARD
        }
        multi_d_names = {
            KernelNaming.generate(c, "fp16", "rcr")
            for c in configs
            if c.variant == GemmVariant.MULTI_D
        }
        overlap = standard_names & multi_d_names
        assert not overlap, f"MULTI_D names collide with standard: {overlap}"

    def test_all_essential_names_unique(self):
        names = _essential_names()
        assert len(names) == len(set(names)), (
            f"Duplicate names in essential set: {names}"
        )

    def test_multi_d_name_contains_variant_suffix(self):
        configs = _essential_configs()
        for c in configs:
            if c.variant == GemmVariant.MULTI_D:
                name = KernelNaming.generate(c, "fp16", "rcr")
                assert "_multid_" in name, f"MULTI_D name missing suffix: {name}"
                assert c.elementwise_op in name, f"Missing op in name: {name}"

    def test_name_includes_layout(self):
        name = KernelNaming.generate(_config(), "fp16", "rcr")
        assert "_rcr_" in name

    def test_multi_d_name_includes_d_layout(self):
        config = _config(
            variant=GemmVariant.MULTI_D,
            elementwise_op="Relu",
            num_d_tensors=1,
            d_layout="r",
        )
        name = KernelNaming.generate(config, "fp16", "rcr")
        assert "_rcrr_" in name, f"MULTI_D name should have 4-char layout: {name}"


# ============================================================================
# TestPreselectedSets — config integrity
# ============================================================================


class TestPreselectedSets:
    """Preselected kernel sets have correct structure and no duplicates."""

    def test_essential_count(self):
        assert len(_essential_configs()) == 6

    def test_essential_variants(self):
        configs = _essential_configs()
        standard = [c for c in configs if c.variant == GemmVariant.STANDARD]
        multi_d = [c for c in configs if c.variant == GemmVariant.MULTI_D]
        assert len(standard) == 4
        assert len(multi_d) == 2

    def test_essential_tiles(self):
        configs = _essential_configs()
        tiles = {(c.tile.tile_m, c.tile.tile_n, c.tile.tile_k) for c in configs}
        assert (256, 256, 32) in tiles
        assert (128, 128, 32) in tiles
        assert (32, 64, 32) in tiles
        assert (64, 32, 32) in tiles

    def test_no_duplicate_configs(self):
        """No two configs should produce the same kernel name."""
        names = _essential_names()
        assert len(names) == len(set(names))

    def test_multi_d_ops(self):
        configs = _essential_configs()
        multi_d = [c for c in configs if c.variant == GemmVariant.MULTI_D]
        ops = {c.elementwise_op for c in multi_d}
        assert "Relu" in ops
        assert "Gelu" in ops


# ============================================================================
# TestGenerateOne — file I/O to tmp dir
# ============================================================================


class TestGenerateOne:
    """_generate_one writes correct .hip and .spec.json files."""

    def test_writes_hip_and_spec_json(self, tmp_path):
        codegen = UnifiedGemmCodegen(**_make_codegen(tmp_path))
        config = _essential_configs()[0]  # standard 256x256x32
        hip_path, _ = codegen._generate_one(config)

        hip = Path(hip_path)
        assert hip.exists(), f".hip not written: {hip}"
        assert hip.suffix == ".hip"

        spec = hip.with_suffix(".spec.json")
        assert spec.exists(), f".spec.json not written: {spec}"

    def test_multi_d_writes_distinct_files(self, tmp_path):
        codegen = UnifiedGemmCodegen(**_make_codegen(tmp_path))
        configs = _essential_configs()
        standard_128 = configs[1]  # standard 128x128x32
        multi_d_relu = configs[4]  # multi_d Relu 128x128x32

        path_std, _ = codegen._generate_one(standard_128)
        path_md, _ = codegen._generate_one(multi_d_relu)

        assert path_std != path_md, "Standard and MULTI_D should have different paths"
        assert Path(path_std).exists()
        assert Path(path_md).exists()

    def test_hip_content_matches_kernel_name(self, tmp_path):
        codegen = UnifiedGemmCodegen(**_make_codegen(tmp_path))
        config = _essential_configs()[0]
        hip_path, _ = codegen._generate_one(config)

        stem = Path(hip_path).stem
        content = Path(hip_path).read_text()
        assert f'extern "C" __global__ void {stem}(' in content

    def test_spec_json_name_matches_filename(self, tmp_path):
        codegen = UnifiedGemmCodegen(**_make_codegen(tmp_path))
        config = _essential_configs()[0]
        hip_path, _ = codegen._generate_one(config)

        spec_path = Path(hip_path).with_suffix(".spec.json")
        spec = json.loads(spec_path.read_text())
        assert spec["name"] == Path(hip_path).stem

    def test_all_essential_produce_distinct_files(self, tmp_path):
        codegen = UnifiedGemmCodegen(**_make_codegen(tmp_path))
        configs = _essential_configs()

        hip_paths = []
        for cfg in configs:
            hip_path, _ = codegen._generate_one(cfg)
            hip_paths.append(hip_path)

        # All paths unique
        assert len(hip_paths) == len(set(hip_paths)), (
            f"Duplicate hip paths: {hip_paths}"
        )

        # All files exist
        for p in hip_paths:
            assert Path(p).exists(), f"Missing .hip: {p}"
            assert Path(p).with_suffix(".spec.json").exists(), (
                f"Missing .spec.json: {p}"
            )

        # Count files on disk
        hip_files = list(tmp_path.glob("*.hip"))
        spec_files = list(tmp_path.glob("*.spec.json"))
        assert len(hip_files) == 6, f"Expected 6 .hip files, got {len(hip_files)}"
        assert len(spec_files) == 6, (
            f"Expected 6 .spec.json files, got {len(spec_files)}"
        )


# ============================================================================
# TestGenerateAll — integration (no compilation, just file generation)
# ============================================================================


class TestGenerateAll:
    """generate_all must produce all kernels, both sequential and parallel."""

    def _count_results(self, results):
        return len(results["kernels"]), len(results["failed"])

    def test_sequential_produces_all_kernels(self, tmp_path):
        codegen = UnifiedGemmCodegen(**_make_codegen(tmp_path))
        results = codegen.generate_all(parallel=False)
        n_kernels, n_failed = self._count_results(results)

        assert n_failed == 0, f"{n_failed} kernels failed"
        assert n_kernels == 6, f"Expected 6 kernels, got {n_kernels}"

    def test_parallel_produces_all_kernels(self, tmp_path):
        codegen = UnifiedGemmCodegen(**_make_codegen(tmp_path))
        results = codegen.generate_all(parallel=True)
        n_kernels, n_failed = self._count_results(results)

        assert n_failed == 0, f"{n_failed} kernels failed"
        assert n_kernels == 6, f"Expected 6 kernels, got {n_kernels}"

    def test_parallel_matches_sequential(self, tmp_path):
        seq_dir = tmp_path / "seq"
        par_dir = tmp_path / "par"

        seq_codegen = UnifiedGemmCodegen(**_make_codegen(seq_dir))
        par_codegen = UnifiedGemmCodegen(**_make_codegen(par_dir))

        seq_results = seq_codegen.generate_all(parallel=False)
        par_results = par_codegen.generate_all(parallel=True)

        seq_names = sorted(Path(p).stem for p in seq_results["kernels"])
        par_names = sorted(Path(p).stem for p in par_results["kernels"])

        assert seq_names == par_names, (
            f"Parallel and sequential produced different kernel sets.\n"
            f"  Sequential only: {set(seq_names) - set(par_names)}\n"
            f"  Parallel only: {set(par_names) - set(seq_names)}"
        )

    def test_all_hip_files_exist_after_generate(self, tmp_path):
        codegen = UnifiedGemmCodegen(**_make_codegen(tmp_path))
        results = codegen.generate_all(parallel=True)

        for hip_path in results["kernels"]:
            assert Path(hip_path).exists(), f"Missing .hip: {hip_path}"
            spec_path = Path(hip_path).with_suffix(".spec.json")
            assert spec_path.exists(), f"Missing .spec.json: {spec_path}"

    def test_all_hip_files_on_disk_match_returned_count(self, tmp_path):
        """Files on disk should match returned kernel count — no phantom files."""
        codegen = UnifiedGemmCodegen(**_make_codegen(tmp_path))
        results = codegen.generate_all(parallel=True)

        hip_files = list(tmp_path.glob("*.hip"))
        assert len(hip_files) == len(results["kernels"]), (
            f"Returned {len(results['kernels'])} paths but "
            f"found {len(hip_files)} .hip files on disk"
        )

    def test_kpack_contains_all_variants(self, tmp_path):
        """The kpack archive must contain all 6 kernel variants."""
        codegen = UnifiedGemmCodegen(**_make_codegen(tmp_path))
        results = codegen.generate_all(parallel=False)

        kpack_path = results.get("kpack_path")
        assert kpack_path, "No kpack created"
        assert Path(kpack_path).exists()

        # Inspect the kpack TOC
        import msgpack
        import struct

        with open(kpack_path, "rb") as f:
            magic = f.read(4)
            assert magic == b"KPAK"
            struct.unpack("<I", f.read(4))  # version
            toc_offset = struct.unpack("<Q", f.read(8))[0]
            f.seek(toc_offset)
            toc = msgpack.unpackb(f.read(), raw=False)

        variant_names = sorted(toc["variant_specs"].keys())
        assert len(variant_names) == 6, (
            f"Expected 6 variants in kpack, got {len(variant_names)}: {variant_names}"
        )

    def test_all_hsaco_files_on_disk(self, tmp_path):
        """All .hsaco files should be on disk after generate_all."""
        codegen = UnifiedGemmCodegen(**_make_codegen(tmp_path))
        codegen.generate_all(parallel=False)

        hsaco_files = sorted(tmp_path.glob("*.hsaco"))
        assert len(hsaco_files) == 6, (
            f"Expected 6 .hsaco files, got {len(hsaco_files)}: "
            f"{[f.name for f in hsaco_files]}"
        )
