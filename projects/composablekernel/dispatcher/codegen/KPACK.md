# Kpack Output Format

The `--output-format kpack` flag generates self-describing `.kpack` archives
for runtime kernel loading. Kernels are loaded at runtime via
`hipModuleLaunchKernel` — the host binary needs no CK Tile template headers.
Each archive targets a single GPU architecture.

This is a design proof of concept for the rocm-ck integration path.

## Quick start

```bash
cd dispatcher/codegen

# Preselected essential set (6 kernels — fast build, good coverage)
python3 unified_gemm_codegen.py \
    --output-dir ./build \
    --datatype fp16 \
    --layout rcr \
    --output-format kpack \
    --gpu-target gfx90a \
    --zstd \
    --preselected fp16_rcr_essential
```

This produces `build/gemm.kpack` containing compiled `.hsaco` blobs and
structured `GemmSpec` metadata for every kernel variant.

### Larger builds

```bash
# Full tile sweep with padding, no persistent (all tiles, all problem sizes)
python3 unified_gemm_codegen.py \
    --output-dir ./build \
    --datatype fp16 \
    --layout rcr \
    --output-format kpack \
    --gpu-target gfx90a \
    --zstd \
    --tile-config-json '{
        "tile_config": {
            "tile_m": [64, 128, 192, 256], "tile_n": [64, 128, 192, 256],
            "tile_k": [64, 128, 192, 256],
            "warp_m": [1, 2, 4], "warp_n": [1, 2, 4], "warp_k": [1],
            "warp_tile_m": [4, 16, 32], "warp_tile_n": [16, 32, 64],
            "warp_tile_k": [8, 16, 32, 64, 128]
        },
        "trait_config": {
            "pipeline": ["compv3", "compv4", "mem"],
            "epilogue": ["cshuffle", "default"],
            "scheduler": ["intrawave", "interwave"],
            "pad_m": [true], "pad_n": [true], "pad_k": [false],
            "persistent": [false]
        }
    }'

# Full sweep: all pipelines, schedulers, tile shapes, persistent modes
python3 unified_gemm_codegen.py \
    --output-dir ./build \
    --datatype fp16 \
    --layout rcr \
    --output-format kpack \
    --gpu-target gfx90a \
    --zstd
```

Available preselected sets: `fp16_rcr_essential`, `bf16_rcr_essential`,
`int8_rcr_essential`, `fp8_rcr_essential`, `mixed_precision`.

## Kpack CLI flags

| Flag | Description |
|------|-------------|
| `--output-format kpack` | Enable kpack mode |
| `--gpu-target ARCH` | GPU architecture to compile for |
| `--zstd` | Enable zstd-per-kernel compression |
| `--preselected SET` | Use a curated kernel set (see above) |
| `--tile-config-json JSON` | Narrow the tile/trait sweep |
| `--rocm-path PATH` | ROCm installation path (default: `/opt/rocm`) |
| `--no-clean` | Skip cleaning stale artifacts from output dir |

## Running a kpack with the dispatcher

The `dispatcher_bridge` CLI loads a kpack archive into the CK dispatcher's
registry, selects a kernel, runs a GEMM, and verifies the result.

### Build the CLI

```bash
cd experimental/rocm_ck/tools/dispatcher_bridge
cmake -B build -S . -G Ninja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_PREFIX_PATH=/opt/rocm
ninja -C build -j16
```

### End-to-end: codegen → kpack → dispatcher

```bash
# 1. Generate a kpack archive
cd dispatcher/codegen
python3 unified_gemm_codegen.py \
    --output-dir ./build \
    --datatype fp16 \
    --layout rcr \
    --output-format kpack \
    --gpu-target gfx90a \
    --zstd \
    --preselected fp16_rcr_essential

# 2. Run it through the dispatcher
cd ../../experimental/rocm_ck/tools/dispatcher_bridge
build/dispatcher_bridge ../../../../dispatcher/codegen/build/gemm.kpack
```

### How it works

The bridge uses `registerKpackKernels()` to project each `GemmSpec` from the
kpack TOC into a dispatcher `KernelKey`, then wraps the kernel as a
`KpackKernelInstance` in the dispatcher's `Registry`:

```cpp
#include <ck_tile/dispatcher/backends/kpack_backend.hpp>
#include <ck_tile/dispatcher/dispatcher.hpp>

Registry registry;
int n = registerKpackKernels("gemm.kpack", registry);

Dispatcher dispatcher(&registry);
auto kernel = dispatcher.select_kernel(problem);
float ms = kernel->run(a, b, c, nullptr, problem, stream);
```

See `experimental/rocm_ck/tools/dispatcher_bridge/` for the full implementation.

## Architecture

```
unified_gemm_codegen.py  --output-format kpack
        |
        v
   rocm_ck_bridge.py
   +---------------------------+
   | translate_kernel_config() |  KernelConfig -> typed dataclasses
   | to_hip_source()           |  dataclasses  -> .hip (C++ source)
   | GemmSpec.to_spec_json()   |  dataclasses  -> .spec.json
   +---------------------------+
        |                  |
     .hip files       .spec.json files
        |                  |
     clang++ --> .hsaco    |
        |                  |
        +---- pack.py -----+
                |
            .kpack archive
                |
                v
        kpack_backend.hpp   (runtime: load -> filter by arch -> register)
                |
                v
        dispatcher_bridge   (CLI: register -> select -> launch -> verify)
```

### Module responsibilities

| Module | Role |
|--------|------|
| `rocm_ck_bridge.py` | Anti-corruption layer between dispatcher and rocm_ck. Translates dispatcher `KernelConfig` into rocm_ck typed dataclasses (Signature, GemmAlgorithm, GemmSpec). Serializes to .hip C++ and .spec.json. |
| `unified_gemm_codegen.py` | Orchestration. Enumerates tile configs, validates against architecture, drives parallel compilation, invokes pack.py. `KpackGenerator` delegates to `rocm_ck_bridge` for translation/serialization, keeps `compile_hsaco()` for build. |
| `pack.py` | Packs `.hsaco` blobs + `.spec.json` metadata into a single `.kpack` archive (msgpack TOC + binary blobs). |
| `kpack_backend.hpp` | Dispatcher runtime backend. Loads `.kpack`, reads `GemmSpec` from TOC, filters by GPU arch, registers kernels into the dispatcher registry. |

### What's in the archive

```
gemm.kpack
  [header]    KPAK magic + version + toc_offset
  [blobs]     .hsaco binaries (one per kernel)
  [TOC]       msgpack map:
                compression_scheme: "none" | "zstd-per-kernel"
                gfx_arches: ["gfx90a"]
                toc: { kernel_name: { arch: { ordinal, size } } }
                variant_specs: {
                  kernel_name: {
                    spec_type: "GemmSpec",
                    targets: ["gfx90a"],
                    spec: { physical_tensors, block_tile, pipeline, ... }
                  }
                }
```

The archive is self-describing — the runtime discovers kernels, their
capabilities, and their specs from the TOC without compiled-in tables.
