# Kpack Output Format

The `--output-format kpack` flag generates self-describing `.kpack` archives
for runtime kernel loading. Kernels are loaded at runtime via
`hipModuleLaunchKernel` — the host binary needs no CK Tile template headers.
A single archive can contain binaries for multiple GPU architectures.

This is a design proof of concept for the rocm-ck integration path.

## Quick start

```bash
cd dispatcher/codegen

# Generate a kpack archive for two GPU architectures
python3 unified_gemm_codegen.py \
    --output-dir ./build \
    --datatype fp16 \
    --layout rcr \
    --output-format kpack \
    --arch gfx90a \
    --arch gfx942

# With zstd compression
python3 unified_gemm_codegen.py \
    --output-dir ./build \
    --datatype fp16 \
    --layout rcr \
    --output-format kpack \
    --arch gfx90a \
    --arch gfx942 \
    --zstd
```

This produces `build/gemm.kpack` containing compiled `.hsaco` blobs and
structured `GemmSpec` metadata for every kernel variant.

## Kpack CLI flags

| Flag | Description |
|------|-------------|
| `--output-format kpack` | Enable kpack mode |
| `--arch ARCH` | GPU architecture to compile for (repeatable). Defaults to `--gpu-target` |
| `--rocm-path PATH` | ROCm installation path (default: `/opt/rocm`) |
| `--zstd` | Enable zstd-per-kernel compression |
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
    --arch gfx90a

# 2. Run it through the dispatcher
cd ../../experimental/rocm_ck/tools/dispatcher_bridge
build/dispatcher_bridge ../../../../dispatcher/codegen/build/gemm.kpack
```

Expected output (on gfx90a):

```
Registered 15 kernels from kpack

Registered kernels:
  gemm_fp16                       fp16_rcr_compv1_intrawave_cshuffle_128x128x32_...
  gemm_bf16                       bf16_rcr_compv1_intrawave_cshuffle_128x128x32_...
  ...

Running gemm_fp16: M=512, N=512, K=256, block=256
gemm_fp16: PASSED  (0.123 ms, max_rel_err=1.23e-04, errors=0/262144)
validate(): PASSED
Performance: 1.23 TFLOPS
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
  [blobs]     .hsaco binaries (one per kernel x arch)
  [TOC]       msgpack map:
                compression_scheme: "none" | "zstd-per-kernel"
                gfx_arches: ["gfx90a", "gfx942"]
                toc: { kernel_name: { arch: { ordinal, size } } }
                variant_specs: {
                  kernel_name: {
                    spec_type: "GemmSpec",
                    targets: ["gfx90a", "gfx942"],
                    spec: { physical_tensors, block_tile, pipeline, ... }
                  }
                }
```

The archive is self-describing — the runtime discovers kernels, their
capabilities, and their specs from the TOC without compiled-in tables.
