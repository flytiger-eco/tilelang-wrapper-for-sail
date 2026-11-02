# tilelang-sail-cuda-wrapper

Run CUDA-style [TileLang](https://github.com/tile-ai/tilelang) code on tilelang-sail (the PPU-backend distribution of TileLang) **without modifying a single line**.

- **Zero-change migration**: existing idioms such as `import tilelang.cuda`, `from tilelang.cuda.intrinsics import ...`, `target="cuda"`, and `tl.transform.LowerTileOp()` work as-is
- **Pure runtime approach**: takes effect upon pip install and reverts upon uninstall; neither the tilelang source tree is modified nor tilelang is repackaged
- **Zero dependencies**: relies only on the Python standard library (import protocol + site mechanism), introducing no runtime dependencies
- **Explicit boundaries**: CUDA-specific features with no PPU counterpart raise explicit errors — never silent degradation

> **Unofficial third-party project**: This repository is not affiliated with, endorsed, or supported by TileLang, tilelang-sail, or their development teams. This package contains and modifies none of their code — all compatibility logic is supplied at runtime by the independently installed wrapper. Please report issues with tilelang-sail itself to its official channels.

## Background

We had an existing collection of TileLang kernels written in CUDA style. When migrating them to tilelang-sail (the PPU-backend distribution of TileLang), we ran into three classes of incompatibility in sequence:

1. **import**: `from tilelang.cuda.intrinsics import TensorCoreIntrinEmitter` failed immediately — the `tilelang.cuda` module itself was importable, but invoking its interfaces triggered native errors. Inspecting `tilelang.ppu` revealed that equivalent capabilities were all available, merely under different names (`TensorCoreIntrinEmitter` → `PPUTensorCoreIntrinEmitter`);
2. **target**: after fixing the imports one by one, compilation with `target="cuda"` failed again — the PPU backend does not accept that target;
3. **pass**: after switching the target to `"ppu"`, compilation still failed during lowering — generic passes such as `LowerTileOp` / `LayoutInference` need to go through the dedicated implementations in `tilelang.ppu.transform` when running on the PPU target.


## Install / Uninstall

Prerequisite: the PPU-only build of tilelang-sail is installed. This wrapper currently targets tilelang-sail **0.1.11**, **0.1.12** and **0.1.15** — all behavior mappings were validated against these three versions; other versions are unverified.

```bash
pip install .                               # run in the repository root (no runtime dependencies; tilelang is NOT reinstalled)
pip uninstall tilelang-sail-cuda-wrapper    # uninstall; native behavior is fully restored
```

A **Python process restart** is required after install/uninstall for changes to take effect (the `.pth` file registers the interceptor only at interpreter startup; running processes are unaffected).

## Quick Start

```python
import tilelang
import tilelang.language as T

# CUDA-style intrinsics — resolved by the wrapper to the PPU implementation (PPUTensorCoreIntrinEmitter)
from tilelang.cuda.intrinsics import TensorCoreIntrinEmitter

@tilelang.jit(target="cuda")    # rewritten at runtime to a PPU target; arch determined by device probing
def matmul(A, B, block_M, block_N, block_K, num_stages, dtype=T.float16, accum_dtype=T.float32):
    ...                          # kernel body identical to the CUDA version

kernel = matmul.compile(M=1024, N=1024, K=1024,
                        block_M=128, block_N=128, block_K=32, num_stages=3)
print(kernel.target)            # PPU target (arch=ppu_10 / ppu_15, depending on the device)
```

For a complete runnable example (with numerical verification against torch), see `tests/e2e_cuda_style.py`.

## Behavior in the Two Installation States

| | Without the wrapper (native) | With the wrapper installed |
|---|---|---|
| `import tilelang.cuda` | Native package is importable, but invoking its interfaces triggers native errors | Aliased to `tilelang.ppu`; CUDA-style names become usable |
| `from tilelang.cuda.intrinsics import TensorCoreIntrinEmitter` | ImportError / native implementation | Forwarded to `PPUTensorCoreIntrinEmitter` |
| `from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter` (module-level path; the `tilelang.intrinsics` facade also resolves through this path) | Native class | Forwarded to the PPU version (resolves to the same class as the package-level entry) |
| `tilelang.cuda` submodules not provided by ppu (`wgmma_macro_generator`, `tcgen05_macro_generator`, generation-specific generators such as `mma_sm70/sm75_*`) | Native module | **Loaded natively** (not intercepted; behaves as if the wrapper were absent) |
| `TensorCoreIntrinEmitterWithLadderTransform` (ladder transform layout plugin; no PPU counterpart) | Native class | **Explicit error** (the name resolves; RuntimeError upon instantiation/call — this keeps the `tilelang.intrinsics` facade loadable) |
| `from tilelang.cuda.op.gemm.gemm_mma import GemmMMA` (generic mma GEMM op implementation) | Native class | Forwarded to `PPUGemmMMA`; generation-specific files (`gemm_mma_sm70/sm75`, `tcgen05`, `wgmma`, etc.) load natively, behaving as if the wrapper were absent |
| `from tilelang.cuda.pipeline import CUDAPassPipelineBody(Prologue)` (lowering pipeline body) | Native function | Forwarded to `PPUPassPipelineBody(Prologue)`; Hopper-specific gating (`allow_warp_specialized`, `module_has_tma`) raises explicit errors |
| `tilelang.cuda.transform.MarkCudaSyncCalls` | Native implementation | Forwarded to `MarkPpuSyncCalls` |
| CUDA-only passes (`LowerBlackwell2SM`, `InjectTcgen05Fence`, etc.) | Native implementation | **Explicit error** (no PPU counterpart) |
| `target="cuda"` (jit/compile/autotune) | Compilation error | Rewritten to ppu; when arch is unspecified it is decided by **device probing** (verified to match what determine_target yields when parsing ppu targets — ppu_10 / ppu_15 by hardware generation); if probing is unavailable, defaults to **ppu_10** — aligned with TVM's behavior of defaulting to sm_50 for CUDA when no device is present; explicit arch mapping was only verified for sm_80→ppu_10 and sm_89→ppu_15 — everything else raises **ValueError** (no speculative mapping) |
| Other targets (llvm/metal/rocm…, via `determine_target`) | Native behavior | **Immediate error** (strict mode) |
| `Target("llvm")` (engine-internal host codegen) | Native behavior | Allowed through (required internally by the engine; not intercepted) |
| Common passes such as `tl.transform.LowerTileOp()(mod)` | Generic implementation | If the mod contains a ppu target → the dedicated pass in `tl.ppu.transform`; otherwise falls back to the native one |
| A separately installed tvm (imported before tilelang) | Native behavior | **Left completely untouched** (only the tvm vendored inside tilelang is patched) |

## How It Works

- **`sys.meta_path` finder (PEP 302/451)**: passively answers the interpreter's import requests — no watcher, no threads, no polling. Inserted at the very front of meta_path so it hits before the native `tilelang/cuda` directory; only the `tilelang.cuda.*` submodules **actually provided by the ppu backend** are intercepted — those ppu does not provide (generation-specific generators such as wgmma/tcgen05) fall through to the native files, behaving exactly as if the wrapper were absent.
- **`.pth` activation**: after pip install, the `site` mechanism executes `import tilelang_sail_cuda_wrapper` on every interpreter startup, registering the finder. Startup cost is O(1); tilelang/tvm are not loaded.
- **Runtime aliasing (PEP 562)**: `tilelang.cuda.*` modules **lazily forward** to `tilelang.ppu.*` via module-level `__getattr__`, performing CUDA-style name mapping on attribute access (`TensorCoreIntrinEmitter → PPUTensorCoreIntrinEmitter`, etc.). Submodule attributes are likewise forwarded to the corresponding cuda aliases, so the raw ppu modules never leak.
  - Why not a `__dict__` snapshot copy: the loading chain of `tilelang.ppu` reverse-imports `tilelang.cuda.*` (under `intrinsics.macro` some modules reference cuda-style layouts), causing the finder to be re-entered while the ppu module is **still mid-initialization**; a snapshot would freeze that incomplete state into `sys.modules`. Lazy forwarding is immune — the module is populated in place, and subsequent accesses always see the complete state.
- **Post-load patching**: the real loaders of `tvm.target` / `tilelang.transform` / tilelang's target entry module / `tilelang.testing` are wrapped, and the patches are applied immediately after the module finishes loading (Target rewriting, pass routing, strict mode, test-gate relaxation). The target entry module varies across tilelang versions: `tilelang.utils.target` in 0.1.11, `tilelang.backend.target` in 0.1.12+; the finder registers both entries and, at runtime, only the one that exists gets loaded — each version gets what it needs without interference. `tilelang.testing` loads only when user test code does `import tilelang.testing`; its patch relaxes `requires_cuda` (natively sourced from tvm's `Feature`, which checks `tvm.runtime.enabled("cuda")` — always False on a PPU-only build → everything skipped) to "skip only if neither CUDA nor PPU is available" — the same approach tilelang officially takes for ROCm via `requires_cuda_or_cdna`; in ≤0.1.11, `requires_cuda_or_cdna` is also fixed, since its native `target_is_cuda` binding is redirected to the ppu check.
- **Version/layout detection** (`_versions.py`): the wrapper identifies which internals layout the running tilelang uses — `legacy` (≤0.1.11: target helpers flat in `tilelang.utils.target`) vs `modern` (≥0.1.12: `tilelang.backend.target` registry + `tilelang.ppu.target`). Layout selection is **capability-probed first** (the authoritative mechanism — it stays correct on fork builds whose version strings don't parse, e.g. `0.1.15+ppu.1.0.0.git...`), with the parsed `tilelang.__version__` acting as an advisory table (assigned before tilelang's heavy imports in every release examined, so it is available whenever a lazy wrapper decision runs). Where the layouts fork (the `tilelang.cuda.target` alias base), the detected layout picks the module and the probe remains the fallback; `tilelang_sail_cuda_wrapper._versions.diagnostics()` prints what was detected — useful when checking whether activation picked the expected layout on a new tilelang release.
- **tvm ownership detection**: the `tvm.target` patch applies **only to the tvm vendored inside tilelang** (`3rdparty/tvm/python`). Detection uses two levels of signals: first, the vendored root is derived from `sys.modules["tilelang"].__file__` (covering both the dev-checkout and wheel layouts) and compared against the tvm file path; if tilelang is not yet loaded (which can only mean the user imported a standalone tvm first), detection falls back to a path signature (a `3rdparty/tvm/python` path segment). A separately installed tvm (e.g. `site-packages/tvm`) is never touched; its `Target("cuda")` passes through unchanged.
- **Strict mode is enforced at `determine_target`** (verified: targets passed by jit/compile/autotune all converge on this entry point), not at `Target.__init__`: during compilation the engine itself constructs host-codegen targets such as `Target("llvm")`, which must be allowed through.

## Repository Structure

```
tilelang_sail_cuda_wrapper/
├── __init__.py        # activation entry: .pth triggers install() (does not load tilelang/tvm)
├── _finder.py         # import-interception skeleton: meta_path finder, two loaders, install()
├── _aliases.py        # tilelang.cuda.* alias module synthesis, CUDA→PPU name mapping, lazy forwarding
├── _targets.py        # Target rewriting (arch mapping / device probing) and strict-mode validation
├── _testing.py        # CUDA gate relaxation for tilelang.testing (requires_cuda accepts PPU)
├── _passes.py         # routing from common passes to PPU-specific passes
├── _versions.py       # tilelang version/layout detection + diagnostics()
└── _tvm_ownership.py  # tvm ownership detection: patch only the tvm vendored inside tilelang
```

The implementation is split into 7 modules by responsibility: `_finder.py` retains only the import-protocol skeleton (finder / loader / `install()`); alias synthesis, target semantics, test gating, pass routing, version detection, and ownership detection each live in their own module. Verification scripts live under `tests/` and are not distributed with the package.

## Verification

Run inside a tilelang-sail environment with PPU hardware:

```bash
python3 tests/verify_states.py native    # native-behavior baseline without the wrapper
python3 tests/verify_states.py wrapped   # behavioral assertions with the wrapper installed
python3 tests/e2e_cuda_style.py          # end to end: CUDA-style matmul compilation + numerical verification
```

Coverage:

- **Common steps** (identical contract on every supported release, in `verify_states.py`): alias forwarding, explicit errors for CUDA-only passes, target rewriting, strict mode, pass routing, tvm ownership detection, the `tilelang.cuda.target` predicate, the CUDA gate in `tilelang.testing` (requires_cuda / requires_cuda_or_cdna no longer skip), the common intrinsics layer (ladder-transform sentinel, consistency between module-level and package-level emitters), the op/pipeline layers, and the version/layout layer (the detected layout must agree with the parsed version);
- **Version-split steps** (`tests/steps/`, dispatched at step 15): checks tied to a particular tilelang release live in per-release modules that self-gate — probe-first with the parsed version as a fast path — and the runner reports each as active or inert, so the log always shows which side of a split the running tilelang exercised. A module is named after the release that **introduced** its surface; its `GATE` states the actual applicability (so `v0_1_12.py` stays active on 0.1.15 too). Currently:
  - `facade_intrinsics.py` — the top-level `tilelang.intrinsics` facade split: present on ≤0.1.12 (native wgmma/tcgen05 passthrough through the facade), removed on ≥0.1.15 (the import must fail cleanly — the wrapper aliases only `tilelang.cuda.*` and must not synthesize it back);
  - `v0_1_11.py` — the legacy layout only 0.1.11 still has: the `tilelang.cuda.target` alias bases itself on `tilelang.utils.target`, the legacy-only name map (`_detect_torch_cuda_arch` / `_cuda_target_from_arch` forward to their ppu counterparts), and the inline cutedsl normalization (0.1.11 calls it directly inside `determine_target`, so under the wrapper a cutedsl dict still resolves to a ppu target carrying the cutedsl key);
  - `v0_1_12.py` — the normalizer-registry surface 0.1.12 introduced (and 0.1.15 keeps): the registry stays free of the cuda/cutedsl entries because their registration carrier (native `tilelang/cuda/target.py`) never executes under the alias, so a cutedsl dict input fails loudly instead of being normalized — different from ≤0.1.11 — while the Target patch's cutedsl-keys exemption keeps an already-keyed cuda target un-rewritten;
  - `v0_1_15.py` — surfaces 0.1.15 added on one backend only: the cuda-only `mma_sm120_macro_generator` (loads natively, base emitter via the alias, SM120 ldmatrix layouts as explicit error stubs), the ppu-only `gemm_mma_blockscaled` (reachable through the cuda-style path under its ppu name), and the new `tilelang.cuda.language` dialect (aliases onto the ppu dialect, common names forward, native dialect submodules load through the alias package path, the ppu dialect's own `cuda.language.tir` dependency resolves, CUDA-only dialect names fail loudly). Adding the next tilelang release means dropping in a `vX_Y_Z.py` module and listing it in `steps/__init__.py` — no runner edits;
- **Arch rules**: device probing (ppu_10 / ppu_15 by hardware generation), explicit sm_80 / sm_89 mappings, ValueError for all other archs, fallback to ppu_10 when no device is present;
- **End to end**: a matmul with `target="cuda"` compiles through the PPU backend (the kernel's target is ppu) and matches the torch reference numerically.

## Known Limitations

- **Strict mode**: this wrapper covers only the "CUDA-style code → PPU" scenario; other targets going through `determine_target` (llvm/metal/rocm…) raise explicit errors — unsupported scenarios fail loudly rather than silently misbehaving. Host-codegen targets constructed internally by the engine during compilation (e.g. `Target("llvm")`) are allowed through and unaffected.
- **Explicit errors for CUDA-only features**: 10 passes with no PPU counterpart (Blackwell / Hopper specific, etc.), `TensorCoreIntrinEmitterWithLadderTransform` (the ladder-transform layout plugin), and — on tilelang ≥0.1.15 — the SM120 block-scale surface (`TensorCoreIntrinEmitterSM120` plus the three `ldmatrix_32x*` layouts it depends on) throw clear RuntimeErrors instead of silent degradation or hard-to-debug AttributeErrors/ImportErrors; the names themselves remain importable (tilelang's own facade re-exports depend on this), and the error is raised only upon actual use.
- **Supported tilelang-sail versions**: this wrapper currently targets tilelang-sail **0.1.11**, **0.1.12** and **0.1.15**, the three versions it was developed and validated against; it has not been tested with other versions.
- **Arch mapping covers only two tiers**: sm_80→ppu_10 and sm_89→ppu_15; every other explicit arch raises ValueError.
- **A Python process restart is required after install/uninstall**: the `.pth` file executes only at interpreter startup.
- After upgrading tilelang-sail, re-running the verification scripts under `tests/` is recommended — the wrapper does not modify its source, but its internal behavior may change across versions. If you observe behavioral drift or new incompatible idioms, feel free to open an issue.
