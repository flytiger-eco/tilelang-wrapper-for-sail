"""Surfaces tilelang 0.1.15 added on one backend only.

Three of them:

* (a) ``mma_sm120_macro_generator`` (cuda-only): loads natively like
  wgmma/tcgen05; its base emitter still resolves through the alias, and
  the SM120-specific ldmatrix layouts it from-imports are explicit error
  stubs -- without the stubs the module would die at import on an
  obscure "cannot import name" error (ppu mma_layout never grew them).
* (b) ``gemm_mma_blockscaled`` (ppu-only): reachable through the
  cuda-style path under its ppu name (no cuda-style name exists on any
  version; the cuda counterpart is the tcgen05-specific module, which
  stays native).
* (c) the ``tilelang.cuda.language`` dialect package (both dialects are
  new in 0.1.15): the cuda-style import aliases onto the ppu dialect,
  common and ppu names forward, the native dialect submodules
  (tir/gemm_op/...) load through the alias package path, the ppu
  dialect's own ``cuda.language.tir`` dependency resolves, and CUDA-only
  dialect names fail loudly (AttributeError names the alias module --
  they are not stubbed: nothing re-exports them into a facade that must
  load, and stubbing ~60 language names would be a maintenance surface
  out of proportion to the confusion saved).  "CUDA-only" here means
  "absent from the ppu dialect": upstream shares the ldg/stg family
  through ``tilelang.language.builtin`` across both dialects, so those
  names forward like common names -- the check probes the ppu dialect
  itself instead of assuming a frozen name list.

Each block is individually gated by what the running tilelang actually
ships (find_spec), mirroring the wrapper's own probe-first selection: on
a fork (or a later release) missing one surface, that block is inert
rather than a failure.
"""

from __future__ import annotations

import importlib.util
import sys

MIN_VERSION = (0, 1, 15)
GATE = "tilelang >= 0.1.15"


def applies(ctx):
    # Version fast path only: the individual surface probes live in run()
    # so a fork missing one surface still verifies the others.
    return ctx.version is not None and ctx.version >= MIN_VERSION


def run(ctx):
    report = ctx.report

    # (a) cuda-only SM120 block-scale macro generator.
    if importlib.util.find_spec(
        "tilelang.cuda.intrinsics.macro.mma_sm120_macro_generator"
    ) is not None:
        import tilelang.cuda.intrinsics.macro.mma_sm120_macro_generator as sm120
        from tilelang.cuda.intrinsics.macro.mma_macro_generator import (
            TensorCoreIntrinEmitter as base_emitter,
        )
        assert "mma_sm120_macro_generator" in sm120.__file__, \
            "sm120 module unexpectedly aliased"
        assert issubclass(sm120.TensorCoreIntrinEmitterSM120, base_emitter), \
            "native sm120 emitter did not resolve its base through the alias"
        try:
            sm120.ldmatrix_32x32_to_shared_16x64_layout_a(0)
            report("FAIL: SM120 layout unexpectedly usable")
            sys.exit(1)
        except RuntimeError:
            report("SM120 ldmatrix layouts stubbed (importable, loud on use)")
        report("sm120 native passthrough OK: base emitter via alias,"
               " SM120 layouts stubbed")
    else:
        report("sm120 macro generator absent on this tilelang: block inert")

    # (b) ppu-only block-scaled GEMM op.
    if importlib.util.find_spec(
        "tilelang.ppu.op.gemm.gemm_mma_blockscaled"
    ) is not None:
        from tilelang.cuda.op.gemm.gemm_mma_blockscaled import (
            PPUGemmMMABlockScaled as bs_cuda_style,
        )
        from tilelang.ppu.op.gemm.gemm_mma_blockscaled import (
            PPUGemmMMABlockScaled as bs_ppu,
        )
        assert bs_cuda_style is bs_ppu, \
            "cuda-style blockscaled path does not resolve to the ppu module"
        report("ppu-only blockscaled reachable via cuda path"
               " (same-name forwarding)")
    else:
        report("ppu blockscaled module absent on this tilelang: block inert")

    # (c) the cuda/ppu language dialects.
    if importlib.util.find_spec("tilelang.cuda.language") is not None:
        import tilelang.cuda.language as lang
        import tilelang.ppu.language as plang
        assert getattr(lang, "_tilelang_sail_cuda_wrapper_alias_of__", None) is plang, \
            "tilelang.cuda.language is not the alias of the ppu dialect"
        assert lang.Tensor is plang.Tensor, \
            "dialect common names do not forward to the ppu dialect"
        # The dialect tir submodule follows the general forwarding rule:
        # aliased when the ppu dialect ships its own tir (tilelang commit
        # 02c59bac added tilelang.ppu.language.tir), native when it does not
        # (the 9a8dece0-era layout the step was first written against).
        if importlib.util.find_spec("tilelang.ppu.language.tir") is not None:
            import tilelang.ppu.language.tir as ptir
            import tilelang.cuda.language.tir as ltir
            # The alias is a distinct synthetic ModuleType, so object identity
            # never holds against the real ppu module; the marker attribute
            # the wrapper plants on every alias is the identity witness.
            assert getattr(
                ltir, "_tilelang_sail_cuda_wrapper_alias_of__", None
            ) is ptir, (
                "dialect tir not aliased despite tilelang.ppu.language.tir existing"
            )
            assert hasattr(ltir, "ptx_ldmatrix"), "ppu dialect tir lost ptx helpers"
            report("dialect tir aliases the ppu tir (ppu-side tir present)")
        else:
            import tilelang.cuda.language.tir as ltir
            assert "cuda/language/tir" in ltir.__file__, \
                "dialect tir submodule unexpectedly aliased"
            assert hasattr(ltir, "ptx_ldmatrix"), "native dialect tir lost ptx helpers"
            report("dialect tir loads natively (ppu-side tir absent)")
        assert getattr(plang, "ptx_mma", None) is not None, \
            "ppu dialect lost its own cuda.language.tir dependency resolution"
        # CUDA-only names must fail loudly through the alias -- but the
        # witness is what the ppu dialect itself exports, not a frozen list:
        # upstream moved the ldg/stg family into the shared
        # tilelang.language.builtin layer and both dialects re-export it
        # now, so those names forward like any common name instead of
        # erroring. A name the ppu dialect ships must resolve through the
        # alias to the very same object; a name it does not must fail
        # loudly (AttributeError, not silent stubbing).
        for dialect_name in ("mbarrier_arrive", "alloc_tmem", "ldg128",
                             "create_tma_descriptor"):
            if hasattr(plang, dialect_name):
                assert getattr(lang, dialect_name) is getattr(
                    plang, dialect_name
                ), f"dialect name {dialect_name!r} forwards but not to the" \
                   " ppu dialect's object"
                continue
            try:
                getattr(lang, dialect_name)
                report(f"FAIL: CUDA-only dialect name '{dialect_name}'"
                       " unexpectedly usable")
                sys.exit(1)
            except AttributeError:
                pass
        report("cuda.language dialect OK: aliases ppu dialect, common names"
               " forward, ppu-absent names fail loudly")
    else:
        report("language dialects absent on this tilelang: block inert")
