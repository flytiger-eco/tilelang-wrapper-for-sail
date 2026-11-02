"""Top-level ``tilelang.intrinsics`` facade -- the <=0.1.12 / >=0.1.15 split.

tilelang <= 0.1.12 ships the facade re-exporting CUDA-only names from the
cuda backend: the ppu backend has no counterparts, so the wrapper serves
the facade's from-imports through explicit error stubs (ladder-transform)
and native fall-through files (wgmma / tcgen05).  tilelang >= 0.1.15
removed the facade entirely; the wrapper aliases only ``tilelang.cuda.*``
and must not synthesize it back, so the import has to fail cleanly there.

Both sides are contracts, so this module always applies and ``run``
branches on the probe (not on the parsed version): a fork that kept the
facade on a 0.1.15+ base gets the presence checks, and vice versa.
"""

from __future__ import annotations

import importlib.util
import sys

GATE = "facade split -- both sides checked (present: <= 0.1.12, removed: >= 0.1.15)"


def applies(ctx):
    return True


def run(ctx):
    report = ctx.report
    if importlib.util.find_spec("tilelang.intrinsics") is not None:
        import tilelang.intrinsics
        import tilelang.cuda.intrinsics.macro.wgmma_macro_generator as wgmma
        import tilelang.cuda.intrinsics.macro.tcgen05_macro_generator as tcgen05
        assert hasattr(wgmma, "WGMMADescriptorParams"), "native wgmma re-exports missing"
        assert hasattr(tcgen05, "TCGEN05DescriptorParams"), "native tcgen05 re-exports missing"
        assert tilelang.intrinsics.WGMMATensorCoreIntrinEmitter is wgmma.TensorCoreIntrinEmitter, \
            "facade wgmma re-export differs from the native module"
        assert tilelang.intrinsics.TCGEN05TensorCoreIntrinEmitter is tcgen05.TensorCoreIntrinEmitter, \
            "facade tcgen05 re-export differs from the native module"
        report("tilelang.intrinsics facade OK: native wgmma/tcgen05 passthrough")
    else:
        # The wrapper aliases only tilelang.cuda.*: it must not synthesize
        # the facade tilelang removed, so importing it fails cleanly.
        try:
            import tilelang.intrinsics  # noqa: F401
            report("FAIL: removed tilelang.intrinsics facade unexpectedly importable")
            sys.exit(1)
        except ModuleNotFoundError:
            report("tilelang.intrinsics facade absent (removed upstream):"
                   " import fails cleanly, wrapper does not synthesize it")
