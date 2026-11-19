"""The execution-backend split: ``nvrtc`` vs the PPU counterpart ``hgrtc``.

tilelang >= 0.1.15 routes every execution-backend request through
``BackendModule.resolve_execution_backend`` (tilelang.backend.module).  The
PPU's runtime-compilation backend ``hgrtc`` is the analogue of CUDA's
NVRTC (tilelang's own adapter docstring says so), and CUDA-styled code asks
for ``execution_backend="nvrtc"``.

Both sides of the split are checked, gated by what the running tilelang
actually ships:

* the module ``tilelang.backend.module`` absent (<= 0.1.12): the whole step
  is inert — those releases resolve execution backends elsewhere and the
  wrapper has nothing to patch;
* module present, ppu backend shipping an ``hgrtc`` spec (tilelang dev /
  future releases): the wrapper must forward an ``"nvrtc"`` request to the
  ``hgrtc`` spec — same resolved spec object the ``"hgrtc"`` request yields,
  native availability errors prefixed with the forwarding note, and other
  requests ("auto" / "tvm_ffi" / "hgrtc" itself) untouched;
* module present, ppu backend without ``hgrtc`` (0.1.15): the wrapper must
  NOT synthesize a backend the tilelang does not ship — the request keeps
  the native loud "Invalid execution backend 'nvrtc'" error.

Note the end-to-end path after a successful forward: the resolved spec's
name is "hgrtc", so ``TorchFunction._compile_and_create_adapter``
dispatches to ``HGRTCKernelAdapter`` with the exact kwargs the nvrtc branch
uses (verified statically: both branches pass identical arguments), and
``BackendContext.__post_init__`` validates the spec against the same ppu
backend as usual.  The resolve-layer assertions below therefore pin the
only seam the wrapper owns; the adapter itself is native code.
"""

from __future__ import annotations

import importlib.util
import sys

GATE = "tilelang >= 0.1.15 (BackendModule execution-backend resolution)"


def applies(ctx):
    # Probe-first: the module the patch targets decides whether the surface
    # exists at all (0.1.11/0.1.12 resolve execution backends elsewhere).
    try:
        return importlib.util.find_spec("tilelang.backend.module") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def run(ctx):
    report = ctx.report
    import tilelang.backend.module as backend_module

    # The patch must have landed on the class by now (the loader wrapped by
    # the finder applies it right after the module finishes executing).
    assert getattr(
        backend_module.BackendModule, "_tilelang_sail_cuda_wrapper_patched__", False
    ), "tilelang.backend.module loaded but resolve_execution_backend was not patched"

    ppu_backend = backend_module.get_backend("ppu")

    # A ppu target for the resolver: use the same entry all tilelang code
    # uses, so the resolver's own target matching stays native.
    from tilelang.backend.target import determine_target

    ppu_target = determine_target("auto", return_object=True)

    # Requests the wrapper must never touch resolve natively.
    spec = ppu_backend.resolve_execution_backend("auto", ppu_target)
    assert spec.name == "tvm_ffi", (
        f"'auto' resolved to {spec.name!r}, expected the native default"
    )
    spec = ppu_backend.resolve_execution_backend("tvm_ffi", ppu_target)
    assert spec.name == "tvm_ffi", "'tvm_ffi' must resolve to itself"

    # Which side of the nvrtc/hgrtc split does this tilelang ship?
    has_hgrtc = any(s.name == "hgrtc" for s in ppu_backend.execution_backends)

    if not has_hgrtc:
        # 0.1.15 side: no PPU counterpart exists yet — the wrapper must not
        # synthesize one; the native loud error names what was asked.
        try:
            ppu_backend.resolve_execution_backend("nvrtc", ppu_target)
            report("FAIL: 'nvrtc' unexpectedly accepted without an hgrtc spec")
            sys.exit(1)
        except ValueError as err:
            assert "tilelang-sail-cuda-wrapper" not in str(err), (
                "wrapper rewrote a request it must pass through untouched: "
                f"{err}"
            )
            assert "'nvrtc'" in str(err), (
                f"native error does not name the requested backend: {err}"
            )
        report("no hgrtc spec on this tilelang: 'nvrtc' keeps the native"
               " loud rejection (wrapper does not synthesize)")
        return

    # dev / future side: the forward must yield the exact spec the
    # 'hgrtc' request itself would yield — or, when the hgrtc spec exists
    # but is unavailable, the availability error must carry the
    # forwarding note naming both ends.
    try:
        forwarded = ppu_backend.resolve_execution_backend("nvrtc", ppu_target)
    except ValueError as err:
        assert "tilelang-sail-cuda-wrapper" in str(err), (
            f"forwarded-unavailable error lacks the forwarding note: {err}"
        )
        assert "'hgrtc'" in str(err) and "'nvrtc'" in str(err), (
            f"forwarded-unavailable error does not name both ends: {err}"
        )
        report("execution_backend forward OK: 'nvrtc' -> 'hgrtc' with"
               " the availability error prefixed by the forwarding note")
        return
    direct = ppu_backend.resolve_execution_backend("hgrtc", ppu_target)
    assert forwarded is direct, (
        "'nvrtc' forward did not resolve to the 'hgrtc' spec object"
    )
    assert forwarded.name == "hgrtc", (
        f"'nvrtc' resolved to {forwarded.name!r}, expected 'hgrtc'"
    )
    report("execution_backend forward OK: 'nvrtc' -> 'hgrtc' spec"
           " (identical to the direct request)")
