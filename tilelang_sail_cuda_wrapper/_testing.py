"""tilelang.testing gates: CUDA-styled test decorators accept the PPU.

``tilelang.testing.requires_cuda`` is a plain re-export of tvm's
``requires_cuda`` Feature, whose checks -- ``tvm.runtime.enabled("cuda")``
and ``tvm.device("cuda").exist`` -- are False on a PPU-only build, so
every CUDA-styled test would skip even though the wrapper lets it run on
the PPU.  The patch swaps the gate for one accepting either a real CUDA
runtime or the PPU, the same relaxation tilelang itself applies for ROCm
via ``requires_cuda_or_cdna``.

On tilelang <= 0.1.11 ``tilelang.testing`` also binds the *native*
``target_is_cuda`` predicate from ``tilelang.utils.target`` (the >= 0.1.12
layout imports it from ``tilelang.cuda.target``, which the wrapper already
aliases to the ppu predicate), so ``requires_cuda_or_cdna`` would skip on
the PPU there too; that binding is redirected to the ppu predicate.
"""

from __future__ import annotations

import sys

_SKIP_REASON = (
    "tilelang-sail-cuda-wrapper: no CUDA runtime and no PPU target available"
)


def _patch_tilelang_testing():
    """Widen the CUDA test gates once, after tilelang.testing loads."""
    testing = sys.modules.get("tilelang.testing")
    if testing is None or getattr(testing, "_tilelang_sail_cuda_wrapper_testing", False):
        return
    if not callable(getattr(testing, "requires_cuda", None)):
        # Layout change: no requires_cuda gate to widen, leave the module alone.
        return
    testing._tilelang_sail_cuda_wrapper_testing = True  # type: ignore[attr-defined]

    import pytest

    def _cuda_available():
        # tvm's Feature gate: True only with a CUDA-enabled build/runtime,
        # so a genuine CUDA setup keeps its native behavior.
        try:
            import tvm

            return bool(tvm.runtime.enabled("cuda"))
        except Exception:
            return False

    def _ppu_available():
        # tilelang's own helper (present in both 0.1.11 and 0.1.12):
        # resolves determine_target("auto") and applies the ppu predicate.
        checker = getattr(testing, "_check_is_ppu", None)
        if checker is not None:
            try:
                return bool(checker())
            except Exception:
                return False
        return False

    class _CudaOrPpu:
        """Drop-in replacement for tvm's ``requires_cuda`` Feature.

        Keeps the parts of the Feature protocol tilelang itself relies on:
        bare decoration, the ``support_required=`` keyword form, and
        ``.marks()`` (used by ``requires_cuda_compute_version`` to inherit
        this gate).
        """

        def __call__(self, func=None, *, support_required="compile-and-run"):
            if not callable(func):
                # Feature called with keyword-only arguments
                # (e.g. requires_cuda(support_required="optional")): behave
                # as the plain decorator.
                return self
            marks = [
                pytest.mark.cuda,
                pytest.mark.gpu,
            ]
            if support_required != "optional":
                marks.append(
                    pytest.mark.skipif(
                        not (_cuda_available() or _ppu_available()),
                        reason=_SKIP_REASON,
                    )
                )
            compose = getattr(testing, "_compose", None)
            if compose is not None:
                return compose([func], marks)
            for mark in reversed(marks):
                func = mark(func)
            return func

        def marks(self):
            if _cuda_available() or _ppu_available():
                return []
            return [pytest.mark.skipif(True, reason=_SKIP_REASON)]

    testing.requires_cuda = _CudaOrPpu()

    # tilelang <= 0.1.11 binds the native target_is_cuda predicate here
    # (>= 0.1.12 imports it from tilelang.cuda.target, already aliased by
    # the wrapper), so requires_cuda_or_cdna would skip on the PPU:
    # redirect the binding to the ppu predicate.
    utils_target = sys.modules.get("tilelang.utils.target")
    if utils_target is not None and getattr(
        testing, "target_is_cuda", None
    ) is getattr(utils_target, "target_is_cuda", None):
        ppu_predicate = getattr(utils_target, "target_is_ppu", None)
        if ppu_predicate is not None:
            testing.target_is_cuda = ppu_predicate
