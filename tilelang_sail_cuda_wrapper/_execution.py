"""Execution-backend forwarding: CUDA-style requests onto PPU counterparts.

tilelang >= 0.1.15 moved backend selection behind ``BackendModule`` +
``ExecutionBackendSpec`` (tilelang.backend.module): every jit / autotune /
compile entry converges on ``BackendModule.resolve_execution_backend``,
which turns the user's ``execution_backend=`` request into a resolved spec.

The PPU's runtime-compilation backend is ``hgrtc`` (device sources compiled
to ``.hgbin`` by ``hgcc``, loaded through libhggc.so) — tilelang's own docs
call it "the analogue of CUDA's NVRTC on the PPU toolchain", and its adapter
(``tilelang.jit.adapter.hgrtc``) mirrors the nvrtc one method for method.
CUDA-styled code therefore asks for ``execution_backend="nvrtc"``; on a ppu
target that request would die on the native "Invalid execution backend
'nvrtc' for target 'ppu'" error even though the exact intended backend
exists under the ppu name.

The patch here rewrites an ``"nvrtc"`` request resolved against the **ppu**
backend to ``"hgrtc"`` and lets the native resolver do the rest — including
its availability check and its loud errors.  Gated so nothing else moves:

* only the backend module named ``"ppu"`` is touched (other registered
  backends keep native semantics);
* only when the ppu backend actually declares an ``hgrtc`` spec — on
  tilelang releases that predate the hgrtc surface (0.1.15 ships
  BackendModule but no hgrtc spec) the request passes through untouched
  and the native "Invalid execution backend 'nvrtc'" error is the honest
  answer: there is no PPU counterpart to forward to yet.

CUDA runtime-compilation surfaces with **no** PPU counterpart in any form
(``tilelang.contrib.nvrtc``'s ``compile_cuda`` producing PTX/CUBIN text, the
``tilelang.jit.adapter.nvrtc`` package) are deliberately NOT stubbed or
forwarded: their semantics (NVRTC compiling to PTX/CUBIN) have no hgrtc
equivalent, and their native failures are already loud import errors.
"""

from __future__ import annotations

import sys

# CUDA-style execution-backend name -> PPU counterpart, applied only to the
# backend module named "ppu" (see the patch below).
_EXECUTION_BACKEND_ALIASES = {
    "nvrtc": "hgrtc",
}


def _patch_tilelang_backend_module():
    """Wrap ``BackendModule.resolve_execution_backend`` after it loads.

    tilelang.backend.module exists from 0.1.15 on (older releases resolve
    execution backends elsewhere and simply never load this module, so the
    patch entry stays inert on them).  Only the class method is patched:
    every call path (jit frontend, autotuner, cache/database loading)
    converges on this resolver, and ``BackendContext.__post_init__`` then
    validates the resolved spec against the same backend as usual.
    """
    mod = sys.modules.get("tilelang.backend.module")
    if mod is None:
        return
    backend_cls = getattr(mod, "BackendModule", None)
    if backend_cls is None:
        return
    if getattr(backend_cls, "_tilelang_sail_cuda_wrapper_patched__", False):
        return

    original = backend_cls.resolve_execution_backend

    def resolve_execution_backend(self, requested, target):
        requested_name = None if requested is None else str(requested).lower()
        forwarded = None
        if self.name == "ppu":
            forwarded = _EXECUTION_BACKEND_ALIASES.get(requested_name)
            if forwarded is not None and not any(
                spec.name == forwarded for spec in self.execution_backends
            ):
                # This tilelang's ppu backend does not ship the counterpart
                # spec yet (e.g. 0.1.15 has BackendModule but no hgrtc):
                # pass the original request through so the native error
                # names what the user actually asked for.
                forwarded = None
        if forwarded is None:
            return original(self, requested, target)
        try:
            return original(self, forwarded, target)
        except ValueError as err:
            # The counterpart spec exists but is unavailable (e.g. hgrtc
            # needs the PPU SDK).  The native message names 'hgrtc'; prefix
            # the forwarding so the error is understandable from the
            # 'nvrtc' request the user actually wrote.
            raise ValueError(
                f"tilelang-sail-cuda-wrapper: execution_backend "
                f"{requested_name!r} was forwarded to {forwarded!r} "
                f"(the PPU counterpart). {err}"
            ) from None

    backend_cls.resolve_execution_backend = resolve_execution_backend
    backend_cls._tilelang_sail_cuda_wrapper_patched__ = True
