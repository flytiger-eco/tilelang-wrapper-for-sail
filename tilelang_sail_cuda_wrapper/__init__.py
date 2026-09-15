"""tilelang-sail-cuda-wrapper: runtime CUDA-to-PPU compatibility wrapper for TileLang.

This package is a standalone wheel installed *on top of* an existing
PPU-only TileLang build (no CUDA backend).  Once installed, it
transparently redirects CUDA-style usage to the PPU backend:

* ``import tilelang.cuda`` / ``tilelang.cuda.*`` submodules are aliased to the
  ``tilelang.ppu`` implementation, exposing CUDA-style names
  (``TensorCoreIntrinEmitter``, ``MarkCudaSyncCalls``, ...) that forward to
  the PPU implementations underneath.
* ``tvm.target.Target("cuda", ...)`` constructions are rewritten to PPU
  targets (explicit arch: ``sm_80`` -> ``ppu_10``, ``sm_89`` -> ``ppu_15``;
  anything else raises; an omitted arch is auto-detected from the device,
  defaulting to ``ppu_10`` when no device is reachable),
  so kernels written with ``target="cuda"`` compile for the PPU backend.
* Common transform entries ``tl.transform.LowerTileOp`` /
  ``tl.transform.LayoutInference`` are routed to the PPU-specific passes when
  the IRModule targets the PPU backend.
* Any other target kind (``llvm``, ``metal``, ``rocm``, ...) raises an error:
  this wrapper is only intended for CUDA-style code running on PPU.

Activation happens through a ``.pth`` file executed by the ``site`` machinery
at interpreter startup, which registers a ``sys.meta_path`` finder.  No
background thread or watcher is involved: the finder only answers ``import``
requests for ``tilelang.cuda`` and wraps the loaders of ``tvm.target`` /
``tilelang.transform`` so patches are applied exactly once, right after those
modules finish loading.

Uninstalling (``pip uninstall tilelang-sail-cuda-wrapper``) fully restores the original
behavior: native ``tilelang.cuda`` imports work again (and fail at call time
on a PPU-only build, as expected) and CUDA targets are no longer rewritten.

This module is intentionally import-light: it must not import ``tilelang`` or
``tvm`` at activation time.
"""

from ._finder import install

__version__ = "0.1.0"

# Register the meta-path finder.  Safe to call multiple times.
install()
