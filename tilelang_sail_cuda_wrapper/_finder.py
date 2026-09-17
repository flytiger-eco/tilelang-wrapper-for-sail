"""The meta-path finder: interception skeleton and activation entry.

Only this machinery runs at interpreter startup (via the ``.pth`` file):
registering the finder is O(1) and imports nothing heavy.  All actual work
happens lazily when the user imports ``tilelang.cuda``, ``tvm.target`` or
``tilelang.transform`` / ``tilelang.utils.target``.

The implementation is split by responsibility:

* ``_aliases``   -- synthetic ``tilelang.cuda.*`` modules forwarding to
  ``tilelang.ppu.*`` (name mapping, PEP 562 lazy forwarding).
* ``_targets``   -- target semantics: CUDA->PPU rewriting, arch mapping,
  and strict validation of user-facing targets.
* ``_testing``   -- widens tilelang.testing's CUDA test gates so
  CUDA-styled tests run (instead of skipping) on the PPU.
* ``_passes``    -- routing of common transform passes to PPU-specific ones.
* ``_tvm_ownership`` -- whether a loaded ``tvm`` is tilelang's vendored
  copy (only that one may be patched).

This module wires them together: ``find_spec`` answers import requests,
``_WrapperLoader`` applies a post-load patch, and ``install`` registers
everything.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys

from ._aliases import _make_cuda_alias
from ._passes import _patch_tilelang_transform
from ._targets import (
    _patch_tvm_target,
    _patch_tilelang_utils_target,
    _patch_tilelang_backend_target,
)
from ._testing import _patch_tilelang_testing

_FINDER = None  # singleton, guarded by install()

# Modules whose loaders we wrap so a patch runs right after they finish
# loading.  Note: ``import tilelang`` already pulls in tilelang.transform and
# whichever target module the running tilelang ships with, so wrapping these
# covers the normal paths.
#
# tilelang moved the user-facing target entry across versions: <= 0.1.11
# exposes it as tilelang.utils.target.determine_target, >= 0.1.12 as
# tilelang.backend.target.determine_target.  Only the module that actually
# exists gets imported, so both entries can stay registered: on each
# version exactly one loader is ever wrapped and the other entry is inert.
# tilelang.testing is separate: it loads when user tests import it (not as
# part of ``import tilelang``), and its patch must run before those tests
# use the requires_cuda gate.
_WRAP_LOAD_MODULES = {
    "tvm.target": _patch_tvm_target,
    "tilelang.transform": _patch_tilelang_transform,
    "tilelang.utils.target": _patch_tilelang_utils_target,
    "tilelang.backend.target": _patch_tilelang_backend_target,
    "tilelang.testing": _patch_tilelang_testing,
}

# Reentrancy guard: while a patch runs it may import modules; those imports
# must not be intercepted again.
_patch_in_progress: dict[str, bool] = {}

# Submodules served by dedicated cross-version builders whose backing
# module moves between tilelang versions (tilelang.cuda.target lives in
# tilelang.ppu.target on >= 0.1.12 but tilelang.utils.target on <= 0.1.11,
# and the native module does not exist at all on the latter): a simple
# ppu-existence probe cannot decide these, so they are always intercepted
# and the builder picks the right base.
_SPECIAL_ALIAS_MODULES = {"tilelang.cuda.target"}


def _delegated_find_spec(finder_self, fullname, path):
    """Find the spec by asking every other meta-path finder (ours excluded).

    Delegating to the remaining finders (instead of only PathFinder) keeps the
    wrap working under editable installs and custom finders.
    """
    for finder in sys.meta_path:
        if finder is finder_self or finder is _FINDER:
            continue
        find_spec = getattr(finder, "find_spec", None)
        if find_spec is None:
            continue
        spec = find_spec(fullname, path)
        if spec is not None:
            return spec
    return None


class _WrapperLoader(importlib.abc.Loader):
    """Delegates to the real loader, then applies a post-load patch."""

    def __init__(self, base_loader, after_exec):
        self._base_loader = base_loader
        self._after_exec = after_exec

    def create_module(self, spec):
        return self._base_loader.create_module(spec)

    def exec_module(self, module):
        self._base_loader.exec_module(module)
        self._after_exec()

    def __getattr__(self, item):
        return getattr(self._base_loader, item)


class _AliasLoader(importlib.abc.Loader):
    """Creates the tilelang.cuda.* alias module (delegates to tilelang.ppu)."""

    def __init__(self, fullname):
        self._fullname = fullname

    def create_module(self, spec):
        return _make_cuda_alias(self._fullname, spec)

    def exec_module(self, module):
        # Nothing to execute: the alias already mirrors the ppu module.
        return


class _CudaPpuFinder(importlib.abc.MetaPathFinder):
    """Intercepts ``tilelang.cuda.*`` and wraps loaders of patch targets."""

    def find_spec(self, fullname, path=None, target=None):
        # 1. Alias every tilelang.cuda.* submodule onto tilelang.ppu.* --
        #    but only those the ppu backend actually ships. Submodules it
        #    does not (the wgmma/tcgen05 generation-specific macro
        #    generators, the older sm70/sm75 variants, ...) fall through
        #    to their native tilelang.cuda files: the exact behavior of a
        #    wrapper-less environment, and what tilelang's own top-level
        #    re-exports (tilelang.intrinsics) expect to load.
        if fullname == "tilelang.cuda" or fullname.startswith("tilelang.cuda."):
            if _patch_in_progress.get(fullname):
                return None
            if fullname != "tilelang.cuda" and fullname not in _SPECIAL_ALIAS_MODULES:
                ppu_fullname = "tilelang.ppu" + fullname[len("tilelang.cuda"):]
                try:
                    ppu_spec = importlib.util.find_spec(ppu_fullname)
                except (ImportError, AttributeError, ValueError):
                    ppu_spec = None
                if ppu_spec is None:
                    return None
            return importlib.machinery.ModuleSpec(fullname, _AliasLoader(fullname))

        # 2. Wrap loaders so patches run right after these modules load.
        if fullname in _WRAP_LOAD_MODULES:
            if _patch_in_progress.get(fullname):
                return None
            real_spec = _delegated_find_spec(self, fullname, path)
            if real_spec is None:
                return None
            real_spec.loader = _WrapperLoader(real_spec.loader, _WRAP_LOAD_MODULES[fullname])
            return real_spec

        return None


def install() -> None:
    """Register the meta-path finder and apply patches (idempotent).

    Called at interpreter startup via the ``.pth`` file, or manually via
    ``import tilelang_sail_cuda_wrapper``.  If some targets were already loaded by the
    time this runs (manual activation in a warm interpreter), patches are
    applied immediately and a loaded native ``tilelang.cuda`` is replaced by
    the alias.
    """
    global _FINDER
    if _FINDER is None:
        _FINDER = _CudaPpuFinder()
    if _FINDER not in sys.meta_path:
        # Insert at the front so we win over the native tilelang.cuda package.
        sys.meta_path.insert(0, _FINDER)

    # Warm-interpreter fallback: patch everything already loaded.
    for modname, patch in _WRAP_LOAD_MODULES.items():
        if modname in sys.modules:
            _patch_in_progress[modname] = True
            try:
                patch()
            except Exception:
                pass  # a failed patch must never break interpreter startup
            finally:
                _patch_in_progress[modname] = False

    if "tilelang.cuda" in sys.modules and not getattr(
        sys.modules["tilelang.cuda"], "_tilelang_sail_cuda_wrapper_alias_of__", None
    ):
        # Native tilelang.cuda was loaded before activation: swap in the alias.
        try:
            alias = _make_cuda_alias("tilelang.cuda")
            setattr(sys.modules.get("tilelang"), "cuda", alias)
        except Exception:
            pass
