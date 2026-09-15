"""Module aliases: ``tilelang.cuda.*`` forwarding onto ``tilelang.ppu.*``.

The finder (see ``_finder``) serves every ``tilelang.cuda`` import with a
synthetic alias module built here.  Attributes are forwarded lazily through
a module-level ``__getattr__`` (PEP 562), with CUDA-style names mapped to
their PPU implementations and CUDA-only APIs exposed as explicit error
stubs.
"""

from __future__ import annotations

import importlib
import sys
import types

# CUDA-style name -> PPU implementation name, injected onto the alias modules
# keyed by the submodule path relative to the backend package.
_ALIASES = {
    "intrinsics": {
        "TensorCoreIntrinEmitter": "PPUTensorCoreIntrinEmitter",
        "SparseTensorCoreIntrinEmitter": "PPUSparseTensorCoreIntrinEmitter",
    },
    "intrinsics.macro": {
        "TensorCoreIntrinEmitter": "PPUTensorCoreIntrinEmitter",
        "SparseTensorCoreIntrinEmitter": "PPUSparseTensorCoreIntrinEmitter",
    },
    "transform": {
        "MarkCudaSyncCalls": "MarkPpuSyncCalls",
    },
}

# CUDA-only transform APIs with no PPU counterpart found: expose explicit
# errors instead of letting users hit an obscure AttributeError.
_UNSUPPORTED_TRANSFORM = {
    "LowerBlackwell2SM": "Blackwell 2SM lowering",
    "LowerHopperIntrin": "Hopper intrinsics lowering",
    "LowerSharedTmem": "shared TMEM lowering",
    "LowerSharedBarrier": "shared barrier lowering (PTX mbarrier)",
    "FuseMBarrierArriveExpectTx": "TMA mbarrier fusing",
    "InjectFenceProxy": "fence.proxy injection",
    "InjectTcgen05Fence": "tcgen05 fence injection",
    "AnnotateWarpGroupRegAlloc": "warp-group register allocation",
    "ProducerConsumerWarpSpecialized": "warp specialization",
    "ProducerConsumerWarpSpecializedTiled": "warp specialization",
}


def _make_unsupported(name: str, feature: str):
    def _raise(*args, **kwargs):
        raise RuntimeError(
            f"tilelang-sail-cuda-wrapper: '{name}' ({feature}) is a CUDA-only "
            "feature with no PPU counterpart I could find."
        )

    _raise.__name__ = name
    _raise.__qualname__ = name
    return _raise


def _make_cuda_alias(fullname: str, spec=None) -> types.ModuleType:
    """Create a ``tilelang.cuda.*`` module forwarding to ``tilelang.ppu.*``.

    Attributes are forwarded lazily through a module-level ``__getattr__``
    (PEP 562) instead of copying a ``__dict__`` snapshot at creation time.
    Snapshots are fundamentally broken here: at least one ppu submodule
    (under ``intrinsics.macro``) imports back from ``tilelang.cuda.*``,
    re-entering this finder while ``tilelang.ppu.*`` is still a
    partially-initialized module, and a snapshot taken then freezes that
    partial state in ``sys.modules`` forever.  Lazy forwarding sidesteps
    the problem because a module is filled in place -- every later lookup
    sees its final state.
    """
    ppu_fullname = "tilelang.ppu" + fullname[len("tilelang.cuda"):]
    # May still be partially initialized when re-entered from tilelang.ppu's
    # own import chain; harmless, see the docstring.
    ppu_mod = importlib.import_module(ppu_fullname)

    rel = ppu_fullname[len("tilelang.ppu"):].lstrip(".")
    name_map = _ALIASES.get(rel, {})

    alias = types.ModuleType(fullname)
    # Let submodule imports (tilelang.cuda.intrinsics.macro, ...) resolve via
    # the real ppu package paths if anything bypasses our finder.
    alias.__path__ = getattr(ppu_mod, "__path__", None)  # type: ignore[attr-defined]
    alias.__file__ = getattr(ppu_mod, "__file__", None)
    alias.__package__ = fullname
    # Mark for introspection and keep a handle to the underlying module.
    alias.__dict__["_tilelang_sail_cuda_wrapper_alias_of__"] = ppu_mod
    if spec is not None:
        alias.__spec__ = spec

    def _forward(name):
        # 1. CUDA-style name -> PPU implementation.
        ppu_name = name_map.get(name)
        if ppu_name is not None:
            try:
                return getattr(ppu_mod, ppu_name)
            except AttributeError:
                raise AttributeError(
                    f"module {fullname!r} has no attribute {name!r} "
                    f"(PPU implementation {ppu_name!r} not found)"
                ) from None
        # 2. CUDA-only APIs without a PPU counterpart: explicit error stubs.
        if rel == "transform" and name in _UNSUPPORTED_TRANSFORM:
            return _make_unsupported(name, _UNSUPPORTED_TRANSFORM[name])
        # 3. Plain attribute forwarding.
        try:
            value = getattr(ppu_mod, name)
        except AttributeError:
            raise AttributeError(
                f"module {fullname!r} has no attribute {name!r}"
            ) from None
        # Forward submodule attributes to their aliases so attribute access
        # and ``from tilelang.cuda import X`` never leak a raw tilelang.ppu.*
        # submodule (which would bypass the CUDA-style names above).
        if isinstance(value, types.ModuleType) and value.__name__.startswith("tilelang.ppu."):
            sub_cuda = "tilelang.cuda" + value.__name__[len("tilelang.ppu"):]
            sub = sys.modules.get(sub_cuda)
            if sub is None:
                sub = _make_cuda_alias(sub_cuda)
            return sub
        return value

    def _dir():
        names = set(dir(ppu_mod)) | set(name_map)
        if rel == "transform":
            names |= set(_UNSUPPORTED_TRANSFORM)
        return sorted(names)

    alias.__dict__["__getattr__"] = _forward
    alias.__dict__["__dir__"] = _dir

    # Register in sys.modules and attach to the parent so that both
    # ``import tilelang.cuda.X`` and ``from tilelang.cuda import X`` work.
    sys.modules[fullname] = alias
    parent_name, _, child = fullname.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child, alias)
    return alias
