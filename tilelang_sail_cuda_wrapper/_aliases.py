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
    # tilelang >= 0.1.12: the target helpers live in tilelang.ppu.target.
    "target": {
        "target_is_cuda": "target_is_ppu",
        "check_cuda_availability": "check_ppu_availability",
        "_detect_torch_cuda_arch": "_detect_ppu_arch",
        "_cuda_target_from_arch": "_ppu_target_from_arch",
        "_detect_cuda_target": "_detect_ppu_target",
        "normalize_cuda_target": "normalize_ppu_target",
    },
}

# tilelang <= 0.1.11 ships no tilelang.ppu.target; the same helpers live
# flat in tilelang.utils.target under their own names (target_is_ppu,
# check_ppu_availability, _detect_torch_ppu_arch, _ppu_target_from_arch).
# Names absent from this map have no 0.1.11 counterpart and fail loudly.
_LEGACY_TARGET_ALIASES = {
    "target_is_cuda": "target_is_ppu",
    "check_cuda_availability": "check_ppu_availability",
    "_detect_torch_cuda_arch": "_detect_torch_ppu_arch",
    "_cuda_target_from_arch": "_ppu_target_from_arch",
}

# CUDA-style generation/feature predicates with no ppu-side python wrapper
# on any tilelang version: forwarded to the same FFI calls the native
# tilelang.cuda.target module makes, so the answers stay exactly what the
# running tilelang itself computes (e.g. TargetIsHopper(ppu) is False on
# both versions; TargetHasLdmatrix differs upstream and that difference
# is preserved, not smoothed over).
_FFI_TARGET_PREDICATES = {
    "target_is_volta": "TargetIsVolta",
    "target_is_turing": "TargetIsTuring",
    "target_is_ampere": "TargetIsAmpere",
    "target_is_hopper": "TargetIsHopper",
    "target_is_sm120": "TargetIsSM120",
    "target_has_async_copy": "TargetHasAsyncCopy",
    "target_has_ldmatrix": "TargetHasLdmatrix",
    "target_has_stmatrix": "TargetHasStmatrix",
    "target_has_bulk_copy": "TargetHasBulkCopy",
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


def _make_cuda_target_alias(fullname: str, spec=None) -> types.ModuleType:
    """Create the ``tilelang.cuda.target`` alias, portable across tilelang versions.

    tilelang moved the target helpers across versions: >= 0.1.12 keeps them
    in ``tilelang.ppu.target``, <= 0.1.11 flat in ``tilelang.utils.target``.
    Pick whichever backend module exists and forward CUDA-style names onto
    it.  Generation/feature predicates (target_is_hopper, ... ) have no
    ppu-side python wrapper on any version; they go straight to the tilelang
    FFI (the exact calls the native cuda module would make).
    """
    try:
        base = importlib.import_module("tilelang.ppu.target")  # >= 0.1.12
        name_map = dict(_ALIASES["target"])
    except ImportError:
        base = importlib.import_module("tilelang.utils.target")  # <= 0.1.11
        name_map = dict(_LEGACY_TARGET_ALIASES)

    alias = types.ModuleType(fullname)
    # Both backends are plain modules (no submodules to re-export), so the
    # alias deliberately stays a non-package too.
    alias.__package__ = fullname
    alias.__dict__["_tilelang_sail_cuda_wrapper_alias_of__"] = base
    if spec is not None:
        alias.__spec__ = spec

    def _forward(name):
        # 1. CUDA-style name -> backend implementation (ppu.target on
        #    >= 0.1.12, utils.target on <= 0.1.11).
        ppu_name = name_map.get(name)
        if ppu_name is not None:
            try:
                return getattr(base, ppu_name)
            except AttributeError:
                raise AttributeError(
                    f"module {fullname!r} has no attribute {name!r} "
                    f"(backend implementation {ppu_name!r} not found)"
                ) from None
        # 2. Generation/feature predicates: same FFI calls the native
        #    tilelang.cuda.target module makes.
        ffi_name = _FFI_TARGET_PREDICATES.get(name)
        if ffi_name is not None:
            from tilelang import _ffi_api

            fn = getattr(_ffi_api, ffi_name, None)
            if fn is None:
                raise AttributeError(
                    f"module {fullname!r} has no attribute {name!r} "
                    f"(FFI {ffi_name!r} not registered by this tilelang)"
                )

            def _predicate(target, *args, _fn=fn, **kwargs):
                return _fn(target, *args, **kwargs)

            _predicate.__name__ = name
            _predicate.__qualname__ = name
            return _predicate
        # 3. Plain same-name forwarding (target_is_ppu and friends on
        #    >= 0.1.12; the full flat helper set on <= 0.1.11).
        try:
            return getattr(base, name)
        except AttributeError:
            raise AttributeError(
                f"module {fullname!r} has no attribute {name!r}"
            ) from None

    def _dir():
        return sorted(set(dir(base)) | set(name_map) | set(_FFI_TARGET_PREDICATES))

    alias.__dict__["__getattr__"] = _forward
    alias.__dict__["__dir__"] = _dir

    # Register in sys.modules and attach to the parent so that both
    # ``import tilelang.cuda.target`` and ``from tilelang.cuda import target``
    # work.
    sys.modules[fullname] = alias
    parent_name, _, child = fullname.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child, alias)
    return alias


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
    if ppu_fullname == "tilelang.ppu.target":
        # The one submodule whose backing module moves across tilelang
        # versions; route through a dedicated builder.
        return _make_cuda_target_alias(fullname, spec)
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
