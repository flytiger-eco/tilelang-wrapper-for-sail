"""Pass routing: common transforms dispatched to PPU-specific passes.

Applied as a post-load patch (see ``_finder._WrapperLoader``) once
``tilelang.transform`` finishes loading: ``LowerTileOp`` and
``LayoutInference`` factories are replaced with routers that inspect the
IRModule and pick the PPU-specific pass when any PrimFunc targets the PPU
backend, falling back to the native implementation otherwise.
"""

from __future__ import annotations

import sys

_ROUTED_PASSES = ("LowerTileOp", "LayoutInference")


class _RoutedPass:
    """Route a common transform pass to the PPU-specific implementation.

    When the IRModule contains a PrimFunc whose target kind is ``ppu``, the
    PPU-specific pass is used; otherwise the native implementation is called.
    """

    def __init__(self, pass_name, native_factory):
        self._pass_name = pass_name
        self._native_factory = native_factory

    def __call__(self, ir_module, *args, **kwargs):
        if _ir_module_uses_ppu(ir_module):
            from tilelang.ppu import transform as _ppu_transform

            return getattr(_ppu_transform, self._pass_name)()(ir_module, *args, **kwargs)
        return self._native_factory()(ir_module, *args, **kwargs)


def _ir_module_uses_ppu(ir_module):
    """Return True if any PrimFunc in *ir_module* targets the PPU backend."""
    for gv in ir_module.get_global_vars():
        func = ir_module[gv]
        attrs = getattr(func, "attrs", None)
        if attrs is None:
            continue
        target = attrs.get("target")
        if target is not None and getattr(target, "kind", None) is not None:
            if target.kind.name == "ppu":
                return True
    return False


def _patch_tilelang_transform():
    """Wrap common pass factories once, after tilelang.transform loads."""
    transform_mod = sys.modules["tilelang.transform"]
    ffi_api = transform_mod._ffi_api

    for pass_name in _ROUTED_PASSES:
        current = getattr(transform_mod, pass_name, None)
        if current is None or getattr(current, "_tilelang_sail_cuda_wrapper_routed", False):
            continue
        native_factory = getattr(ffi_api, pass_name)

        def make_routed(pass_name=pass_name, native_factory=native_factory):
            def routed():
                return _RoutedPass(pass_name, native_factory)

            routed.__name__ = pass_name
            routed.__qualname__ = pass_name
            routed.__doc__ = (
                f"{pass_name} (routed: PPU-targeted modules use the PPU-specific "
                f"pass, others use the native implementation)."
            )
            routed._tilelang_sail_cuda_wrapper_routed = True  # type: ignore[attr-defined]
            return routed

        setattr(transform_mod, pass_name, make_routed())
