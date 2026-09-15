"""Target semantics: CUDA-to-PPU rewriting, arch mapping, strict validation.

Applied as post-load patches (see ``_finder._WrapperLoader``):

* ``_patch_tvm_target`` rewrites ``Target("cuda", ...)`` constructions to PPU
  targets at construction time -- but only on tilelang's vendored tvm.
* ``_patch_tilelang_utils_target`` wraps ``determine_target`` (in practice
  the entry every jit/compile/autotune target passes through) with strict
  validation: only the 'cuda' and 'ppu' kinds are accepted for user-facing
  code.
"""

from __future__ import annotations

import sys

from ._tvm_ownership import _tvm_belongs_to_tilelang

# ---------------------------------------------------------------------------
# tvm.target patch: rewrite CUDA targets to PPU at construction time
# ---------------------------------------------------------------------------

_CUDA_TO_PPU_ARCH = {80: "ppu_10", 89: "ppu_15"}

_PPU_GENERIC_KEYS = {
    "max_threads_per_block",
    "max_shared_memory_per_block",
    "thread_warp_size",
    "registers_per_block",
    "max_num_threads",
    "host_num_threads",
}


def _map_cuda_arch_to_ppu(arch):
    """Map a CUDA arch to its PPU counterpart.

    Returns None when *arch* is None (omitted): the caller then resolves
    the arch from the device.  sm_80/sm_89 map to ppu_10/ppu_15; every
    other value is rejected with an explicit error instead of guessed.
    """
    import re

    if arch is None:
        return None
    m = re.match(r"^sm_(\d+)[af]?$", str(arch).strip().lower())
    sm = int(m.group(1)) if m else None
    mapped = _CUDA_TO_PPU_ARCH.get(sm)
    if mapped is None:
        raise ValueError(
            f"tilelang-sail-cuda-wrapper: CUDA arch '{arch}' has no PPU "
            "counterpart; the only workable mappings I found are sm_80 -> "
            "ppu_10 and sm_89 -> ppu_15. Omit arch to auto-detect it from "
            "the device."
        )
    return mapped


def _detect_device_ppu_arch():
    """PPU arch detected from the device, mirroring tilelang's own logic.

    Reuses ``tilelang.utils.target._detect_torch_ppu_arch`` when it is
    already loaded: in practice its output matches what ``determine_target``
    resolves for a 'ppu' target, so direct construction and the compile
    entry agree on the arch.  Returns None when detection is unavailable
    (no device, or constructed before tilelang finished importing); the
    caller then falls back to the static ppu_10 default.
    """
    try:
        utils_target = sys.modules.get("tilelang.utils.target")
        if utils_target is not None:
            detect = getattr(utils_target, "_detect_torch_ppu_arch", None)
            if detect is not None:
                return detect()
        hgcc = sys.modules.get("tilelang.contrib.hgcc")
        if hgcc is not None:
            return f"{hgcc.get_target_arch(hgcc.get_target_compute_version())}"
    except Exception:
        pass
    return None


def _ppu_target_dict(cuda_arch):
    """Build the ppu target dict for a CUDA target carrying *cuda_arch*.

    An omitted arch resolves from the device; when detection is
    unavailable too (no reachable device), it defaults to ``ppu_10`` --
    the same idea as TVM defaulting a deviceless CUDA build to ``sm_50``.
    """
    arch = _map_cuda_arch_to_ppu(cuda_arch)
    if arch is None:
        arch = _detect_device_ppu_arch()
    if arch is None:
        arch = "ppu_10"
    return {"kind": "ppu", "arch": arch}


def _rewrite_cuda_target_to_ppu(target):
    """Rewrite a CUDA target (str/dict/Target) to PPU; leave others untouched.

    Note: non-CUDA/non-PPU kinds (e.g. the ``llvm``/``c`` host targets that
    show up internally while compiling) must pass through unmodified here.  Strict rejection of user-facing non-PPU targets happens
    in ``determine_target`` (see ``_patch_tilelang_utils_target``).

    Arch rules: sm_80 -> ppu_10, sm_89 -> ppu_15, anything else raises;
    an omitted arch is resolved from the device (same helper
    ``determine_target`` uses), falling back to ppu_10 when no device
    is reachable (mirrors TVM's deviceless sm_50 default for CUDA).
    """
    Target = sys.modules["tvm.target"].Target

    if isinstance(target, Target):
        if target.kind.name != "cuda" or "cutedsl" in target.keys:
            return target
        target = dict(target.export())

    if isinstance(target, dict):
        if target.get("kind") != "cuda":
            return target
        if "cutedsl" in target.get("keys", ()):
            return target
        new_dict = _ppu_target_dict(target.get("arch"))
        for k, v in target.items():
            if k in ("kind", "arch", "keys"):
                continue
            if k in _PPU_GENERIC_KEYS:
                new_dict[k] = v
        return new_dict

    if isinstance(target, str):
        s = target.strip()
        if not s:
            return target
        parts = s.split(None, 1)
        if parts[0] != "cuda":
            return target
        rest = parts[1] if len(parts) > 1 else ""
        opts = {}
        for item in rest.split():
            if "=" in item:
                k, v = item.split("=", 1)
                opts[k.lstrip("-").strip()] = v.strip()
        new_dict = _ppu_target_dict(opts.get("arch"))
        for k, v in opts.items():
            if k in ("kind", "arch"):
                continue
            if k in _PPU_GENERIC_KEYS:
                new_dict[k] = v
        return Target(new_dict)

    return target


def _patch_tvm_target():
    """Patch ``tvm.target.Target.__init__`` once, after tvm.target loads."""
    if not _tvm_belongs_to_tilelang():
        # Not tilelang's bundled tvm (e.g. a standalone tvm the user imported
        # before tilelang): hands off entirely.
        return
    tvm_target = sys.modules["tvm.target"]
    Target = tvm_target.Target

    if getattr(Target.__init__, "_tilelang_sail_cuda_wrapper_patched", False):
        return

    original_init = Target.__init__

    def patched_init(self, target=None, host=None, *args, **kwargs):
        target = _rewrite_cuda_target_to_ppu(target)
        if host is not None:
            host = _rewrite_cuda_target_to_ppu(host)
        return original_init(self, target, host, *args, **kwargs)

    patched_init._tilelang_sail_cuda_wrapper_patched = True  # type: ignore[attr-defined]
    Target.__init__ = patched_init


# ---------------------------------------------------------------------------
# tilelang.utils.target patch: strict user-facing target validation
# ---------------------------------------------------------------------------

_STRICT_TARGET_ERROR = (
    "tilelang-sail-cuda-wrapper: target kind '{kind}' is not supported. "
    "This wrapper only aims to run CUDA-style kernels on the PPU backend "
    "(targets 'cuda' and 'ppu'); anything else fails loudly instead of "
    "silently misbehaving. Uninstall it to restore native behavior."
)


def _extract_target_kind(result):
    """Best-effort extraction of the target kind from determine_target's result."""
    Target = sys.modules["tvm.target"].Target
    if isinstance(result, Target):
        return result.kind.name
    if isinstance(result, dict):
        kind = result.get("kind")
        return kind if isinstance(kind, str) else None
    if isinstance(result, str):
        return result.strip().split(None, 1)[0] if result.strip() else None
    return None


def _patch_tilelang_utils_target():
    """Wrap ``determine_target`` once, after tilelang.utils.target loads.

    In practice every target handed to jit/compile/autotune ends up here,
    which makes this the single place to validate the user's target.
    Strict mode: only 'cuda' (rewritten to ppu downstream) and 'ppu' are
    accepted; anything else raises.  The host targets the engine constructs
    on its own ('llvm'/'c') do not pass through here and stay functional.
    """
    utils_target = sys.modules["tilelang.utils.target"]

    original = getattr(utils_target, "determine_target", None)
    if original is None or getattr(original, "_tilelang_sail_cuda_wrapper_strict", False):
        return

    def determine_target_strict(target="auto", return_object=False):
        result = original(target, return_object)
        kind = _extract_target_kind(result)
        if kind not in ("cuda", "ppu"):
            raise ValueError(_STRICT_TARGET_ERROR.format(kind=kind))
        return result

    determine_target_strict._tilelang_sail_cuda_wrapper_strict = True  # type: ignore[attr-defined]
    determine_target_strict.__name__ = original.__name__
    determine_target_strict.__qualname__ = original.__qualname__
    determine_target_strict.__doc__ = original.__doc__
    determine_target_strict.__wrapped__ = original  # type: ignore[attr-defined]
    utils_target.determine_target = determine_target_strict

    # tilelang.utils re-exports determine_target; keep both views in sync.
    utils_pkg = sys.modules.get("tilelang.utils")
    if utils_pkg is not None and getattr(utils_pkg, "determine_target", None) is original:
        utils_pkg.determine_target = determine_target_strict
