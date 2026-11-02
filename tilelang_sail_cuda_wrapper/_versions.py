"""tilelang version detection: which layout is the running tilelang?

The wrapper supports tilelang releases whose internals moved across
versions.  Two layouts exist so far:

* ``"legacy"``  (tilelang <= 0.1.11): the user-facing target entry lives in
  ``tilelang.utils.target.determine_target`` and the PPU arch detection in
  ``tilelang.utils.target._detect_torch_ppu_arch``; there is no
  ``tilelang.ppu.target`` module.
* ``"modern"``  (tilelang >= 0.1.12): the entry moved to
  ``tilelang.backend.target.determine_target`` (normalizer-registry based)
  and the arch detection to ``tilelang.ppu.target._detect_ppu_arch``.

Selection is capability-probed first (``detect_layout``): probes stay
correct on fork builds whose version strings do not parse, and the version
number only accelerates/validates the decision.  ``tilelang.__version__``
is assigned before tilelang's heavy imports in every release examined
(0.1.11/0.1.12/0.1.15), so it is already available by the time any lazy
wrapper decision runs.
"""

from __future__ import annotations

import importlib.util
import re
import sys

# Layout per tilelang release line, consulted by _layout_for_version:
# adding support for a new release line that moves internals again means
# adding one entry here (plus the layout-specific facts below).  A version
# that parses but matches no operator here (e.g. some future 0.1.16 that
# kept the modern layout) needs no entry: probing already covers it.
_LAYOUT_BY_VERSION = {
    "legacy": ("<", (0, 1, 12)),
    "modern": (">=", (0, 1, 12)),
}
_OPS = {
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
}

# Loader-independent facts each layout implies; kept in one place so a new
# layout only needs one entry here plus (when needed) probe fallbacks.
_LAYOUT_TARGET_MODULES = {
    # layout -> (ppu-side target module, legacy flat target module or None)
    "legacy": (None, "tilelang.utils.target"),
    "modern": ("tilelang.ppu.target", "tilelang.utils.target"),
}


def parse_tilelang_version(version_str):
    """Parse a tilelang version string into a (major, minor, patch) tuple.

    Handles the fork suffixes tilelang-for-sail ships with, e.g.
    ``"0.1.15+ppu.1.0.0.git9a8dece0"`` or ``"0.1.11+v0.1.0"``.  Returns
    None when the string does not start with a parseable ``X.Y.Z``.
    """
    if not version_str:
        return None
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(version_str).strip())
    if m is None:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def tilelang_version():
    """The running tilelang's ``__version__`` string, or None.

    Lazy and cached: the wrapper activates before tilelang imports, so the
    first callers (post-load patches) find it set; earlier callers get
    None and must not treat that as an error.
    """
    tl = sys.modules.get("tilelang")
    version = getattr(tl, "__version__", None) if tl is not None else None
    return version


def tilelang_version_tuple():
    """(major, minor, patch) of the running tilelang, or None."""
    return parse_tilelang_version(tilelang_version())


def _layout_for_version(version_tuple):
    """Map a parsed version onto a known layout, or None when unknown.

    Walks _LAYOUT_BY_VERSION in insertion order and returns the first
    matching line; None means "unknown version, rely on probing".
    """
    if version_tuple is None:
        return None
    for layout, (op, boundary) in _LAYOUT_BY_VERSION.items():
        if _OPS[op](version_tuple, boundary):
            return layout
    return None


def _probe_layout():
    """Detect the layout from what the running tilelang actually ships.

    The authoritative mechanism: works on any fork regardless of its
    version string.
    """
    try:
        if importlib.util.find_spec("tilelang.ppu.target") is not None:
            return "modern"
    except (ImportError, AttributeError, ValueError):
        pass
    try:
        if importlib.util.find_spec("tilelang.utils.target") is not None:
            return "legacy"
    except (ImportError, AttributeError, ValueError):
        pass
    return None


def detect_layout():
    """The layout name of the running tilelang (``"legacy"``/``"modern"``).

    Order: capability probe (authoritative) -> version table (advisory).
    A disagreement (probe says one thing, version another) means a fork
    moved things again: the probe wins, and ``diagnostics()`` reports the
    mismatch so it surfaces instead of silently misrouting.
    """
    probed = _probe_layout()
    if probed is not None:
        return probed
    return _layout_for_version(tilelang_version_tuple())


def layout_target_modules(layout):
    """(ppu target module, legacy flat target module) implied by *layout*.

    Either entry may be None for layouts without that module; callers use
    the pair to pick import bases in one place instead of scattering
    try/except probes (see ``_aliases._make_cuda_target_alias``).
    """
    return _LAYOUT_TARGET_MODULES.get(layout, (None, None))


def diagnostics():
    """Human-readable summary of what the wrapper detected, for debugging.

    Usage: ``python -c "import tilelang_sail_cuda_wrapper as w;
    print(w._versions.diagnostics())"`` in any environment where the
    wrapper is installed -- useful when filing issues or checking whether
    activation picked the expected layout.
    """
    lines = ["tilelang-sail-cuda-wrapper diagnostics:"]
    version = tilelang_version()
    lines.append(f"  tilelang version : {version!r}"
                 + (f" (parsed {tilelang_version_tuple()})" if version else ""))
    probed = _probe_layout()
    from_table = _layout_for_version(tilelang_version_tuple())
    layout = detect_layout()
    lines.append(f"  layout (probed)  : {probed!r}")
    lines.append(f"  layout (version) : {from_table!r}")
    if probed is not None and from_table is not None and probed != from_table:
        lines.append("  NOTE: probe/version disagree -- fork layout drift, probe wins")
    lines.append(f"  layout (active)  : {layout!r}")
    ppu_target, legacy_target = layout_target_modules(layout or "")
    lines.append(f"  target modules   : ppu={ppu_target!r}, legacy={legacy_target!r}")
    return "\n".join(lines)
