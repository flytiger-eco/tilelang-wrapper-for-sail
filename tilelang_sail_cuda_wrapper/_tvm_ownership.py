"""Ownership check: is the loaded ``tvm`` the one bundled with tilelang?

Observed behavior: while tilelang initializes it puts its vendored
``3rdparty/tvm/python`` tree at the front of ``sys.path``, so any
``import tvm`` afterwards resolves to that copy.  A standalone tvm
(``site-packages/tvm``) can therefore only be loaded when the user imports
it *before* tilelang -- and that one is not ours to patch.

The patches in ``_targets`` consult this module before touching anything.
"""

from __future__ import annotations

import os
import sys


def _tvm_belongs_to_tilelang() -> bool:
    """True if the loaded ``tvm`` is tilelang's vendored 3rdparty copy.

    Layouts recognized (checked against the loaded tvm's ``__file__``):
    - dev checkout:    ``<repo>/3rdparty/tvm/python`` next to ``<repo>/tilelang``
    - installed wheel: ``<site>/tilelang/3rdparty/tvm/python``

    Both layouts verified by inspecting ``tvm.__file__`` in each kind of
    install.
    """
    tvm_mod = sys.modules.get("tvm")
    tvm_target = sys.modules.get("tvm.target")
    origin = getattr(tvm_mod, "__file__", None) or getattr(tvm_target, "__file__", None)
    if not origin:
        return False
    real = os.path.realpath(origin)

    # Preferred signal: derive the vendored roots from tilelang's own location
    # (tilelang is in sys.modules by the time it imports tvm).
    tl_mod = sys.modules.get("tilelang")
    tl_file = getattr(tl_mod, "__file__", None)
    if tl_file:
        tl_dir = os.path.dirname(os.path.realpath(tl_file))
        for root in (tl_dir, os.path.dirname(tl_dir)):  # wheel / dev checkout
            vendored = os.path.join(root, "3rdparty", "tvm", "python")
            if real.startswith(vendored + os.sep):
                return True
        return False  # tilelang loaded, and its tvm is NOT this one: hands off

    # tilelang not imported yet (standalone tvm imported first): only the
    # vendored-layout signature can match; site-packages installs never have
    # a "3rdparty/tvm/python" path segment.
    parts = real.split(os.sep)
    return any(
        parts[i:i + 3] == ["3rdparty", "tvm", "python"]
        for i in range(len(parts) - 2)
    )
