"""Version-split verification steps.

``verify_states.py`` keeps the checks every supported tilelang release
shares (the common contract: aliasing, target rewriting, strict mode,
pass routing, test gates, the version layer itself).  This package holds
the checks tied to a particular release or to a surface that moved
between releases: each module declares ``GATE`` (when it applies),
``applies(ctx)`` (probe-first; the parsed version is only a fast path)
and ``run(ctx)``.

The runner builds ``ctx`` -- a SimpleNamespace with ``tl`` (the tilelang
module), ``version`` (the parsed ``(major, minor, patch)`` tuple, or None
when unparseable) and ``report(msg)`` -- and calls ``run_all(ctx)`` after
the common steps.  ``run_all`` reports every module as active or inert,
so the log always shows which side of each split the running tilelang
exercised -- an inert module on an old release is the expected outcome,
not a gap.

Adding support for a new tilelang release: drop in a ``vX_Y_Z.py`` module
(``MIN_VERSION`` plus the new surfaces, each still probe-gated inside
``run`` so a fork missing one surface still verifies the others) and list
it in ``_STEP_MODULES``.

Naming convention: a module is named after the release that introduced
its surface, not after every release it applies to -- ``v0_1_12.py``
covers the normalizer-registry surface on both 0.1.12 and 0.1.15, and
``v0_1_11.py`` covers the legacy layout that only 0.1.11 still has
(gated by module presence, like facade_intrinsics, since fork version
strings do not always parse).  Each module's ``GATE`` states its actual
applicability.
"""

from __future__ import annotations

import importlib

# Ordered: split surfaces first (facade_intrinsics covers both sides of
# the 0.1.15 facade removal and thus always applies), then per-release
# modules in release order.
_STEP_MODULES = (
    "facade_intrinsics",
    "v0_1_11",
    "v0_1_12",
    "v0_1_15",
)


def run_all(ctx):
    """Run every applicable step module; report the inert ones."""
    for name in _STEP_MODULES:
        mod = importlib.import_module(f".{name}", package=__name__)
        if mod.applies(ctx):
            ctx.report(f"steps/{name}: active (gate: {mod.GATE})")
            mod.run(ctx)
        else:
            ctx.report(
                f"steps/{name}: inert on tilelang {ctx.version} (gate: {mod.GATE})"
            )
