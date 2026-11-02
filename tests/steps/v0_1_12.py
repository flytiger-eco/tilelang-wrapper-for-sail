"""Surfaces tilelang 0.1.12 introduced (the normalizer-registry target entry).

tilelang >= 0.1.12 moved target normalization into a registry
(``_TARGET_NORMALIZERS`` in ``tilelang.backend.target``) whose cuda and
cutedsl entries are registered by native ``tilelang/cuda/target.py`` at
import time.  Under the wrapper that module is never executed
(``tilelang.cuda.target`` is served as the alias of ``tilelang.ppu.target``),
so:

* (a) the cuda/cutedsl normalizers stay unregistered -- every
  ``determine_target`` input takes the plain validation path.  This is
  the wrapper's stable contract on this layout; the test pins it so a
  future tilelang registering from elsewhere (which would change cutedsl
  behavior) surfaces immediately.
* (b) a cutedsl dict input therefore fails loudly inside
  ``_validate_manual_target`` (tvm has no cutedsl target kind) instead of
  being normalized -- different from <= 0.1.11 where the inline
  normalization path survives (see v0_1_11.py).
* (c) the Target patch's cutedsl-keys exemption still holds: a cuda
  target already carrying the cutedsl key passes through un-rewritten
  (the exemption exists precisely because normalizer flows append the
  key after construction; it must keep such targets intact).

Note (c) shares code with <= 0.1.11 (the same wrapper branch), but its
consumer -- the normalizer flow that produces such targets -- is a
0.1.12+ concept, so the check lives here.
"""

from __future__ import annotations

GATE = "tilelang >= 0.1.12 (modern layout: normalizer registry)"


def applies(ctx):
    return ctx.version is not None and ctx.version >= (0, 1, 12)


def run(ctx):
    report = ctx.report
    import importlib.util

    if importlib.util.find_spec("tilelang.backend.target") is None:
        report("tilelang.backend.target absent on this tilelang: step inert")
        return
    import tilelang.backend.target as backend_target

    # (a) the registry exists, and the wrapper's hijack of the
    #     registration carrier keeps the cuda/cutedsl entries out of it.
    registry = getattr(backend_target, "_TARGET_NORMALIZERS", None)
    if registry is None:
        report("no _TARGET_NORMALIZERS on this tilelang: registry checks inert")
    else:
        assert "cuda" not in registry, (
            "cuda normalizer registered under the wrapper: native"
            " tilelang/cuda/target.py must not have executed (the alias"
            " hijacks it); a registration from elsewhere means the"
            " wrapper's target semantics drifted"
        )
        report("normalizer registry OK: cuda/cutedsl entries absent"
               " (registration carrier hijacked by the alias)")

    # (b) a cutedsl dict input fails loudly (the normalizer never
    #     registers, and tvm has no cutedsl target kind).
    try:
        backend_target.determine_target({"kind": "cutedsl", "arch": "sm_80"},
                                        return_object=True)
        report("FAIL: cutedsl dict unexpectedly accepted")
        import sys
        sys.exit(1)
    except AssertionError:
        report("cutedsl dict input fails loudly (normalizer unregistered)")

    # (c) the cutedsl-keys exemption: a cuda target already carrying the
    #     key passes through the rewrite un-rewritten.
    import tvm
    t = tvm.target.Target({"kind": "cuda", "keys": ["gpu", "cutedsl"],
                           "arch": "sm_80"})
    assert t.kind.name == "cuda" and "cutedsl" in t.keys, (
        f"cutedsl-keyed cuda target was rewritten to {t.kind.name};"
        " the exemption branch must leave it intact"
    )
    report("cutedsl-keys exemption OK: keyed cuda Target passes through"
           " un-rewritten")
