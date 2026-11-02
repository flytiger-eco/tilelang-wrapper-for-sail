"""Surfaces exclusive to tilelang <= 0.1.11 (the legacy layout).

Three of them:

* (a) the ``tilelang.cuda.target`` alias bases itself on
  ``tilelang.utils.target`` (tilelang <= 0.1.11 ships no
  ``tilelang.ppu.target``; the helpers live flat in utils.target).
* (b) the legacy-only name map: ``_detect_torch_cuda_arch`` ->
  ``_detect_torch_ppu_arch`` and ``_cuda_target_from_arch`` ->
  ``_ppu_target_from_arch`` (the modern layout renamed both helpers, so
  these aliases exist only for the legacy base).
* (c) the inline cutedsl normalization: tilelang <= 0.1.11 calls
  ``normalize_cutedsl_target`` directly inside ``determine_target``
  (no registry), so the path stays live under the wrapper -- a cutedsl
  dict resolves to a ppu target carrying the cutedsl key (the cuda->ppu
  rewrite happens at the intermediate Target construction, then the
  normalizer re-exports with the key appended).  When the optional
  CuTeDSL package is absent the input still fails loudly
  (AssertionError), which is the equally acceptable outcome; the test
  accepts either but never a silent kind=cuda result.
"""

from __future__ import annotations

import importlib.util

GATE = "tilelang <= 0.1.11 (legacy layout: no tilelang.ppu.target)"


def applies(ctx):
    # The authoritative legacy marker: the module tilelang >= 0.1.12
    # introduced does not exist.
    try:
        return importlib.util.find_spec("tilelang.ppu.target") is None
    except (ImportError, AttributeError, ValueError):
        return False


def run(ctx):
    report = ctx.report

    # (a) the alias bases itself on the flat legacy target module.
    import tilelang.cuda.target as cuda_target
    import tilelang.utils.target as utils_target
    assert cuda_target._tilelang_sail_cuda_wrapper_alias_of__ is utils_target, \
        "legacy tilelang.cuda.target alias is not based on tilelang.utils.target"
    report("alias base OK: tilelang.cuda.target -> tilelang.utils.target")

    # (b) the legacy-only name map.
    from tilelang.cuda.target import (
        _cuda_target_from_arch,
        _detect_torch_cuda_arch,
    )
    assert _detect_torch_cuda_arch is utils_target._detect_torch_ppu_arch, \
        "_detect_torch_cuda_arch does not forward to _detect_torch_ppu_arch"
    assert _cuda_target_from_arch is utils_target._ppu_target_from_arch, \
        "_cuda_target_from_arch does not forward to _ppu_target_from_arch"
    report("legacy-only target aliases OK: _detect_torch_cuda_arch /"
           " _cuda_target_from_arch forward to their ppu counterparts")

    # (c) the inline cutedsl normalization stays live (no registry on
    #     this layout, so the wrapper's hijack of the registry carrier
    #     cannot remove it).
    from tilelang.utils.target import determine_target
    try:
        result = determine_target({"kind": "cutedsl", "arch": "sm_80"},
                                   return_object=True)
    except AssertionError:
        # CuTeDSL package absent on this environment: loud failure is
        # the acceptable outcome.
        report("cutedsl inline normalization: loud AssertionError"
               " (CuTeDSL package absent) -- acceptable")
        return
    assert result.kind.name == "ppu" and "cutedsl" in result.keys, (
        f"cutedsl dict resolved to {result.kind.name}/{list(result.keys)},"
        " expected a ppu target carrying the cutedsl key"
    )
    report("cutedsl inline normalization OK: dict -> ppu target with"
           " cutedsl key (cuda->ppu rewrite then key append)")
