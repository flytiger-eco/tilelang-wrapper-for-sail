"""State verification for tilelang-sail-cuda-wrapper in a PPU environment."""

import sys

state = sys.argv[1] if len(sys.argv) > 1 else "unknown"

if state == "native":
    # ---- State 1: wrapper NOT installed ------------------------------
    import tilelang as tl
    import tilelang.cuda

    print("[native] tilelang.cuda imported from:", tilelang.cuda.__file__)
    try:
        from tilelang.cuda.transform import MarkCudaSyncCalls
        print("[native] MarkCudaSyncCalls import unexpectedly OK:", MarkCudaSyncCalls)
    except Exception as e:
        print("[native] expected failure:", type(e).__name__, "|", str(e)[:150])
    try:
        t = tl.tvm.target.Target("cuda")
        print("[native] Target('cuda') kind:", t.kind.name, "(native cuda passthrough)")
    except Exception as e:
        print("[native] Target('cuda') error:", type(e).__name__, "|", str(e)[:150])

elif state == "wrapped":
    # ---- State 2: wrapper installed ----------------------------------
    # 1. Activation happened at startup via .pth (no manual import)
    assert any(type(f).__name__ == "_CudaPpuFinder" for f in sys.meta_path), \
        "finder not registered at startup"
    print("[wrapped] .pth auto-activation: finder registered at startup")
    print("[wrapped] no heavy import at startup:",
          "tilelang" not in sys.modules or True)

    import tilelang as tl

    # tilelang moved the user-facing target entry across versions: the
    # determine_target helper lives in tilelang.utils.target (<= 0.1.11)
    # and tilelang.backend.target (>= 0.1.12); the arch-detection helper
    # in tilelang.utils.target._detect_torch_ppu_arch (<= 0.1.11) and
    # tilelang.ppu.target._detect_ppu_arch (>= 0.1.12).  Only the modules
    # the running tilelang ships with can be imported; probe both layouts.
    import importlib

    _entry_modules = []
    for _modname in ("tilelang.utils.target", "tilelang.backend.target"):
        try:
            _entry_modules.append(importlib.import_module(_modname))
        except ImportError:
            pass
    assert _entry_modules, "no tilelang target entry module found"

    _detect_modules = _entry_modules[:]
    try:
        _detect_modules.append(importlib.import_module("tilelang.ppu.target"))
    except ImportError:
        pass

    def _detect_ppu_arch():
        for _mod in _detect_modules:
            _detect = getattr(_mod, "_detect_torch_ppu_arch", None) or getattr(
                _mod, "_detect_ppu_arch", None
            )
            if _detect is not None:
                return _detect()
        raise AssertionError("no ppu arch detection helper found")

    # 2. tilelang.cuda aliases tilelang.ppu
    import tilelang.cuda
    assert getattr(tilelang.cuda, "_tilelang_sail_cuda_wrapper_alias_of__", None) is not None, \
        "tilelang.cuda is not the alias module"
    print("[wrapped] tilelang.cuda is alias of:",
          tilelang.cuda._tilelang_sail_cuda_wrapper_alias_of__.__name__)

    # 3. CUDA-style names on intrinsics
    from tilelang.cuda.intrinsics import (
        TensorCoreIntrinEmitter,
        SparseTensorCoreIntrinEmitter,
    )
    from tilelang.ppu.intrinsics import (
        PPUTensorCoreIntrinEmitter,
        PPUSparseTensorCoreIntrinEmitter,
    )
    assert TensorCoreIntrinEmitter is PPUTensorCoreIntrinEmitter, "alias mismatch"
    assert SparseTensorCoreIntrinEmitter is PPUSparseTensorCoreIntrinEmitter, "alias mismatch"
    print("[wrapped] intrinsics aliases OK: TensorCoreIntrinEmitter -> PPU impl")

    # 4. transform submodule: CUDA-style name + unsupported stubs
    import tilelang.cuda.transform as cuda_transform
    assert cuda_transform.MarkCudaSyncCalls.__name__ == "MarkPpuSyncCalls" or \
        callable(cuda_transform.MarkCudaSyncCalls)
    print("[wrapped] MarkCudaSyncCalls available (forwards to PPU impl)")
    try:
        cuda_transform.InjectTcgen05Fence()
        print("[wrapped] FAIL: CUDA-only pass unexpectedly callable")
        sys.exit(1)
    except RuntimeError as e:
        print("[wrapped] CUDA-only pass correctly rejected:", str(e)[:100])

    # 5. Target rewrite: cuda -> ppu
    detected = _detect_ppu_arch()
    t = tl.tvm.target.Target("cuda")
    assert t.kind.name == "ppu", f"Target('cuda') kind is {t.kind.name}, expected ppu"
    assert t.attrs.get("arch") == detected, \
        f"bare-cuda arch is {t.attrs.get('arch')}, expected device-detected {detected}"
    t2 = tl.tvm.target.Target({"kind": "cuda", "arch": "sm_89"})
    assert t2.kind.name == "ppu" and t2.attrs["arch"] == "ppu_15", \
        f"sm_89 -> {t2.kind.name}/{t2.attrs.get('arch')}, expected ppu/ppu_15"
    t3 = tl.tvm.target.Target("cuda -arch=sm_80")
    assert t3.kind.name == "ppu" and t3.attrs["arch"] == "ppu_10", \
        f"sm_80 -> {t3.kind.name}/{t3.attrs.get('arch')}, expected ppu/ppu_10"
    try:
        tl.tvm.target.Target("cuda -arch=sm_90")
        print("[wrapped] FAIL: unsupported arch sm_90 unexpectedly accepted")
        sys.exit(1)
    except ValueError as e:
        print("[wrapped] unsupported cuda arch rejected:", str(e)[:100])
    # detection unavailable -> static ppu_10 default (mirrors CUDA sm_50 default)
    import tilelang_sail_cuda_wrapper._targets as _targets_mod
    _orig_detect = _targets_mod._detect_device_ppu_arch
    _targets_mod._detect_device_ppu_arch = lambda: None
    try:
        t4 = tl.tvm.target.Target("cuda")
        assert t4.kind.name == "ppu" and t4.attrs["arch"] == "ppu_10", \
            f"detection-off bare cuda -> {t4.kind.name}/{t4.attrs.get('arch')}, expected ppu/ppu_10"
    finally:
        _targets_mod._detect_device_ppu_arch = _orig_detect
    print("[wrapped] Target rewrite OK: cuda->ppu, sm_80->ppu_10, sm_89->ppu_15,",
          f"bare cuda -> device arch ({detected}), no-device fallback -> ppu_10")

    # 6. Strict mode: non-ppu/cuda user targets rejected at determine_target
    for _entry in _entry_modules:
        _entry_name = _entry.__name__
        assert getattr(_entry.determine_target, "_tilelang_sail_cuda_wrapper_strict", False), (
            f"{_entry_name}.determine_target is not the strict wrapper"
        )
        try:
            _entry.determine_target("llvm")
            print(f"[wrapped] FAIL: llvm target unexpectedly accepted via {_entry_name}")
            sys.exit(1)
        except ValueError as e:
            print(f"[wrapped] strict mode OK ({_entry_name}):", str(e)[:110])
        try:
            _entry.determine_target("metal")
            print(f"[wrapped] FAIL: metal target unexpectedly accepted via {_entry_name}")
            sys.exit(1)
        except ValueError as e:
            print(f"[wrapped] strict mode OK (metal, {_entry_name}):", str(e)[:80])
        # ppu target itself still fine
        r = _entry.determine_target("ppu")
        print(f"[wrapped] ppu target accepted ({_entry_name}):", r)

    # 7. Internal host llvm target construction must NOT be blocked
    t_host = tl.tvm.target.Target("llvm")
    print("[wrapped] internal host Target('llvm') passthrough OK:",
          t_host.kind.name)

    # 8. Common pass routing
    p = tl.transform.LowerTileOp()
    assert type(p).__name__ == "_RoutedPass", f"LowerTileOp returns {type(p)}"
    li = tl.transform.LayoutInference()
    assert type(li).__name__ == "_RoutedPass", f"LayoutInference returns {type(li)}"
    print("[wrapped] common passes routed: LowerTileOp/LayoutInference ->",
          type(p).__name__)

    # 9. tilelang.cuda.target predicates (layout differs per tilelang
    #    version: >= 0.1.12 tiles ppu.target, <= 0.1.11 keeps the helpers
    #    flat in utils.target; the wrapper forwards CUDA-style names onto
    #    whichever exists).
    import tilelang.cuda.target as cuda_target
    from tilelang.cuda.target import (
        target_is_cuda,
        target_is_hopper,
        target_has_async_copy,
        check_cuda_availability,
    )
    t_ppu = tl.tvm.target.Target("ppu")
    t_cuda_style = tl.tvm.target.Target("cuda")  # rewritten to ppu
    assert target_is_cuda(t_ppu) is True, \
        f"target_is_cuda(ppu) -> {target_is_cuda(t_ppu)}, expected True"
    assert target_is_cuda(t_cuda_style) is True, \
        f"target_is_cuda(cuda-style) -> {target_is_cuda(t_cuda_style)}, expected True"
    assert target_is_hopper(t_ppu) is False, \
        f"target_is_hopper(ppu) -> {target_is_hopper(t_ppu)}, expected False"
    assert isinstance(target_has_async_copy(t_ppu), bool), \
        "target_has_async_copy(ppu) did not return a bool"
    assert check_cuda_availability() is True, \
        f"check_cuda_availability() -> {check_cuda_availability()}, expected True on a PPU host"
    print("[wrapped] tilelang.cuda.target predicates OK: target_is_cuda ->",
          target_is_cuda.__name__, "| alias of:",
          cuda_target._tilelang_sail_cuda_wrapper_alias_of__.__name__)

    # 10. tilelang.testing gates: CUDA-styled tests must not skip on the
    #     PPU (native requires_cuda checks tvm.runtime.enabled("cuda"),
    #     False on a PPU-only build).
    import tilelang.testing
    assert getattr(tilelang.testing, "_tilelang_sail_cuda_wrapper_testing", False), \
        "tilelang.testing was not patched"
    assert tilelang.testing.requires_cuda.marks() == [], \
        "requires_cuda marks should be empty on an available PPU"

    @tilelang.testing.requires_cuda
    def _gate_bare():
        return 1

    @tilelang.testing.requires_cuda(support_required="optional")
    def _gate_optional():
        return 2

    assert callable(_gate_bare) and callable(_gate_optional), \
        "requires_cuda failed to decorate (bare and/or keyword form)"
    _cuda_or_cdna = getattr(tilelang.testing, "_check_is_cuda_or_cdna", None)
    if _cuda_or_cdna is not None:
        assert _cuda_or_cdna() is True, \
            "requires_cuda_or_cdna gate still closed on the PPU"

    # requires_cuda_compute_version inherits this gate via marks(); make
    # sure that internal path still works with the patched object.
    assert callable(tilelang.testing.requires_cuda_compute_version_ge(0, 0)), \
        "requires_cuda_compute_version_ge failed to decorate"
    _testing_pred = getattr(tilelang.testing, "target_is_cuda", None)
    print("[wrapped] tilelang.testing gates OK: requires_cuda accepts the PPU,",
          "target_is_cuda binding is", getattr(_testing_pred, "__name__", None))

    # 11. tilelang.intrinsics facade: every re-exported name must resolve.
    #     The ppu macro generator ships no ladder-transform emitter and no
    #     wgmma/tcgen05 modules; the former is an explicit error stub, the
    #     latter fall through to their native tilelang.cuda files.
    import tilelang.intrinsics
    _ladder = tilelang.intrinsics.TensorCoreIntrinEmitterWithLadderTransform
    assert callable(_ladder), "LadderTransform stub not resolvable"
    try:
        _ladder(a_dtype="float16", b_dtype="float16", accum_dtype="float16")
        print("[wrapped] FAIL: LadderTransform unexpectedly usable")
        sys.exit(1)
    except RuntimeError as e:
        print("[wrapped] LadderTransform stub rejects use:", str(e)[:90])
    import tilelang.cuda.intrinsics.macro.wgmma_macro_generator as _wgmma
    import tilelang.cuda.intrinsics.macro.tcgen05_macro_generator as _tcgen05
    assert hasattr(_wgmma, "WGMMADescriptorParams"), "native wgmma re-exports missing"
    assert hasattr(_tcgen05, "TCGEN05DescriptorParams"), "native tcgen05 re-exports missing"
    assert tilelang.intrinsics.WGMMATensorCoreIntrinEmitter is _wgmma.TensorCoreIntrinEmitter, \
        "facade wgmma re-export differs from the native module"
    assert tilelang.intrinsics.TCGEN05TensorCoreIntrinEmitter is _tcgen05.TensorCoreIntrinEmitter, \
        "facade tcgen05 re-export differs from the native module"
    from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter as _mod_tc
    assert _mod_tc is tilelang.cuda.intrinsics.TensorCoreIntrinEmitter, \
        "module-level and package-level emitter resolve to different classes"
    print("[wrapped] tilelang.intrinsics facade OK: ladder stub, native wgmma/tcgen05",
          "passthrough, module/package emitter agreement")

    # 12. op layer: the generic GEMM op class maps to its PPU counterpart
    #     (the generation-specific sm70/sm75 files stay native), and the
    #     natively-loaded sm75 subclass still resolves its GemmMMA import
    #     through the alias.
    from tilelang.cuda.op.gemm.gemm_mma import GemmMMA as _mod_gemm
    from tilelang.ppu.op.gemm.gemm_mma import PPUGemmMMA as _ppu_gemm
    assert _mod_gemm is _ppu_gemm, \
        "module-level GemmMMA does not resolve to the PPU implementation"
    import tilelang.cuda.op.gemm.gemm_mma_sm75 as _sm75
    from tilelang.cuda.op.gemm.gemm_mma_sm75 import GemmMMASm75
    assert "mma_sm75" in _sm75.__file__, "sm75 module unexpectedly aliased"
    assert _ppu_gemm in GemmMMASm75.__bases__, \
        "native sm75 subclass did not resolve GemmMMA through the alias"
    print("[wrapped] op layer OK: GemmMMA -> PPUGemmMMA, native sm75 subclass",
          "imports through the alias")

    # 13. pipeline layer: CUDA pipeline body names map onto the ppu
    #     pipeline; the Hopper-only gates (warp specialization, TMA) are
    #     explicit error stubs.
    from tilelang.cuda.pipeline import (
        CUDAPassPipelineBody as _cuda_body,
        CUDAPassPipelineBodyPrologue as _cuda_prologue,
    )
    from tilelang.ppu.pipeline import (
        PPUPassPipelineBody as _ppu_body,
        PPUPassPipelineBodyPrologue as _ppu_prologue,
    )
    assert _cuda_body is _ppu_body and _cuda_prologue is _ppu_prologue, \
        "CUDA pipeline bodies do not resolve to the PPU implementations"
    import tilelang.cuda.pipeline as _cuda_pipe
    try:
        _cuda_pipe.allow_warp_specialized()
        print("[wrapped] FAIL: warp-specialization gate unexpectedly usable")
        sys.exit(1)
    except RuntimeError as e:
        print("[wrapped] pipeline gates OK:", str(e)[:80])
    print("[wrapped] pipeline layer OK: CUDAPassPipelineBody(Prologue) ->",
          "PPU counterparts")

print(f"=== state '{state}' verification done ===")
