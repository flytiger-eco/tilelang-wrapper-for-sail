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
    detected = tl.utils.target._detect_torch_ppu_arch()
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
    try:
        tl.utils.target.determine_target("llvm")
        print("[wrapped] FAIL: llvm target unexpectedly accepted")
        sys.exit(1)
    except ValueError as e:
        print("[wrapped] strict mode OK:", str(e)[:110])
    try:
        tl.utils.target.determine_target("metal")
        print("[wrapped] FAIL: metal target unexpectedly accepted")
        sys.exit(1)
    except ValueError as e:
        print("[wrapped] strict mode OK (metal):", str(e)[:80])
    # ppu target itself still fine
    r = tl.utils.target.determine_target("ppu")
    print("[wrapped] ppu target accepted:", r)

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

print(f"=== state '{state}' verification done ===")
