"""End-to-end check: ``execution_backend="nvrtc"`` runs through PPU's hgrtc.

CUDA-style code asks for the NVRTC runtime-compilation backend; the wrapper
forwards the request to ``hgrtc`` (the PPU analogue).  This exercises the full
chain -- resolve (the wrapper's patch) -> hgrtc adapter -> hgcc runtime
compile -> run -> numerics -- not just the resolve layer.

Requires a running PPU environment whose tilelang ships the hgrtc spec
(tilelang dev / >= 0.1.16).  On 0.1.15 (no hgrtc spec) the compile raises the
native loud "Invalid execution backend" error -- see tests/steps/nvrtc_hgrtc.py
for the resolve-layer contract on that side.  Run:

    python3 e2e_nvrtc_hgrtc.py
"""

import tilelang
import tilelang.language as T
from tilelang.transform import PassConfigKey

# CUDA-style API names resolved through the wrapper alias
from tilelang.cuda.intrinsics import TensorCoreIntrinEmitter

assert TensorCoreIntrinEmitter.__name__ == "PPUTensorCoreIntrinEmitter"
print("[e2e-hgrtc] CUDA-style intrinsics name resolves to PPU impl:", TensorCoreIntrinEmitter.__name__)


@tilelang.jit(
    target="cuda",  # CUDA-style target, rewritten to ppu by the wrapper
    execution_backend="nvrtc",  # CUDA-style backend, forwarded to hgrtc
    pass_configs={PassConfigKey.TL_DISABLE_AIU_LOWER: False, PassConfigKey.TL_DISABLE_LDMAT_SWZL: False},
)
def matmul(A, B, block_M, block_N, block_K, num_stages, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * block_M, bx * block_N])
    return C


kernel = matmul.compile(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32, num_stages=3)
print("[e2e-hgrtc] compiled target:", kernel.target)
assert kernel.execution_backend == "hgrtc", (
    f"execution_backend='nvrtc' request resolved to {kernel.execution_backend!r},"
    " expected the forwarded 'hgrtc'"
)
print("[e2e-hgrtc] requested 'nvrtc', ran through:", kernel.execution_backend)

import torch

a = torch.randn(1024, 1024).cuda().half()
b = torch.randn(1024, 1024).cuda().half()
c = kernel(a, b)
ref_c = a @ b
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("[e2e-hgrtc] numerics validated against torch reference")
print("=== E2E PASS: execution_backend='nvrtc' ran through the PPU hgrtc backend ===")
