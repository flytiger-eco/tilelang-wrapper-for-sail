# tilelang-sail-cuda-wrapper

使 CUDA 风格的 [TileLang](https://github.com/tile-ai/tilelang) 代码**无需修改一行**即可运行于 tilelang-sail（TileLang 的 PPU 后端版本）。

- **零改动迁移**：`import tilelang.cuda`、`from tilelang.cuda.intrinsics import ...`、`target="cuda"`、`tl.transform.LowerTileOp()` 等存量写法原样保留
- **纯运行时方案**：pip 安装即生效、卸载即还原；不修改 tilelang 源码树，也不重新打包 tilelang
- **零依赖**：只用 Python 标准库（import 协议 + site 机制），不引入任何运行时依赖
- **边界明确**：未找到 PPU 对应实现的 CUDA 专有特性显式报错，绝不静默降级

> **非官方第三方项目**：本仓库与 TileLang、tilelang-sail 及其开发团队没有任何关联，也未获得其认可或支持。本包不包含、不修改二者的任何代码——所有兼容逻辑在运行时由独立安装的 wrapper 提供。tilelang-sail 自身的问题请反馈至其官方渠道。

## 背景

我们现有一批按 CUDA 风格编写的 TileLang 内核，迁移至 tilelang-sail（TileLang 的 PPU 后端版本）运行时，依次遇到三类不兼容问题：

1. **import**：`from tilelang.cuda.intrinsics import TensorCoreIntrinEmitter` 即报错——`tilelang.cuda` 模块本身可导入，但调用其中接口时触发原生错误。检查 `tilelang.ppu` 后发现同类能力均已提供，仅命名不同（`TensorCoreIntrinEmitter` → `PPUTensorCoreIntrinEmitter`）；
2. **target**：逐一修正 import 后，`target="cuda"` 编译再次失败——PPU backend 不接受该 target；
3. **pass**：将 target 改为 `"ppu"` 后，编译仍于 lowering 阶段失败——`LowerTileOp` / `LayoutInference` 等通用 pass 在 PPU target 上需改走 `tilelang.ppu.transform` 的专用实现。


## 安装 / 卸载

前置条件：已安装 tilelang-sail 的 PPU-only 构建（本 wrapper 的行为映射基于 tilelang-sail `<版本号>` 实测，其他版本未验证）。

```bash
pip install .                               # 在本仓库根目录执行（无任何运行时依赖，不会重装 tilelang）
pip uninstall tilelang-sail-cuda-wrapper    # 卸载，完全还原原生行为
```

安装/卸载后需**重启 Python 进程**方可生效（`.pth` 仅在解释器启动时注册拦截器；对已运行的进程无效）。

## 快速上手

```python
import tilelang
import tilelang.language as T

# CUDA 风格 intrinsics —— wrapper 解析到 PPU 实现（PPUTensorCoreIntrinEmitter）
from tilelang.cuda.intrinsics import TensorCoreIntrinEmitter

@tilelang.jit(target="cuda")    # 运行时重写为 PPU target，arch 按设备探测
def matmul(A, B, block_M, block_N, block_K, num_stages, dtype=T.float16, accum_dtype=T.float32):
    ...                          # 内核体与 CUDA 版完全相同

kernel = matmul.compile(M=1024, N=1024, K=1024,
                        block_M=128, block_N=128, block_K=32, num_stages=3)
print(kernel.target)            # PPU target（arch=ppu_10 / ppu_15，按设备）
```

完整可运行示例（含 torch 数值校验）见 `tests/e2e_cuda_style.py`。

## 两种安装状态的行为

| | 未安装 wrapper（原生） | 安装 wrapper 后 |
|---|---|---|
| `import tilelang.cuda` | 原生包可导入，但调用其接口时触发原生错误 | 别名到 `tilelang.ppu`，CUDA 风格名可用 |
| `from tilelang.cuda.intrinsics import TensorCoreIntrinEmitter` | ImportError / 原生实现 | 转发到 `PPUTensorCoreIntrinEmitter` |
| `tilelang.cuda.transform.MarkCudaSyncCalls` | 原生实现 | 转发到 `MarkPpuSyncCalls` |
| CUDA-only pass（`LowerBlackwell2SM`、`InjectTcgen05Fence` 等） | 原生实现 | **显式报错**（无 PPU 对应实现） |
| `target="cuda"`（jit/compile/autotune） | 编译报错 | 重写为 ppu；arch 未指定时由**设备探测**决定（实测与 determine_target 解析 ppu target 的结果一致，ppu_10 / ppu_15 按硬件代次），探测不可用则默认 **ppu_10**——对齐 TVM 对 CUDA 无设备时默认 sm_50 的做法；显式 arch 仅验证了 sm_80→ppu_10、sm_89→ppu_15，其余 **ValueError**（不做猜测映射） |
| 其他 target（llvm/metal/rocm…，经 `determine_target`） | 原生行为 | **直接报错**（严格模式） |
| `Target("llvm")`（引擎内部 host codegen） | 原生行为 | 放行（引擎内部必需，不拦截） |
| `tl.transform.LowerTileOp()(mod)` 等 common pass | 通用实现 | mod 内含 ppu target → `tl.ppu.transform` 专用 pass；否则回退原生 |
| 独立安装的 tvm（先于 tilelang 被 import） | 原生行为 | **不做任何改动**（仅 patch tilelang 自带的 vendored tvm） |

## 工作原理

- **`sys.meta_path` finder（PEP 302/451）**：被动应答解释器的 import 请求，无 watcher、无线程、无轮询。插在 meta_path 最前端，先于原生 `tilelang/cuda` 目录命中。
- **`.pth` 激活**：pip 安装后由 `site` 机制在每次解释器启动时执行 `import tilelang_sail_cuda_wrapper`，注册 finder。启动开销 O(1)，不加载 tilelang/tvm。
- **运行时别名（PEP 562）**：`tilelang.cuda.*` 模块通过模块级 `__getattr__` **懒转发**到 `tilelang.ppu.*`，并在属性访问时做 CUDA 风格名映射（`TensorCoreIntrinEmitter → PPUTensorCoreIntrinEmitter` 等）。子模块属性同样转发为对应的 cuda 别名，不会泄漏原始 ppu 模块。
  - 为什么不用 `__dict__` 快照拷贝：`tilelang.ppu` 的加载链会反向 import `tilelang.cuda.*`（`intrinsics.macro` 下有模块引用 cuda 风格 layout），finder 会在 ppu 模块**尚未完成初始化**时被重入，快照会将该不完整状态固化于 `sys.modules`。懒转发不受影响——模块在原地填充，后续访问始终得到完整状态。
- **加载后 patch**：`tvm.target` / `tilelang.transform` / `tilelang.utils.target` 的真实 loader 被包装，模块加载完成后立即打 patch（Target 重写、pass 路由、严格模式）。
- **tvm 归属判定**：`tvm.target` 的 patch 只作用于 **tilelang 自带的 vendored tvm**（`3rdparty/tvm/python`）。判定依据两级信号：优先从 `sys.modules["tilelang"].__file__` 推导 vendored 根（dev checkout 与 wheel 两种布局）比对 tvm 文件路径；tilelang 尚未加载时（此时只可能是用户先 import 了独立 tvm）退回路径签名（含 `3rdparty/tvm/python` 段）。独立安装的 tvm（如 `site-packages/tvm`）一律不做改动，其 `Target("cuda")` 原样透传。
- **严格模式卡点在 `determine_target`**（实测 jit/compile/autotune 传入的 target 均汇聚于该入口），而非 `Target.__init__`：编译过程中引擎会自行构造 `Target("llvm")` 这类 host codegen target，必须放行。

## 仓库结构

```
tilelang_sail_cuda_wrapper/
├── __init__.py        # 激活入口：.pth 触发 install()（不加载 tilelang/tvm）
├── _finder.py         # import 拦截骨架：meta_path finder、两个 loader、install()
├── _aliases.py        # tilelang.cuda.* 别名模块合成、CUDA→PPU 名字映射、懒转发
├── _targets.py        # Target 重写（arch 映射 / 设备探测）与严格模式校验
├── _passes.py         # common pass → PPU 专用 pass 的路由
└── _tvm_ownership.py  # tvm 归属判定：只 patch tilelang 自带的 vendored tvm
```

实现按职责拆成 5 个模块，各司其职：`_finder.py` 只保留 import 协议的骨架（finder / loader / `install()`），别名合成、target 语义、pass 路由、归属判定分别在对应模块。验证脚本在 `tests/` 下，不随包分发。

## 验证

需在配有 PPU 设备的 tilelang-sail 环境中运行：

```bash
python3 tests/verify_states.py native    # 未安装 wrapper 的原生行为基线
python3 tests/verify_states.py wrapped   # 安装 wrapper 后的行为断言
python3 tests/e2e_cuda_style.py          # 端到端：CUDA 风格 matmul 编译 + 数值验证
```

覆盖点：

- **双状态对照**：alias 转发、CUDA-only pass 显式报错、target 重写、严格模式、pass 路由、tvm 归属判定；
- **arch 规则**：设备探测（ppu_10 / ppu_15 按硬件代次）、sm_80 / sm_89 显式映射、其余 arch ValueError、无设备时回退至 ppu_10；
- **端到端**：`target="cuda"` 的 matmul 经 PPU backend 编译（kernel target 为 ppu），数值与 torch 参考一致。

## 已知限制

- **严格模式**：本 wrapper 仅覆盖“CUDA 风格代码 → PPU”这一场景，经 `determine_target` 的其他 target（llvm/metal/rocm…）明确报错——无法覆盖的场景立即说明，而非静默产生错误行为；编译过程中引擎内部构造的 host codegen target（如 `Target("llvm")`）放行，不受影响。
- **CUDA-only 特性显式报错**：10 个未找到 PPU 对应实现的 pass（Blackwell / Hopper 专有等）会抛出明确的 RuntimeError，而非静默降级或难以排查的 AttributeError。
- **arch 映射仅覆盖两个档位**：sm_80→ppu_10、sm_89→ppu_15，其余显式 arch 一律 ValueError。
- **安装/卸载后需重启 Python 进程**：`.pth` 仅在解释器启动时执行。
- tilelang-sail 升级后建议重新运行 `tests/` 下的验证脚本——wrapper 不修改其源码，但其内部行为可能随版本变化；如发现行为漂移或新的不兼容写法，欢迎提交 issue。
