# Stage 5: Production API Packaging + Real Workload Integration

状态：**真机主体已验收**（package import / fresh-cwd / legacy API /
梯度一致性 / shim 同一性 / SPLROT train-eval / Workload A / B 全部
PASS；Stage 4 回归 17/17 PASS）。剩余 harness 修复（splr_ot_batch
数值化判据、Workload C / example cost conditioning）交付待最终复跑。
前置：Stage 3C FINAL FROZEN、Stage 4 COMPLETE / FROZEN。本阶段只做
packaging / API cleanup / integration / real workload validation——
未修改任何 solver 数学、preconditioner、cuDSS ownership、PCG、
forward 数值算法、Stage 4 autograd 语义，**native C++/CUDA 零改动，
setup.py 零改动，无 native rebuild**。

---

## 1. Package structure

```text
regot-cuda/
    setup.py                       # unchanged —— Stage 5 无需任何改动
    curegot/                       # 既有 Python package（Stage 5 之前已存在）
        __init__.py                # 既有；Stage 5 扩充：挂载公开 API + 顶级 re-export
        _cuda_loader.py            # 既有，未动
        _internal_numpy.<abi>.so   # 既有 native 扩展，未动
        _internal_torch.<abi>.so   # 既有 native 扩展，未动
        _backward_core.py          # Stage 5 新增：唯一 backward 数学实现（verbatim 迁入）
        backward.py                # Stage 5 新增：solve_backward / select_preconditioner
        autograd.py                # Stage 5 新增：splr_ot / splr_ot_with_info /
                                   #   SPLROT / splr_ot_batch / SPLROTFunction
    backward_core.py               # 根目录 compatibility shim（re-export）
    splr_backward.py               # 同上
    splr_autograd.py               # 同上
    validate_stage5_api.py         # 新增
    validate_stage5_workload.py    # 新增
    examples/
        differentiable_splr_ot.py  # 新增
```

关键事实（纠正常见误述）：repo 在 Stage 5 之前**已经**是 package
结构——`curegot/__init__.py` + `_cuda_loader.py` + 两个 native 扩展
`_internal_numpy.<abi>.so` / `_internal_torch.<abi>.so`。真正的 native
torch 模块是 **`curegot._internal_torch.torch`**。Stage 5 的 packaging
工作量因此只有：**新增三个纯 Python 文件**（`_backward_core.py` /
`backward.py` / `autograd.py`）+ **扩充既有 `__init__.py`**。
**setup.py unchanged；扩展名 unchanged（无改名、无移动）；
_internal_numpy unchanged；_internal_torch unchanged；Stage 5 无需任何
native rebuild**。

机制：扩充后的 `curegot/__init__.py` 通过既有 loader 拿到
`curegot._internal_torch.torch` pybind 子模块（pybind module 对象的
`__dict__` 可写），把 Python API **挂载**上去——只新增属性，不 shadow、
不修改任何 native symbol。`curegot/autograd.py` 从 `._internal_torch`
导入 native torch 子模块。

依赖方向（production 不依赖任何 repo 根目录脚本）：

```text
curegot._backward_core  ←  curegot.backward  ←  curegot.autograd  ←  public API
        ↑（经根目录 shim）stage3b_validate / benchmarks / Stage 4 脚本
```

## 2. Public API

```python
import curegot

T = curegot.torch.splr_ot(M, a, b, eta=0.05)   # 推荐入口
loss = downstream_loss(T)
loss.backward()
```

| 入口 | 说明 |
|---|---|
| `curegot.torch.splr_ot` | 可微 OT plan（主 API；顶级 `curegot.splr_ot` 亦可） |
| `curegot.torch.splr_ot_with_info` | debug/introspection，不可微，non-convergence 报 flag |
| `curegot.torch.SPLROT` | nn.Module config-holder wrapper |
| `curegot.torch.splr_ot_batch` | **顺序** Python batching（非 fused kernel） |
| `curegot.torch.solve_backward` | 低层 manual backward（研究用） |

向后兼容：`curegot.torch.sinkhorn_splr` / `sinkhorn_splr_bwd` /
`SPLRBackwardContext` 签名与行为完全不变；根目录 Stage 3/4 脚本经
shim 继续工作（对象同一性：`splr_autograd.splr_ot is
curegot.autograd.splr_ot`）。

## 3. SPLROT nn.Module

```python
layer = curegot.torch.SPLROT(eta=0.05, tol=1e-6, density=None,
                             backward_preconditioner="diag")
T = layer(M, a, b)
```

纯 config holder：**无** trainable parameter、**无** buffer；
`.train()/.eval()` 不改变数学行为（两次独立 GPU 迭代求解的数值等价
判据为 rel err ≤ 1e-12——不要求 bitwise，迭代求解器运行间调度噪声
属正常）；`.to(device)` 是安全 no-op（不偷偷管理 M/a/b）；
`state_dict()` 为空（属正常——从 repr/config 重新实例化即可）；
backward context 每次 forward 新建并属于该次 autograd graph，
**不存到 module 上**。repr 示例：

```text
SPLROT(eta=0.05, tol=1e-06, density=None, backward_preconditioner='diag')
```

## 4. Supported gradients

`M` 可微（dL/dM，implicit differentiation）；`a`/`b`/`eta` 为常量
（`a/b.requires_grad=True` → `NotImplementedError`）。标量参数在
public API 层做轻量校验（`eta>0`、`tol>0`、`max_iter>0`、
`0<density<=1`、`candidate_sinkhorn_iter>=0`、`backward_tol>0`、
`backward_max_iter>0`），非法值 → `ValueError`。

## 5. Device / dtype

CUDA float64 only，三输入同 device；CPU/float32 → `RuntimeError`；
不静默转换。Python 层不硬编码 `cuda:0`。

## 6. Single-problem limitation

一次调用解**一个** `[n, m]` OT 问题；`[B, n, m]` 输入被明确拒绝并
提示 `splr_ot_batch`（顺序循环 + `torch.stack`，非 fused native batch
kernel）。不为 batching 改 native solver。

## 7. Real workload results

真机（CUDA）已确认：Workload A（5.77e-08 → 3.54e-09 量级，loss 大幅
下降）与 Workload B（−2.54e-02 → −3.53e-01）PASS；package import /
fresh-cwd / legacy API / package-vs-manual 梯度一致（~1e-15）/ shim
同一性 / SPLROT train-eval（rel err ~1e-15）全部 PASS。Workload C 与
example 在原始（未归一化）cost scale 下个别迭代可能打满
max_iter=1000——这是 cost conditioning 问题，**不是** solver 缺陷；
修复方式是把 squared-distance cost 除以 embedding 维度 d（见下），
保持 max_iter=1000 / tol=1e-8 不变。

训练类 workload 的 forward 预算统一为 `WORKLOAD_MAX_ITER = 1000`
（= production forward 默认）：训练迭代中 cost matrix 会演化，开发期
的 200 上限可能被个别迭代打满——production 对不收敛正确 raise，所以
harness 必须给求解器与生产一致的预算。所有 workload 随机性均用显式
固定种子/Generator（含 nn.Linear 初始化），脚本每次运行走同一条优化
轨迹。

| workload | 内容 | 结果 |
|---|---|---|
| A | learnable M（SPLROT module），MSE→target plan，Adam×60 | 真机 PASS（loss 大幅下降） |
| B | embedding cost M=‖Xi−Yj‖²，affinity target，Adam×60 | 真机 PASS（−2.54e-02 → −3.53e-01） |
| C | tiny NN upstream → (cdist²/d) → OT → MSE，Adam×50 | cost conditioning 后待真机复跑；验收断言不变 |
| F | SGD smoke | loss 下降 |

核心结论：**gradient propagates THROUGH the OT layer**（X.grad /
net.parameters().grad 均 finite + nonzero）。每个训练 workload 打印
initial/final loss 与 final gradient norm。

## 8. Training example

```bash
python examples/differentiable_splr_ot.py
# 输出 initial loss / final loss / PASS
```

注意：对 `M` 加**常量**不改变 EROT plan（被 scaling 吸收）——构造
target 时必须用非常量扰动。

## 9. Performance sanity

只测 forward+backward step wall（100x120 / 300x360，5 次中位数 +
2 warmup），确认真实 workload 无异常 overhead；不做新 solver
optimization benchmark，不 sweep density。对照 Stage 4 benchmark 数字
即可。

## 10. Known limitations

- CUDA float64 only
- 一次调用一个 OT 问题（`splr_ot_batch` 为顺序 Python batching）
- 无 dL/da、dL/db、dL/deta
- 无 double backward（`@once_differentiable`；线性 loss 下 PyTorch 会
  静默返回不可微一阶梯度——上游文档化行为）
- AMP/autocast unsupported（显式 raise）
- `torch.compile` 不保证支持（custom autograd.Function + pybind 调用
  预期 graph break；沙盒审计路径已验证为非致命记录）
- TorchScript 不保证支持（审计：`NotSupportedError`，符合预期）

## 11. Future work

larger-scale workloads、native/fused batching、dtype 支持（float32）、
AMP、double backward、eta/a/b 梯度、torch.compile custom op、
distributed/multi-GPU 优化、论文实验接入。

## 12. 真机验收命令

```bash
# 0. 无需任何安装/构建改动：setup.py unchanged，native 扩展 unchanged，
#    Stage 5 只是向既有 curegot/ package 添加三个 .py 文件并扩充
#    __init__.py —— 不需要 native rebuild，不需要重装。

# 1. fresh-process import（packaging 最重要检查）
cd /tmp && python - <<'PY'
import curegot
print(curegot.__file__)
print(curegot.torch.splr_ot)
print(curegot.torch.SPLROT)
PY

# 2. Stage 5 验证
python validate_stage5_api.py        # STAGE 5 API VALIDATION: PASS
python validate_stage5_workload.py   # STAGE 5 WORKLOAD VALIDATION: PASS

# 3. example
python examples/differentiable_splr_ot.py

# 4. Stage 4 回归（Stage 5 未触碰 Stage 4 路径，按需）
python validate_stage4_autograd.py

# 5. 全过后跑一次完整 benchmark（quick benchmark 噪声 ±20%，不作优化依据）
python benchmark_stage4_autograd.py
```
