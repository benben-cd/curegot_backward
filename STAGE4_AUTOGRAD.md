# Stage 4: End-to-End PyTorch Autograd Integration

状态：**交付，待真机验收**。本档是 Stage 4 的用户/API 文档与验证摘要。
前置：Stage 3C FINAL FROZEN（implicit-diff 数学、native diagonal
preconditioner、selector policy 均未改动）。

---

## 1. Public API

```python
from splr_autograd import splr_ot

T = splr_ot(M, a, b, eta=0.05)     # M: [n,m], a: [n], b: [m]，CUDA float64
loss = downstream_loss(T)
loss.backward()                    # M.grad = dL/dM（implicit gradient）
```

`splr_ot` 表现得像一个普通 PyTorch op：返回 plan 张量 `T`（不返回 dict、
不暴露 context），`M.requires_grad=True` 时自动挂 `grad_fn`。

签名（keyword-only 参数）：

```python
splr_ot(M, a, b, *, eta,
        tol=1e-6, max_iter=1000,            # forward solver（native 默认）
        density=None,                       # None -> native: min(10/min(n,m), 1)
        candidate_sinkhorn_iter=3,          # native 默认
        backward_preconditioner="diag",     # "diag" | "splr" | "none"
        backward_tol=1e-12,                 # backward PCG 相对容差
        backward_max_iter=2000)
```

命名约定：`tol/max_iter` 属于 forward solver，`backward_*` 属于
implicit-gradient PCG，两者刻意区分。

辅助函数 `splr_ot_with_info(M, a, b, *, eta, ...)` 返回
`{plan, dual, niter, converged}`，**不可微**（no_grad + detached），
non-convergence 通过 flag 报告而不 raise，供 debug/introspection 使用。

## 2. 支持与不支持的梯度

| 输入 | 梯度 | 行为 |
|---|---|---|
| `M` | dL/dM | 支持（Stage 3A/3B oracle 验证的 implicit-diff 路径） |
| `a`, `b` | — | `requires_grad=True` → `NotImplementedError`（不静默算错） |
| `eta` | — | 常量标量（float 转换后传入） |

第一版刻意保守：不推导 dL/da、dL/db、dL/deta。

## 3. dtype / device contract（严格，不静默转换）

- `M/a/b` 必须是 **同一 device 上的 CUDA float64** 张量。
- CPU 输入、float32 输入 → `RuntimeError`；shape 不符 → `ValueError`。
- 不做 `float32 in → float64 internal → float32 out` 的静默转换：那会让
  返回梯度的 dtype/device 对用户产生误导。
- **AMP/autocast**：`torch.is_autocast_enabled()` 为真 → `RuntimeError`
  （solver 是 float64-only）。
- multi-GPU：Python 层不硬编码 `cuda:0`；device 完全由输入张量决定，
  native 侧 `CUDAGuard` 跟随输入 device（Stage 3 源码 audit 确认）。

## 4. 默认策略

- backward preconditioner 默认 `"diag"`（native diagonal，Stage 3C-4
  冻结的 production policy），经 `ctx.apply_diagonal_preconditioner`，
  与 manual path（`sinkhorn_splr_bwd` + `solve_backward`）**逐位一致**。
- `"splr"` 显式 opt-in（cuDSS + rank-2 Woodbury）；`"none"` 仅 debug。
- 非法值 → `ValueError`；不存在 `"python_diag"` production 模式。

## 5. Non-convergence 行为（production-safe）

- **forward 不收敛** → `RuntimeError`（含 niter/max_iter/tol 信息）。
- **backward PCG 不收敛** → `RuntimeError`（含 iterations/max_iter/tol/
  preconditioner 信息）。
- 设计理由：训练循环里 silent bad gradient 的风险远大于一次显式失败。
- 底层 manual API（`curegot.torch.sinkhorn_splr_bwd` +
  `splr_backward.solve_backward`）保留 flag 语义，供研究用途。

## 6. Double backward 状态

**不支持**。`backward` 标记 `@once_differentiable`（torch 2.8 语义已
确认）：第一次 `grad(..., create_graph=True)` 返回接在 `DelayedError`
节点上的别名，**第二次**微分时 raise
`RuntimeError: trying to differentiate twice a function that was marked
with @once_differentiable`。

已知上游限制：对**线性** loss（grad_output 为常量），PyTorch 的
`once_differentiable` 无法注入 DelayedError，一阶梯度会被静默地以
不可微张量返回（此后对其 backward 报 "does not require grad"）。这是
PyTorch 的文档化行为，不是本模块的数值错误。

## 7. autograd 语义细节（已验证）

- `M.requires_grad=False` → 返回的 `T` 无 `grad_fn`。
- 非 contiguous `grad_output` → 内部 contiguous 化（已是则零开销）。
- `retain_graph=True` 重复 backward → 梯度正确累加。
- 多分支 loss（G = G1 + G2）→ autograd 引擎聚合后一次 PCG。
- 多 context 独立、跨图逆序 backward → 正确（pybind context 由
  autograd ctx 属性的 Python refcount 保活，无全局 singleton/static
  state 依赖）。
- forward 局部变量释放后再 backward → 正确（`ctx.backward_context`
  属性持引用；`save_for_backward` 只存 `T` 张量）。
- in-place 安全：backward 不修改 `T`、`M`、上游梯度。
- **零上游梯度**：`G ≡ 0`（如被 mask 的 loss）时 `mu = 0`，
  `solve_backward` 精确短路（λ=0 是 Qλ=0 的精确解，0 次 PCG 迭代），
  返回精确梯度，不触发 PCG breakdown guard。

## 8. Examples

```python
import torch
from splr_autograd import splr_ot

M = torch.rand(100, 120, dtype=torch.float64, device="cuda")
a = torch.full((100,), 1 / 100, dtype=torch.float64, device="cuda")
b = torch.full((120,), 1 / 120, dtype=torch.float64, device="cuda")

# 训练循环
P = torch.nn.Parameter(M.clone())
T_target = splr_ot(M + 0.1 * torch.randn_like(M), a, b, eta=0.05).detach()
opt = torch.optim.Adam([P], lr=0.03)
for _ in range(40):
    opt.zero_grad()
    T = splr_ot(P, a, b, eta=0.05)
    loss = ((T - T_target) ** 2).mean()
    loss.backward()
    opt.step()
```

注意：对 `M` 加**常量** c 不改变 EROT plan（被 u/v scaling 吸收），
构造 target/扰动时不要用常量平移。

## 9. Validation 摘要

沙盒（无 CUDA）验证：fake curegot（CPU Sinkhorn forward + 忠实
FakeCtx）注入 `sys.modules` 后，跑**真实 autograd 链路**——
`validate_stage4_autograd.py` 全部 16 项测试经 CPU dry-run 通过
（仅 CPU-input 子项与 CUDA memory 统计需在真机执行）：

- manual equivalence（线性/非线性 loss、多分支、多 context、逆序、
  retain_graph）：rel err = 0，**bitwise**。
- sampled FD（n=5, m=6，8 元素，eps 1e-4→3e-6 sweep）：normalized
  error 最好 3.1e-10（CPU fake forward 为收敛 Sinkhorn，此检查是
  真实梯度正确性检查，非仅链路检查）。
- `torch.autograd.gradcheck`：zero-G 修复后返回 True（非致命项）。
- training loop（Adam lr=0.03，40 步）：loss 4.8e-04 → 1.1e-05。
- 9 类 error path 全部按预期 raise；错误后干净调用不受影响。
- zero upstream gradient：精确短路，0 次 PCG 迭代。

真机命令：

```bash
python validate_stage4_autograd.py        # 期望: STAGE 4 AUTOGRAD VALIDATION: PASS
python benchmark_stage4_autograd.py --quick
python benchmark_stage4_autograd.py       # 4 sizes, repeat=5
# Stage 3 回归
python stage3b_validate.py                # 三项 oracle 回归
python benchmark_stage3c3.py --quick
python benchmark_stage3c4.py --quick      # default==diag bitwise + lifetime
```

判据：FD normalized error ~1e-4 或更好（solver-tolerance 所限，接受
≤1e-3）；integration overhead <5% 好、>10% 才需 audit；3B/3C 回归
不变（Stage 4 未触碰其数学路径；`solve_backward` 的 zero-G 短路仅在
mu 精确为零时触发，不影响任何既有 benchmark 的 bitwise 检查）。

## 10. Known limitations

- 仅 dL/dM；a/b/eta 梯度未实现（显式 raise，不静默）。
- 仅 CUDA float64；无 AMP；double backward 不支持（见第 6 节线性
  loss 的上游限制说明）。
- 非收敛一律 raise——若用于超参搜索等"失败是常态"的场景，请用
  `splr_ot_with_info` + manual `solve_backward`（flag 语义）。
- `torch.autograd.gradcheck` 在真机上可能因 solver 容差/非确定性
  数值路径而不稳定；这是预期行为（Stage 4 prompt 第 18 节），
  sampled FD 是权威检查。不为 gradcheck 改 solver。
