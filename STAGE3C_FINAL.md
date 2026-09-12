# Stage 3C FINAL：backward preconditioner 最终结论与 production policy

> 状态：**Stage 3C COMPLETE**（待真机最终回归/benchmark 确认后冻结）。
> 本文档是 Stage 3C 系列的收官总结，限定条件见末节——结论不外推。

## 1. Correctness（Stage 3A / 3B oracle → 最终 native backward）

- Stage 3A 建立 reduced-dual implicit differentiation 的 dense oracle；
  Stage 3B 验证 native backward context（SPLR apply、CSR export、secant
  材料、fallback 路径）与 oracle 一致。
- 最终 native diagonal 路径：与 Python benchmark callback 的逐元素一致性
  误差为 **0**（`inv_diag` 向量与单次 apply 均 bitwise 相同——同一份 CSR
  值、同样的 float64 倒数与乘法）；PCG λ / gradient 对 dense oracle 的
  相对误差 ≤ 1e-8；Stage 3B oracle 持续保留。

## 2. Factorization reuse（Stage 3B.2 / 3C-1）

- `adopt` 模式实现 **zero-extra-factorization handoff**：forward 最后一次
  迭代的 cuDSS factorization 原样转移给 backward context，不为 backward
  重做 analyze/factorize。
- Stage 3C-4 audit 确认：该 factorization 是 **forward 算法本身必需**
  （每次迭代的方向求解），不是为 backward SPLR 保留的额外计算；diag
  默认下不存在"本可避免的 backward-only factorization 成本"。保留的
  backward-only state 仅为少量内存（rhs/sol 缓冲、低秩别名、pinned
  buffer、`d_inv_diag`），无额外算力开销。故不重构。

## 3. Bottleneck diagnosis（Stage 3C-2A / 2B）

- 修正确认的 instrumentation 后，SPLR apply 的 host wall 50%→88%（随规模）
  是 **cuDSS solve**；Python/PyBind 开销占比 ~1% 量级，**不是瓶颈**。
- 逐行源码 audit + cuDSS 官方模式核对：**no redundant cuDSS state setup
  found**——solve 已是单个 `cudssExecute(PHASE_SOLVE)`，descriptor /
  config / data / workspace / stream 全部 persistent，即官方推荐的
  repeated-solve 模式。multi-RHS 对严格串行的 backward PCG 无收益，不实现。
- 结论：强 SPLR preconditioner 的单次 apply 成本接近 API 下限，无法靠
  消除冗余来降低——只能换更便宜的 preconditioner。

## 4. Trade-off（Stage 3C-2C / 2C1）

统一系统、共享 HVP break-even 分析（清理后的 methodology）：

| size | diag wall | diag speedup vs CG |
|---|---|---|
| 100x120 | 12.01 ms | 1.41x |
| 300x360 | 8.93 ms | 1.75x |
| 500x600 | 8.11 ms | 1.74x |
| 1000x1200 | 7.30 ms | 1.97x |

- SPLR 的迭代数更少，但 `prec_over_break_even` 大多 > 1：
  **更强的 preconditioner 减少迭代，却不降低总 wall**。
- cheap diagonal 在中大规模有最佳 cost-benefit。

## 5. Structured exploration（Stage 3C-2D）

- Variant B（2-block diagonal）audit：reduced Hessian 的 D_r/D_c 本身即
  对角 → **精确退化为普通 diagonal**，不实现。
- Variant C（paired 2×2 block-Jacobi，strongest-edge greedy + 显式逆 +
  PD guard）：数学上更强（每 size 少 0–3 次迭代），但 apply 更贵，
  总 wall 全部落败，且规模越大相对越差：
  100x120 14.54 vs 13.93ms；300x360 10.75 vs 9.28ms；
  500x600 11.55 vs 9.52ms；1000x1200 10.34 vs 7.93ms。
- 触发预定义 stop rule：**diagonal is the best practical preconditioner
  among tested variants**，停止 block preconditioner 设计。

## 6. Final production result（Stage 3C-3 / 3C-4）

Native diagonal（context-owned `d_inv_diag`，setup 一次性提取 + 校验
~86–95µs；apply 为单异步 elementwise kernel）：

| size | native diag | vs CG | vs SPLR |
|---|---|---|---|
| 100x120 | 12.03 ms | 1.23x | 1.20x |
| 300x360 | 9.57 ms | 1.50x | 1.54x |
| 500x600 | 8.69 ms | 1.62x | 1.97x |
| 1000x1200 | 7.30 ms | 1.93x | 3.03x |

- native 与 Python diag：迭代数相同、数值 bitwise 相同、wall 基本持平
  （大规模略优）。
- lifetime 20× create/destroy + repeated apply：PASS。

**最终 policy**：

```
default:   preconditioner="diag"   (native)
explicit:  preconditioner="splr"   (cuDSS + rank-2，完全保留)
debug:     preconditioner="none"   (plain CG，ablation only)
非法值:    ValueError
```

无 size threshold——四个测试规模全部 diag 更优，无需引入未广泛验证的
复杂 policy。

## 7. Production API

新增 `splr_backward.py`：

```python
from splr_backward import solve_backward

out = solve_backward(T, G, ctx, eta)                    # default: diag
out = solve_backward(T, G, ctx, eta, preconditioner="splr")
# -> dict: lambda / grad / iterations / converged / preconditioner
```

- 单一 production 入口；PCG / RHS / 梯度装配使用 `backward_core.py` 中的
  唯一共享实现——Stage 3B validation 同样 import 并验证这同一份实现
  （production 不依赖 validation module）。
- `sinkhorn_splr_bwd`（forward + context 导出）语义完全不变。
- 兼容性：既有调用点全部是显式 `ctx.apply_preconditioner`（validation /
  benchmark 脚本），不存在带隐式 SPLR 默认的旧 public backward helper，
  因此 default=diag 不构成任何既有入口的行为变化。
- `extract_inv_diag` + `lambda r: inv_diag * r`（Python diag）仅作
  benchmark reference 保留在 benchmark 脚本中，**不进入 production**。

## 8. 限定条件（勿泛化）

以上结论仅在以下条件测得，**不要**泛化为 "diag universally optimal"：

- workload：随机生成的 EROT 问题（seed 固定），eta = 0.05；
- 规模：100x120 – 1000x1200（更大规模未测）；
- tolerance：forward 1e-8 / PCG 1e-12；density reference 0.10；
- hardware：单机单卡（见各阶段 benchmark 环境）。

更大规模、不同 eta/tolerance 或病态问题下，SPLR（或未来的更强变体）
可能重新占优——`preconditioner="splr"` 路径为此完整保留。
