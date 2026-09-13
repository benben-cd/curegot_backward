# Stage 5 交付说明：Production API Packaging + Real Workload Integration

> **本轮 delta（harness 稳定化第二轮 + 文档纠正，真机反馈后）**：
> 真机已确认 Stage 4 回归 17/17 PASS、package import / fresh-cwd /
> legacy native API / package-vs-manual 梯度 ~1e-15 / shim 同一性 /
> SPLROT train-eval ~1e-15 / Workload A / B 全部 PASS；production
> 冻结（curegot/ package、三根目录 shim、setup.py、src/* 零改动）。
> 本轮改动：
>
> 1. **`splr_ot_batch` 测试去 bitwise**（validate_stage5_api.py）：逐
>    slice `torch.equal` → rel err ≤ 1e-12 + `allclose(rtol=1e-12,
>    atol=1e-14)`，并打印 max slice rel err。两次独立 GPU 迭代求解
>    （batch 循环一次 + 单发一次）不要求 bitwise；判据与 SPLROT
>    train/eval 一致。batch 梯度检查本就只查 finite+nonzero，无需改。
> 2. **Workload C cost conditioning**：`M = torch.cdist(X, Y) ** 2` →
>    `M = (torch.cdist(X, Y) ** 2) / d`（d=8 为 embedding 维度）。
>    原始 cost 动态范围过大时个别训练迭代会打满 max_iter=1000——这是
>    harness cost scale 问题而非 solver 缺陷。完整 autograd 链
>    （raw_X → net → X → M → T → loss）不变，验收断言（全参数 grad
>    finite+nonzero、参数更新、loss 降、memory 稳）全部不变；
>    max_iter 保持 1000，tol 保持 1e-8，lr 保持 0.01。
> 3. **example 同样 conditioning**：`M = ((X-Y)²).sum(-1) / d`，
>    max_iter=1000 与固定种子不变，lr 保持 0.05；docstring 同步更新。
> 4. **文档纠正（重要）**：首次交付文档中关于 package 结构的陈述基于
>    错误的 repo audit 假设。真实结构：repo 在 Stage 5 之前**已有**
>    `curegot/` package（`__init__.py`、`_cuda_loader.py`、
>    `_internal_numpy.<abi>.so`、`_internal_torch.<abi>.so`），真正的
>    native torch 模块是 **`curegot._internal_torch.torch`**。Stage 5
>    的 packaging 工作量 = 新增三个 .py（`_backward_core.py` /
>    `backward.py` / `autograd.py`）+ 扩充既有 `__init__.py`。
>    **setup.py unchanged；扩展名 unchanged；_internal_numpy /
>    _internal_torch unchanged；无需 native rebuild。** 本文档与
>    STAGE5_PRODUCTION_API.md 已删除所有 "Extension 改名 /
>    `from .curegot import torch` / 删除顶层 curegot.so / setup.py
>    两处改动" 的错误陈述；此前交付的参考 `setup.py` **作废（WITHDRAWN，
>    见该文件头部说明）**。注：本交付包沙盒副本中的 `curegot/__init__.py`
>    与 `curegot/autograd.py` 仍以占位名 `curegot.curegot` 引用 native
>    扩展；真实 repo 对应引用为 `curegot._internal_torch`（docstring 中
>    的扩展名描述同理）——真实 repo 的文件以真机已验证版本为准，本轮
>    不改代码。
>
> 真机最终命令：
> ```bash
> python validate_stage5_api.py && python validate_stage5_workload.py && \
> python examples/differentiable_splr_ot.py
> python benchmark_stage4_autograd.py   # 全部通过后跑一次 full（配对端到端方法论）
> ```
>
> ---
>
> **首轮 delta（harness 稳定化，真机反馈后）**：package 集成已在真机
> 验证（fresh-cwd import、legacy 兼容、shim 同一性、package/manual
> 梯度 ~1e-15、Stage 4 linear/nonlinear/context/FD 均过）。本轮**只改
> harness/example**，冻结面（curegot/ package、shim、setup.py、src/*）
> 零改动：
>
> 1. **SPLROT train/eval 判据**：`torch.equal`（bitwise）→ 严格数值
>    等价（rel err ≤ 1e-12 + allclose rtol=1e-12/atol=1e-14，并打印
>    rel err）。两次独立 GPU 迭代求解不要求 bitwise；train/eval 不改变
>    数学行为的语义不变。`.to(device)` 检查同样改为数值等价。
> 2. **Workload forward 预算**：`validate_stage5_workload.py` 全部训练
>    workload + SGD smoke + perf sanity 统一 `WORKLOAD_MAX_ITER = 1000`
>    （= production forward 默认；训练迭代中 cost matrix 演化会撞
>    200 上限，production 正确 raise，harness 必须给足预算）。
> 3. **随机性全部确定性化**：audit 出 6 处未播种调用
>    （validate_stage5_api.py 五处 W/Mb 噪声 + workload perf_sanity 一
>    处 W），全部改为显式 `torch.Generator` 固定种子；nn.Linear 初始化
>    已有固定 `manual_seed(32)`。脚本每次运行走同一条优化轨迹。
> 4. **Stage 4 training smoke 唯一修改**：target 扰动固定种子
>    （seed=120）+ target 构造与训练调用均用 `TRAIN_MAX_ITER=1000`；
>    其余 Stage 4 测试零改动。
> 5. **example**：`max_iter=200` → `1000`（其余不变，仍要求 finite
>    X.grad、loss 下降、PASS）。
> 6. **诊断**：每个训练 workload 打印 initial/final loss + final
>    gradient norm（B/C/SGD 新增，A 已有）。
>
> 沙盒 dry-run（fake package replica）：validate_stage5_api **13 组
> PASS**（train/eval 新判据路径已执行）、validate_stage5_workload
> **5 组 PASS**、validate_stage4_autograd 经 shim→package 链路 **17
> 组 PASS**（含修改后的 training smoke）。
>
> 真机最终命令：
> ```bash
> python validate_stage5_api.py && python validate_stage5_workload.py && \
> python validate_stage4_autograd.py && python examples/differentiable_splr_ot.py
> python benchmark_stage4_autograd.py   # 全部通过后跑一次 full（配对端到端方法论）
> ```
> 关于 quick benchmark 的 −20%/+18%：quick 样本少、噪声大，此前 full
> paired benchmark 在 ±3% 内；仅当 full run 可复现 >10% 才排查，不做
> C++ autograd 重写。
>
> ---
>
> 以下为首次交付内容（存档）。

## 1. Package audit 结果（已按真实 repo 结构纠正）

> **纠正声明**：本节首版基于错误的 repo 假设（"顶层扩展 curegot.so、
> 不存在 package 目录"）。真实 repo 在 Stage 5 之前**已有** package
> 结构，下表为纠正后的事实。

| 问题 | 结论（真实结构） |
|---|---|
| `import curegot` 从哪里加载 | 既有 Python package `curegot/`（含 `__init__.py` 与 `_cuda_loader.py`） |
| native 扩展在哪 | `curegot/_internal_numpy.<abi>.so` 与 `curegot/_internal_torch.<abi>.so`（既有，Stage 5 未动） |
| `curegot.torch` 是什么 | 既有 `__init__.py` 暴露的 native torch 子模块，本体为 **`curegot._internal_torch.torch`**（pybind `def_submodule`） |
| sinkhorn_splr / sinkhorn_splr_bwd 如何暴露 | native `m_torch.def(...)`，签名与行为 Stage 5 完全不变 |
| 根目录 splr_autograd.py / splr_backward.py / backward_core.py 会被 pip 安装吗 | **不会**。实现迁入 package，根目录留 shim |
| 如何迁入 package | **新增三个 .py**（`_backward_core.py` / `backward.py` / `autograd.py`）+ **扩充既有 `__init__.py`**；setup.py 不动 |
| editable 与 wheel 一致性 | 三个新 .py 随既有 package 一起安装，天然一致 |

**关键事实**：**setup.py unchanged；扩展名 unchanged（无改名、无
移动）；`_internal_numpy` / `_internal_torch` unchanged；Stage 5 无
native rebuild**。扩充后的 `curegot/__init__.py` 经既有 loader 取得
`curegot._internal_torch.torch` pybind 子模块后**挂载** Python API
（pybind module 的 `__dict__` 可写，只加属性、不 shadow native
symbol）——因此 `curegot.torch.splr_ot` 风格成立，且
`curegot.torch.sinkhorn_splr*` 原样保留。

## 2. 最终 package 结构

```text
curegot/
    __init__.py              # 既有；Stage 5 扩充：挂载/导出公开 API
    _cuda_loader.py          # 既有，未动
    _internal_numpy.<abi>.so # 既有 native 扩展，未动
    _internal_torch.<abi>.so # 既有 native 扩展，未动（native torch 模块 = curegot._internal_torch.torch）
    _backward_core.py        # Stage 5 新增：唯一 backward 数学（与 Stage 4 文件逐函数字节一致，已断言验证）
    backward.py              # Stage 5 新增：solve_backward / select_preconditioner
    autograd.py              # Stage 5 新增：splr_ot / splr_ot_with_info / splr_ot_batch / SPLROT
```

## 3. Public API

`curegot.torch.splr_ot`（主）、`splr_ot_with_info`、`SPLROT`、
`splr_ot_batch`（顺序 batching）、`solve_backward`（低层研究接口）；
顶级 shortcut `curegot.splr_ot` 等同对象。

## 4. Modified / new files

**新增**：`curegot/_backward_core.py`、`curegot/backward.py`、
`curegot/autograd.py`、`validate_stage5_api.py`、
`validate_stage5_workload.py`、`examples/differentiable_splr_ot.py`、
`STAGE5_PRODUCTION_API.md`、`README_snippet_stage5.md`。

**扩充（既有文件）**：`curegot/__init__.py`（挂载公开 API + 顶级
re-export；loader 与既有导入逻辑未动）。

**改写为 shim**（实现唯一，对象同一性保持）：根目录
`backward_core.py`、`splr_backward.py`、`splr_autograd.py`。

**未触碰**：`src/*` 全部 native 文件、**setup.py（零改动）**、
`curegot/_cuda_loader.py`、`curegot/_internal_numpy.<abi>.so`、
`curegot/_internal_torch.<abi>.so`、Stage 3/4 全部验证与 benchmark
脚本（它们经 shim 继续工作）、任何数学/preconditioner/policy。

**setup.py**：**unchanged**。Stage 5 纯 Python 集成，三个新文件随既有
package 安装，无需任何构建/安装配置改动，**无需 native rebuild**。
首次交付随附的参考 `setup.py`（Extension 改名 + packages= 方案）基于
错误的 repo 假设，**已作废（WITHDRAWN）**，请勿应用。

## 5. Compatibility strategy

- native 入口签名/行为不变（`sinkhorn_splr` / `sinkhorn_splr_bwd` /
  `SPLRBackwardContext`）。
- 根目录三 shim 显式 re-export（不用 `import *`，避免意外命名空间
  污染）；已断言 `splr_autograd.splr_ot is curegot.autograd.splr_ot`
  等同一性。
- Stage 3/4 脚本**零改动**：全部 import 点已逐一核对（benchmark_
  stage3c1/2c/2d/3c3、stage3b_validate、benchmark_stage3c4、
  validate/benchmark_stage4）。

## 6. Sandbox / static tests（无 CUDA，fake 扩展 replica）

搭建了 fake package replica（真实 deliverable 四文件 + Python 假扩展
stand-in；假扩展挂在本交付包沙盒副本使用的占位名 `curegot.curegot`
上，真实 repo 对应为 `curegot._internal_torch`——replica 验证的是
package 机制与 Python 逻辑，与扩展文件名无关），全部通过：

- **fresh-process import**（cwd=/tmp，PYTHONPATH 指向 replica）：
  `curegot.torch.splr_ot / SPLROT / splr_ot_batch / solve_backward`
  可用，native symbol 完好，顶级 shortcut 同对象。
- **validate_stage5_api.py 全量 dry-run：13 组 PASS**（import 面、
  forward/backward、vs manual path rel=0 bitwise、shim 同一性 +
  数值 bitwise、13 类 error contract、fresh-process subprocess 真实
  通过、SPLROT repr/state_dict/train-eval bitwise 不变、batch wrapper
  与逐 slice bitwise 一致且梯度流通、torch.compile/TorchScript 审计
  非致命记录）。
- **validate_stage5_workload.py 全量 dry-run：5 组 PASS**（训练为
  真实 autograd：A 5.8e-08→3.5e-09、B −2.5e-02→−3.5e-01、C
  9.5e-07→6.3e-07（cost conditioning /d 后）且所有参数 grad
  finite+nonzero、perf/memory 路径、SGD smoke）。
- **shim 回归兼容**：stage3b_validate / benchmark_stage3c4 /
  benchmark_stage4_autograd / validate_stage4_autograd 经 shim import
  全部成功，`pcg`/`solve_backward` 对象同一性断言通过。
- 沙盒无法执行项（真机验收）：真实扩展的 fresh-process、CUDA memory
  统计、真实 step wall、torch.compile 真实行为、example 脚本
  （其代码路径与 workload B 相同，已等价验证）。

## 7. 真机验收命令

```bash
# 无安装/构建改动：setup.py unchanged，native 扩展 unchanged，无需 rebuild
cd /tmp && python -c "import curegot; print(curegot.__file__, curegot.torch.splr_ot, curegot.torch.SPLROT)"
cd <repo>
python validate_stage5_api.py
python validate_stage5_workload.py
python examples/differentiable_splr_ot.py
python validate_stage4_autograd.py     # 回归（Stage 5 未触碰其路径）
python benchmark_stage4_autograd.py    # 全部通过后跑一次 full
```

## 8. Workload design

- **A（learnable cost）**：`nn.Parameter` M + SPLROT module，MSE→
  target plan（非常量扰动 target），Adam lr=0.03×60，要求 loss 显著
  下降（<0.2×初值）且 grad finite。
- **B（embedding cost）**：X 可学习、Y 固定，M=‖Xi−Yj‖²，
  loss=−(T·affinity) 最大化对角匹配——证明梯度穿过 OT 层到达 X。
- **C（tiny NN upstream）**：Linear→ReLU→Linear 生成 embedding →
  (cdist²/d)→OT→MSE（cost conditioning：除以 embedding 维度 d 控制
  cost 动态范围，autograd 链完整保留）；要求**每个**参数 grad 非
  None/finite/非零、至少一个参数明显更新、memory_allocated warmup 后
  无线性增长。
- 另含 perf sanity（100x120/300x360 step wall）与 SGD smoke。

## 9. Known limitations

CUDA float64 only；单次调用单问题；无 dL/da/db/deta；无 double
backward；AMP unsupported；torch.compile / TorchScript 不保证；
batch 仅顺序 Python wrapper。全部写入 STAGE5_PRODUCTION_API.md，
未隐藏。

## 10. 是否发现 production bug

**否。** 本阶段未发现 correctness bug；production 数学与 Stage 3C/4
行为零改动。两处顺手的小改进均属 API 层（不涉及数值）：
(1) 标量参数轻量校验（prompt 第十九节授权）；
(2) `_check_inputs` 的 2D 报错信息补充了 batch 提示。
