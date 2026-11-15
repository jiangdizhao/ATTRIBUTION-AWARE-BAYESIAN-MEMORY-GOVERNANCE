# D6-P2-E1-R1 数值初态核验

**状态：`PASS`（仅初态门槛）**  
**执行 HEAD：** `4c4319397de9496526248214a0ebe9e678f5ede7`  
**分支：** `stage3-eg-famar-execution`

## 范围与数据核验

本 R1 是针对 E1 的一次授权协议修订：重求共同 D-fit 初态，并从该初态配对重跑 γ=0、1e-4、1e-3。G0 文档及旧 E1 BLOCKED 结果均未改写。P1-B 保存的 sklearn 状态只作为 L-BFGS-B 初始猜测和历史参照，不再充当新锚点。

仅使用既有 P1-A 特征缓存和已登记 P1 监督：102 条记录（54 D-fit、48 D-dev），D-fit 中 52 条二元有效（27 Present、25 Absent），1024 维、FP32、L2 归一化。反馈 tape SHA-256 为 `13561809074aad72e2f13289723ccba8addc4768395480c50d3173abf66e8cc1`。没有读取新图像像素、mask、D-conf 或 sealed/HOLD 数据；没有加载 Torch/DINO 或使用 GPU。

输入来源指纹：P1-A input manifest `820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63`；feature manifest `ee0232b29f64f9fd4f10198ba3ef20da89fa85b2083a389b774f6a9723bbb99d`。P1-A 输入快照运行前后均为 `580940b5bc9b204d6bded6a548e1cc76fdba65e8b05be1f8bfbbb8fc35d8b0c7`。原 P1-B 15 个输出均通过既有 SHA-256 清单。

## 初态求解

目标为 D-fit 二元样本上的平均 binary log loss 加 `λ/2 ||w||²`，`λ=1e-4`，截距不正则化。FP64 SciPy L-BFGS-B 固定 `ftol=1e-12, gtol=1e-8, maxiter=10000, maxls=50, maxcor=10`。沿用注册的 `||∇J||∞ ≤ 1e-6` 验收门槛。

| 指标 | 历史 P1-B 初态（仅猜测） | 新 R1 初态 |
|---|---:|---:|
| Objective | 0.36843788928793797 | 0.36843788074427364 |
| 梯度无穷范数 | 1.7226304473e-5 | **4.8268330935e-9** |
| L-BFGS-B 迭代 / 函数求值 | 历史记录 108 次 sklearn 迭代；终止原因未保存 | 33 / 40 |
| `||w_new - w_old||₂` | — | 0.0105535309 |
| `b_new - b_old` | — | -0.0074153642 |

新解满足目标、有限值和梯度门槛。以 12 个固定随机方向、中心差分 `ε=1e-6` 检查解析梯度，最大方向导数误差为 `1.2168e-10`，通过。新状态摘要 SHA-256：`86e8e8868e7d869ccc2430555db314014daf0f7befd47c0a529e12fa3057c6f2`。

新旧状态在四组历史 checkpoint-0 future scores 上的最大绝对差为 `8.3123e-4`。这记录为初态变化诊断，不要求与历史 sklearn 数值迭代完全相同。γ=0、1e-4、1e-3 的各协议 checkpoint-0 系数和截距数组均逐值完全相同；新初态的 `b0` 固定，proximal 项只约束权重。

## 保留的历史 BLOCKED 输出

R1 使用独立目录 `outputs/research_d6/p2/e1/run_20261115_r1_01/`。旧 `run_20261115_01/` 仍保留原 BLOCKED 文件；当前 SHA-256 记录如下，供后续检查。旧目录与新目录没有共用写入路径。

| 旧文件 | 当前 SHA-256 |
|---|---|
| `anchor_compatibility_gate.json` | `6e258aacec6684377756c7d3d5d2319cd39a24bd60404a4a8b9cfa81b1c9610c` |
| `archived_gamma0_product_support.csv` | `015aa65c77647c4a0d860df399ea689873278db316ca6586d14c9f06fa5c4e5b` |
| `archived_gamma0_readonly_diagnostic.json` | `48d0931b5b08e6af554890e99ccf93a2953e2ad7760b23aea086a16a97e2784d` |

## 执行源代码

| 文件 | 执行时 SHA-256 |
|---|---|
| `scripts/research_d6/p2/e1/run_e1.py` | `e0d198f4239e07b0a0e6729b0e48cb3a186204d8e07de53fd0ebd230a5fe7a9d` |
| `scripts/research_d6/p2/e1/proximal_logistic.py` | `6fc1a5765de46dfb3a51a4b2be8ee3205bbee480fd84b494352be877b971b2df` |

G0 批准提交 `a50459f6061a951ca8aa7d6f34bd0863b6145ebd` 是实际 HEAD 的祖先。后续执行记录分别保存实际 HEAD、G0 ancestor 状态及上列源文件哈希。
