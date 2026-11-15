# LP-D6-P2-E2-B 协议与损失注册

**授权日期：** 2026-11-16  
**注册时点：** 在 E2-B 任何未来评价结果产生之前  
**起始身份：** `stage3-eg-famar-execution` / `1e6a602cb8b788bca03a61773ef442a55f3fa7e2`  
**范围：** 固定 E2-A1 特征缓存、已注册 D-fit 标签和现有 W/X tape。

## 排序损失修订

D6-P2 v0.3 原先登记的是不可微 hard-hinge MIL 排序项：

`L_rank_hinge = mean_{i in P,j in N} max(0, m - S(B_i) + S(B_j))`。

本次 E2-B 任务明确批准并修订为单独命名的 **M-Rank-Softplus**：

`u_ij = m - S(B_i_positive) + S(B_j_negative)`  
`L_rank = mean_pairs [kappa * log(1 + exp(u_ij / kappa))]`  
`J_MRank = J_BCE + rho * L_rank`。

固定 `m=1.0`、`kappa=0.1`、`rho=0.1`。它不是 v0.3 hard hinge 或论文原版损失的精确复现；不得根据结果切回 hinge、搜索参数或删除 BCE 项。

## 输入、监督与事件顺序

- D-fit 有 54 张图像：27 `present`、25 `absent_assessable`、2 `unobservable_or_uncertain`。只有 52 个二元标签用于训练。
- W 与 X 各使用冻结 tape 的 12 个反馈事件和 24 个未来评价视角。排除/未标注记录不进入特征输入或评价。
- W 的 uncertain 事件占用事件序号，不添加二元样本，也不执行重复拟合。
- 每个事件按“先预测并记录模型状态，再揭示合法标签，二元标签入完整历史，之后重拟合”执行。
- `uncertain` 与 `not_applicable` 不进入二元损失。重复呈现按图像只取首次标签，不增权。
- 未来标签仅由 evaluator 使用，不得进入初始化、优化、停止、模型选择、阈值或超参数选择。
- 产品、sample-key、视角、来源类型、原始缺陷代码、来源 split、mask 和标注元数据均不作为 learner 特征。
- D-dev 已曝光，只构成探索性开发证据，不是独立确认集。

## 冻结特征与目标

- `G-CLS`：缓存的单位化 CLS 向量。
- `G-Mean`：仅对有效单位 patch token 求算术平均，再作 L2 normalization。
- `G-Concat`：拼接单位化 CLS 和单位化 patch mean，再对拼接结果作 L2 normalization。
- MIL bag 只包含有效 patch；`S(B)=tau * [logsumexp(a_p/tau)-log(P)]`，`tau=1.0`，`a_p=w^T z_p+b`。
- 全局头和 MIL BCE 使用平均二元交叉熵加 `lambda/2 * ||w||^2`，`lambda=1e-4`；截距不正则化。
- Rank 对当前完整合法历史中的每个唯一正负 bag 配对计算一次，均匀平均。负 bag 可含其他缺陷；不复制标签、不按 patch 数加权。
- 不把正图像全部 patch 当作正 patch。无属性 mask，MIL 高分 patch 不被宣称为人工确认定位。

## 初始化、求解器与失败门槛

- 全局头从全零参数开始；在线更新 warm-start 最近一个验收 checkpoint，并在完整合法历史上重拟合。
- MIL seeds 为 0、1；使用 `numpy.random.default_rng(seed)`，权重按 `Normal(0, 1/sqrt(D))` 初始化，bias 为零。同 seed 的 M-BCE 与 M-Rank-Softplus 共用完全相同初态，W/X 共用同一 D-fit 初始解。
- Full-batch SciPy L-BFGS-B，FP64，CPU 最多 2 个计算线程：`ftol=1e-12`、`gtol=1e-8`、`maxiter=10000`、`maxls=50`、`maxcor=10`。验收要求 SciPy success、参数/目标/梯度有限且 `||grad||_inf <= 1e-6`。仅 FTOL 停止但梯度超限记为 `FTOL_ONLY_NOT_CONVERGED`；失败停止该轨迹，不换优化器、不放宽门槛、不挑 seed。
- MIL 目标非凸；验收仅表示局部求解满足注册标准，不表示全局最优。

## 主矩阵与评价

- 全局：2 backbone × 3 readout × 2 protocol = 12 条轨迹。
- MIL：2 backbone × 2 方法 × 2 protocol × 2 seeds = 16 条轨迹。
- 合计 28 条更新轨迹，每条配对自身 checkpoint-0 冻结 B0。checkpoint 记录 0–12。
- 主指标：X、checkpoint 12、产品宏平均 Balanced Accuracy。配套报告 paired product-macro AP、明确标为 degenerate 的全正 AP、产品宏 AUC、每产品 support、正预测率、逐 checkpoint 曲线、有害/纠正翻转、求解器诊断、时间和内存。
- BA/AUC 仅在同时有正负类的产品上定义；绝不填 0 或 0.5。未知状态不进入二元指标。固定阈值 score 0，平分判为正。
- 预注册比较包括 G-CLS/G-Mean、G-Mean/G-Concat、G-Mean/M-BCE、配对 seed 的 M-BCE/M-Rank、同读出 DINOv2/DINOv3，以及每个方法的 checkpoint-0 到 12 变化。

## 资源门槛

先只用 D-fit 做一次 DINOv2 M-BCE seed-0 完整历史拟合；禁止读取未来标签或报告未来性能。计划窗口固定为 1800 秒。实现预先采用保守投影 `2 × 单次 profile process-CPU-seconds × 336 个全矩阵拟合`；profile 失败或投影超窗即停止，请负责人另行决定资源/范围，不得缩矩阵、换精度或换 optimizer。

本次 profile 通过数值验收，但投影超窗，因此未运行主矩阵。

## 科学边界

24 个 Real-IAD sample-key 只是操作性分组，物理对象独立性未被证明；样本小且有来源类别富集。需披露来源缺陷类型与人工属性可能混杂；不得把来源类型当属性监督。结果即使将来获得，也不支持真实发生率、前瞻性零样本迁移、主动查询节约或经确认的擦伤定位结论。
