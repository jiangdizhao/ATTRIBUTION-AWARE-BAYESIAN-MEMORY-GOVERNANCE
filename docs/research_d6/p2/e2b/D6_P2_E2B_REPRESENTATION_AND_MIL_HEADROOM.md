# LP-D6-P2-E2-B 表示与 MIL Headroom

## 判定

**表示与 MIL headroom 尚未被实验裁决。** 只完成了预定的一个 D-fit profile；资源投影超过计划窗口，按协议停止。因此没有 G-CLS/G-Mean/G-Concat 与 M-BCE 的未来样本比较，也没有 M-Rank-Softplus 增益比较。

## 已知 profile

- 唯一模型：DINOv2 patch MIL，BCE，seed 0。
- 特征：E2-A1 冻结的 `d6.e2.shared_spatial_tokens.v1`；本轮未读 backbone 权重、未提取特征。
- D-fit：52 个二元图像 bag；52 之外的 2 个 uncertain 标签不进入拟合。
- 初始优化通过：objective 0.09816，`||grad||_inf=9.02e-9`，101 次迭代，峰值 RSS 约 716 MiB。
- 全矩阵预登记计划：336 次完整历史拟合。保守估算 3,979.5 CPU 秒，超过 1,800 秒窗口；主矩阵未运行。

## 当前可说与不可说

可说的是：冻结特征缓存和被选中的 D-fit MIL-BCE 初始优化能够通过输入检查与数值门槛。不能据此判断：

1. patch aggregation 是否比单位化全局 mean 更有利；
2. Softplus ranking 是否带来额外收益；
3. 在线属性反馈是否改善未参与更新的未来样本；
4. DINOv3 是否优于 DINOv2；
5. 高分 patch 是否对应真实可见擦伤区域。

最后一项还缺少独立属性级区域真值；本任务没有读取 mask，也未创建 patch 标签。

## 混淆与外推限制

旧的 D-fit evaluator-only 统计显示来源 scratch 样本 18/18 被人工标为 present；因此不能把未来如果观察到的属性预测收益直接归因于视觉属性概念，而不先检验 defect/source shortcut。Real-IAD D-dev 也存在来源类型与人工属性相关的风险。本轮没有使用 defect type 作为训练监督或 learner feature，也没有生成新的来源-属性结果表。

即使后续经批准完成 E2-B，当前 24 个 sample-key 仍只是操作性分组，不能称为已证明相互独立的 24 个物理工件；D-dev 已曝光，不是 D-conf。任何后续结论都应限定为当前 tape 和开发对象上的探索性证据。

## 资源建议

当前唯一建议是由研究负责人先审阅本次 profile 和停止原因，再明确授权资源计划（延长 CPU 时间、单独授权 GPU 方案或批准精简矩阵中的某一种）。本报告不授权重跑 profile、启动矩阵或更改 loss、seed、精度、求解器、评价组。
