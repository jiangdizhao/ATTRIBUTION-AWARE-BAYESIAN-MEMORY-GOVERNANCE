# LP-D6-P2-E2-B 结果状态

**最终状态：`E2B_BLOCKED`**  
**原因：** D-fit profile 数值收敛，但预注册完整矩阵资源投影为 66.3 分钟，超过 30 分钟 CPU 计划窗口。主矩阵没有启动。

## 实际执行

| 范围 | 注册数量 | 实际完成 |
|---|---:|---:|
| DINOv2 D-fit M-BCE seed 0 bounded profile | 1 | 1，数值 PASS / 资源门槛 BLOCKED |
| Global 在线轨迹 | 12 | 0 |
| MIL 在线轨迹 | 16 | 0 |
| 全矩阵 | 28 | 0 |
| B0 对照 | 随方法配对 | 0 |

12 条 global 原计划为 2 backbone × 3 readout × X/W；16 条 MIL 原计划为 2 backbone × 2 loss × X/W × seed 0/1。没有产生任何方法的 checkpoint-0 或 checkpoint-12 未来预测。

## 结果字段

| 科学问题 | 结果 |
|---|---|
| 各方法 X/W 的 BA、AP、AUC 与产品 support | `NOT_RUN`；无可报告结果 |
| MIL-BCE 是否优于全局空间 pooling | `NOT_TESTED` |
| M-Rank-Softplus 是否优于配对 M-BCE | `NOT_TESTED` |
| DINOv3 与 DINOv2 差异 | `NOT_TESTED` |
| 在线反馈是否改善未来视角 | `NOT_TESTED` |
| 有害/纠正预测翻转 | `NOT_GENERATED` |
| 各产品类别支持与 degenerate AP | 未由 E2-B evaluator 复核；不得由本报告补算未来真值 |

这些空缺是资源门槛触发后的预定停止结果，不是模型表现为零、算法失败或方向 A 被否定。E1 `PARTIAL` 也不是本实验的匹配对照。

## 研究解释边界

本轮仅确认所选 DINOv2 MIL-BCE 初始 D-fit 优化在当前环境可达注册数值精度。它不说明其泛化、反馈收益、MIL headroom 或定位能力。当前数据仍是已曝光的小规模开发集；既有 D-fit 来源统计曾提示 scratch 来源与人工属性的强相关风险，但本轮未完成主矩阵 evaluator-only 交叉表，也未建立对该混淆的模型诊断。

不得据此宣称 MIL、在线学习、DINOv3 或跨产品迁移优于/劣于任何基线，也不得进入 E2-G1、增加标注或开发 A1/A3。
