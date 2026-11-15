# D6-P2-E2B-R1 — 审核结论与下一步

## 执行裁决

**状态：`E2B_R1_PARTIAL`。** 28 个 learner 中 27 个完整到 checkpoint 12；一个注册 learner 因梯度验收门槛失败而终止。28 个 B0 控制全部完成。未超资源预算，所有已写入记录通过身份与 SHA 聚合审计。

不应把该结果称为完整矩阵 PASS，也不应把失败的轨迹补齐或在此报告后自行重试。旧 R0 状态继续是 `E2B_BLOCKED / OVER_BUDGET`；R1 的资源授权不会改变 R0 的结论。

## 科学结论范围

可支持：在冻结视觉特征、已标注 D-fit 与固定稀疏反馈 tape 下，普通全历史属性头更新在小型 Real-IAD 开发评价上可改变未来预测；W 的匹配 BA 增益一致，X 的平均 BA 增益为正但只有 6/13 提升、6/13 持平、1/13 下降。结果还显示来源 defect type 不能完全决定人工属性，但强关联仍存在。

不支持：

- 已证明 A 方案可发表或相对普通在线学习有方法新颖性；
- 对未见工业产品、独立物理工件或 D-conf 的泛化；
- 所有属性来源混杂均已消除；
- MIL 或 ranking loss 普遍优于全局 readout/BCE；
- 纠正风险已被治理；本轮确有 10（X）和 13（W）次终点二元错误新增；
- E2B 完整验收 PASS。

## 建议的唯一下一步

将本次四份结果报告、Git-ignored 聚合结果与 1 条失败轨迹交由研究负责人审核，并保留 `PARTIAL`。下一轮若负责人仍需完整 28/28 learner 矩阵，应另行批准一次**只针对 DINOv2/G-CLS/X event 10 的数值诊断或恢复决策**；在新授权前不要重跑该轨迹、不要更换优化器/容差、不要执行 E2-G1 或新增数据实验。

任何进一步的确认性结论都需要未曝光的、对象级隔离的独立数据；D4-Lite、D4B 与 Real-IAD D5R pilot 的所有已曝光图像及可识别近重复均须从 D-conf 排除。历史 sealed/HOLD 的全文件级重叠状态仍沿用既有 UNKNOWN，不得改写为已排除。

## 归档与工作区

本任务新增报告：

- `D6_P2_E2B_R1_EXECUTION_AND_NUMERICAL_AUDIT.md`
- `D6_P2_E2B_R1_COMPLETE_RESULTS.md`
- `D6_P2_E2B_R1_REPRESENTATION_AND_MIL_HEADROOM.md`
- `D6_P2_E2B_R1_REVIEW_AND_NEXT_ACTION.md`

加上此前冻结的 `D6_P2_E2B_R1_RESOURCE_OVERRIDE_AND_PROTOCOL.md`，形成完整报告集。R1 代码与合成测试在 `scripts/research_d6/p2/e2b_r1/`、`tests/research_d6/test_e2b_r1.py`；结果与逐样本明细只保留在 Git-ignored 输出目录。没有修改已跟踪文件，没有 commit/push。
