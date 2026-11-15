# LP-D6-P2-E2-A0 — 审核与下一步

**日期：** 2026-11-15  
**最终状态：** `E2A0_PASS`

## 决策

两套冻结 backbone 在指定环境中均能对共同的 448×448 FP32 输入执行前向。Token shape、有效 patch mask、输出有限性、CPU/GPU 数值容差和重复 GPU 前向均通过。共同 `d6.e2.shared_spatial_tokens.v1` 转换器已实现，并用保存的真实合成 smoke token 再验证：CLS/register/patch 单位化、mask 数和 CPU/GPU 归一化表示一致性均通过。

因此，**具备向研究负责人申请下一步有限的 102 张已授权 source-side 图像特征缓存生成的工程条件**。本结论不授权该缓存读取，也不授权 E2-B。需由研究负责人另行批准后，才能按 G0 严格限定图像、manifest、缓存内容、存储预算及数据访问记录执行。

## 结论边界

- 这次只测试一个合成输入，不含属性或缺陷监督；没有性能或泛化结论。
- 未重跑 E1；E1 数值裁决仍为 `PARTIAL`。
- DINOv2 和 DINOv3 输出都先保留各自官方最终层语义；共同接口只增加逐 token L2 单位化并附带 patch mask，不意味着两种预训练表示相同。
- padding token 仍参与 attention，mask 仅控制之后的 pooling。
- 两套环境不同，因此 forward latency 不作为速度排名。
- DINOv3 的第一次报告写盘失败已保留；一次恢复运行得到通过结果。原 orchestrator 的 partial receipt 未覆盖，最终补充裁决另存。
- DINOv3 recovery JSON 中 token 总数的派生字段曾有计数错误（1812）；现已另存校正版 789，保留原记录且未重跑前向。该元数据错误不影响 smoke 结果和数值验收。

## 验证清单

| 检查 | 结果 |
|---|---|
| 分支与 HEAD | `stage3-eg-famar-execution` / `2d33f899c7c17157376b8be1bf9c311114ff2344` |
| DINOv2 GPU smoke | PASS；本地 checkpoint 与 P1-A 源树指纹匹配 |
| DINOv3 GPU smoke | PASS；本地 HF revision、weights/config/processor 指纹匹配 |
| Canonical preprocessing 单测 | 6/6 PASS；另完成两个 backbone 脚本与测试文件的 `py_compile` |
| Shared feature interface | 实际保存的两 backbone CPU/GPU token 均通过；单位范数误差最大 `1.2e-7` |
| 受保护研究资产 | 开始前与 recovery 后 14 项 tree manifest 全部一致 |
| Backbone 资产 | 开始前与 recovery 后 SHA/tree identity 一致 |
| 数据/训练权限边界 | 无研究图像、mask、标签、D-conf 或 HOLD 数据访问；无正式缓存、训练或 E2-B |
| 输出目录 | `outputs/research_d6/p2/e2a0/` 由 `.gitignore` 忽略 |

## 文件与工作树

本任务代码新增于 `scripts/research_d6/p2/e2a/`，测试位于 `tests/research_d6/test_e2a0_canonical_preprocess.py`，报告位于本目录。输出与 tokens 保存在 Git 忽略目录。没有提交或推送。

仓库中另有任务开始前已存在的未跟踪 handoff DOCX、`docs/research_triage/h1/` 和 `research_pilot/`；它们不属于本次变更，未修改。当前工作树还包含本次新增 E2A0 文档/代码/测试。

**建议的唯一下一步：** 由研究负责人审核并单独批准有限的 102 张 source-side 特征缓存生成范围；未获批前停止在 E2A0，不执行 E2-B。
