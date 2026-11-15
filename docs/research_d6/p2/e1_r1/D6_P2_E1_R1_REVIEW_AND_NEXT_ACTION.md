# D6-P2-E1-R1 审阅结论与下一步

## 最终结论：`PARTIAL`

本任务按授权完成了新 D-fit 数值初态和 W/X 匹配 headroom 运行，但**没有完成全部六条轨迹与全部数值交叉核验**：X、γ=1e-3 在反馈事件 8 因 `FTOL_ONLY_NOT_CONVERGED` 停止；Newton-CG 只对 15 个可用的 16 个点执行，且 11 个 score 比较未达到登记的 `1e-5` 容差。因此不得把本轮报告为 full PASS，也不得据此声称 Proximal 优于或劣于 gamma=0。

## 已完成的工作

- 以 P1-B 历史参数作为数值初猜，重新求解同一个 `λ=1e-4` D-fit 目标；新梯度 `4.83e-9`，有限差分误差 `1.22e-10`，通过初态门槛。
- 从完全相同的新状态运行 gamma=0、1e-4、1e-3 的 X/W 轨迹；完整成功 5/6 条。所有接受的 L-BFGS-B 解满足梯度门槛。失败配置按协议停止，未放宽参数或重试。
- 完成 23 个 checkpoint objective/gradient 检查；记录 15 次 Newton-CG 对照的目标、梯度、score 差。
- P1-A 输入快照前后相同；原 P1-B 输出 15/15 哈希通过；G0 和 P1-A/P1-B 保护指纹不变。旧 E1 BLOCKED 输出保留在原目录，新 R1 输出独立写入。
- CPU-only；没有 DINO/GPU、E2/MIL、新图像、mask、D-conf 或 sealed/HOLD 访问。没有 Git commit/push。

## 科学解释边界

X 主协议中完整的 γ=1e-4 轨迹与 gamma=0 checkpoint-12 BA 相同，paired AP 较低；X γ=1e-3 未完成。W 中 γ=1e-4 末端主要指标与 gamma=0 相同，γ=1e-3 BA 较低。Newton-CG 与 L-BFGS-B 的 objective 差都在 `1.49e-9` 以内，但 score 对照有 11/15 未达阈值。此结果属于数值核验限制，原因尚未证实。

结果来自已曝光的 24 个操作性 sample-key 组和 21 个二元 future views；不构成未见产品、独立物理对象或 D-conf 证据。旧 P1-B gamma=0 只作 `HISTORICAL_REFERENCE_ONLY`。G0 文件保持原样；R1 的 gamma=0 重跑是负责人对本 E1 任务的明确专门批准。

## 唯一建议

暂停 E2/MIL 与任何参数选择。由研究负责人决定是否另行登记一个**数值方法复核任务**，专门处理 X/γ=1e-3 的严格 L-BFGS-B 停止及 Newton-CG score 差异；在新批准前不重试、不改 solver/容差，也不把当前 incomplete 结果推进成模型机制结论。

## 文件与版本状态

本次新建四份受控审查文件于 `docs/research_d6/p2/e1_r1/`。R1 执行产物在 Git-ignored `outputs/research_d6/p2/e1/run_20261115_r1_01/`，包括逐样本预测/标签 join、模型状态、事件日志与数值核验。实现更新位于 `scripts/research_d6/p2/e1/run_e1.py` 和 `proximal_logistic.py`；14 项定向测试位于 `tests/research_d6/test_p2_e1.py`，全部通过。测试文件 SHA-256 为 `29c898f7ab8b3396d0c0fd731806ff061fa50b42343379b36e5d80bb04c8b09f`；执行时源代码 SHA-256 记录在 [初态核验](D6_P2_E1_R1_NUMERICAL_INITIALIZATION.md)。

当前分支仍是 `stage3-eg-famar-execution`，执行 HEAD 为 `4c4319397de9496526248214a0ebe9e678f5ede7`，G0 批准提交是祖先。没有提交 Git。原有未跟踪 researcher DOCX、`docs/research_triage/h1/` 和 `research_pilot/` 不属于本任务修改。
