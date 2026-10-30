# N0 研究投入决策登记表

**审计 ID：** LASTPAPER-N0-20261031  
**决策状态：** 仅为 Codex 的暂定建议；等待人类审阅。  
**本登记表不授权任何实验或下一阶段工作。**

## EISCA

| 字段 | 评估 |
|---|---|
| Novelty | 对“从 B-only feedback 推断操作性持久 memory address 的校准不确定性 posterior”这一窄问题：**UNKNOWN**。对“弱图像标签转 patch responsibility 并在部署期编辑 memory”宽泛主张：**FAIL**；DPB-DF、MIL/WeakREST、Binary HITL memory correction 已覆盖这些部分。 |
| Fundamental Scientific Gap | **UNKNOWN（候选 gap）**。B-only feedback + 冻结 DINO 状态是否能识别有用地址后验尚未解决。单个正 bag label 在无结构假设/重复证据时不能指出哪一 patch/address 负责。 |
| Strong Simple Baseline Threat | **高。** DPB-DF 局部密度 hard filter + class bank；冻结 DINO top-K/score-softmax；在线类别 prototype；MIL/top-K；Binary PatchCore bank correction。任务说明的 E4 History Credit 比 Binary-only 只高 0.0020。 |
| Main Identifiability Risk | oracle 已知来源结果证明正确目标地址有价值，却没有证明 B-only estimator 能推断该地址。Type/Mask/source 是更丰富监督，不可悄悄算成 binary feedback。 |
| Data / Information Requirements | 当前冻结 features 与 external state；预测后才揭示 queried B label；仅在授权时 Defect 后付费 Type+Mask；校准/不可部署上限使用 evaluator-only masks/source labels；未来 never-updated sentinels；产品/fold ID 及完整 query/write/cost 日志。 |
| Proposed Minimal Falsifier | **S1：** 固定 query IDs、总写入质量、容量、support、threshold，比较自动 B-only responsibility 与 no-update、broadcast、shuffle、top-K、prototype、MIL、DPB-DF-style 基线的未来 held-out utility。oracle responsibility 仅作 evaluator-only 上限。若 oracle 对简单基线无额外效用、B-only 不胜简单基线，或收益需 rich/oracle label，则失败。 |
| Estimated Implementation Difficulty | **中–高**。主要成本在无泄漏配对状态 replay、责任校准、写入质量/容量/成本匹配及产品留出评估，而不是初版 estimator 代码。 |
| Scientific Risk | **高。** 直接重叠多；标签未必能识别地址责任；历史自动 credit 增益小；收益可能只体现在定位而非异常决策。 |
| Recommendation | **HOLD** |
| Human Approval | **PENDING** |

### EISCA 决策备注

- 用户提供的 five-branch 分数是 oracle-address 结果，不是 EISCA 自动推断证据。
- 在相同 binary feedback 下，与 DPB-DF-style filter 和标准 MIL/top-K/prototype 比较前，不应宣称 manuscript 新颖性。
- 立即停止条件：oracle 对最强简单基线无 headroom；收益仅在 rich-label/oracle 出现；B-only address signal 不校准；future sentinel 无提升；或 unknown-defect recall 明显受损。

## SCAEA

| 字段 | 评估 |
|---|---|
| Novelty | 对“查询前预测耦合视觉异常状态中标签条件未来效用”：**UNKNOWN**。通用决策感知 value-of-information 公式：**FAIL**，decision-theoretic AL 和 targeted Bayesian AL 已覆盖。 |
| Fundamental Scientific Gap | **UNKNOWN（候选 gap）**。控制 query 时间、label prevalence、初始状态和选择效应后，N/D feedback 是否有稳定且可从查询前观测预测的不同未来效用，尚未验证。 |
| Strong Simple Baseline Threat | **很高。** Cost-sensitive decision-theoretic AL / one-step Bayesian VOI；random audit；entropy/margin；threshold reliability 与 memory coverage；旧 BEA/simple AIF。HITL memory correction 报告 random 与 uncertainty query 在其研究中终点无显著差异。 |
| Main Identifiability Risk | outcome-specific future utility 可能无法在查询前预测；query selection 决定哪些标签可见；旧 BA 增益因指标而异；时点/标签比例/初态或可解释观测差异；任务说明称 E02-R6 的部分 flow order 使用 offline label。 |
| Data / Information Requirements | Pre-query (x,B_t)、状态快照、精确事件顺序、query action 概率、选中后可见 B 标签、授权状态转移、实测标注成本、只供 evaluator 使用的完整 future sentinel 标签，以及相关区域非零抽样概率的随机审计。 |
| Proposed Minimal Falsifier | **Q1：** 冻结状态/候选；在同一未来流上将每种实际 Binary outcome 的授权状态转移与 skip 配对；控制 label prevalence、query timing、initial state，检验标签条件未来风险差是否稳定、可在查询前预测。与 random、entropy/margin、coverage、threshold rules、BEA/simple AIF、decision-theoretic VOI 按标注分钟匹配比较。 |
| Estimated Implementation Difficulty | source-side 审计为**中等**；可信的全流 selective-label 校正与多产品确认实验为**高**。 |
| Scientific Risk | **高。** 通用 VOI 数学新意有限；控制匹配后增益可能消失；selection bias 与指标冲突会误导结论。 |
| Recommendation | **HOLD** |
| Human Approval | **PENDING** |

### SCAEA 决策备注

- 第一个 falsifier 不必上完整 AIF/EFE。先用明确 cost-sensitive VOI 对照。
- 立即停止条件：标签条件未来价值不能在 query 前预测；同实际标注分钟下 random 或简单 coverage 达到相同效果；匹配时点/类别比例/初态后增益消失；或 EFE 与 VOI 排序一致且无额外效用。
- EFE 独立主张需要明确 latent-state/transition 不确定性或多步 policy value，并通过消融显示 epistemic/pragmatic 组件有可测增益。

## 独立性与共同投入

EISCA 改变**写入地址/更新规则**；SCAEA 改变**查询哪些样本/动作策略**。若两者分别冻结另一机制并通过各自 falsifier，可拆成两篇研究。若后续有充分理由做 factorial study，应使用共同 stream 和成本匹配估计交互，不能用组合增益掩盖单线失败。

**当前投入决定：** 文献审计完成，来源不完整处已注明；两条线均 `HOLD`。这不是科学 GO，也不授权 S1/Q1，任何下一阶段门槛需人类审核。
