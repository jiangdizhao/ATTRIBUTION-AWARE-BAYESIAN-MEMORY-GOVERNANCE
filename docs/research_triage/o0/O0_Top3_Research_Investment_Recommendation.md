# O0 — Top Research Investment Recommendation

**审计 ID：** LASTPAPER-O0-20261031  
**决定状态：** 两条条件性 shortlist；不作科学 GO，不授权后续研究阶段或实验。  
**推荐原则：** 若证据不支持唯一 winner，就明确留下未决，而不是给架构排奖。

## 有条件保留的方向

### 1. O01：选择性/噪声反馈下有风险边界的持久更新

**科学问题。** 工业视觉异常检测在正常外观变化时可以利用部署后反馈适配，但未查询样本不能自动视为正常；当反馈被选择性返回、延迟或误标时，何种可观测条件足以支持安全的永久记忆更新，何种情况必须隔离或撤销？

**为何经过反向检索后仍可继续审计。** HITL Memory Correction、SH-HIL-PCF 已覆盖人类反馈更新和漂移适配，OWAD/RoDA 已覆盖开放世界 normality shift/域适配，因此它们关闭了宽泛 novelty。剩下的可检验 gap 是持久更新风险与选择性反馈之间的明确识别/评估条件。此差异目前只是 **INFERRED/UNKNOWN**，尚未证明超出标准 online selective learning、IPS、随机审计或 gate。

**最危险的竞争者。** [HITL Memory Correction](https://arxiv.org/abs/2608.17775)：直接使用 Binary 人类反馈更新 PatchCore bank；其工作报告主动和被动查询终点无显著差异，并且缺陷记忆扩展失败。另有 [SH-HIL-PCF](https://doi.org/10.1109/WcCST67302.2026.11496141) 与 [RoDA](https://arxiv.org/abs/2503.14910)。

**可能的理论创新。** 定义 query propensity、反馈噪声、状态转移与未来 sentinel 风险，给出“不可识别条件”和有假设限制的估计/风险界；先查清普通 selective-label/off-policy estimation 是否已给出该结论。不要把 AIF/EFE、NIG 或“治理记忆”本身当理论结果。

**第一项研究设计。** 设计一个离线、no-leak 的最小反事实流：保持 support、阈值、初始 memory 和 future sentinel 固定；在同一候选图像流上施加受控正常外观 shift、新 defect、query-dependent non-response、有限反馈延迟和误标；只在反馈发生后更新；比较不更新、固定 novelty gate、滑窗/greedy bank、SH-HIL 风格 drift trigger、随机审计+IPS、可撤销状态。主指标预先定义为 future-sentinel 的固定 operating-point risk 与误吸收率，另报未知 defect recall 和恢复时间。当前任务只提出该设计，不执行。

**立即终止条件。** 以下任一成立即停止复杂机制投资：simple fixed gate/greedy memory 在同等误吸收率和实际查询分钟下达到同样效用；随机 audit + 标准估计能解释全部收益；反馈噪声/响应机制不可观测，positivity 无法满足；提升只来自多看数据、阈值校准或当前样本；未来风险界需要现实中无法验证的噪声/漂移假设。

**可复用项目资产。** 可复用的研究边界文档、prequential protocol 设计、Real-IAD 官方公开元数据/特征缓存想法、既有 scorer/memory 对照规划。Stage-3 handoff 明确 G0-G4 尚未运行或验证；不得把 proposed mechanism、历史研究数字或本地环境存在数据路径说成已验证实现。需先核实旧 v22 与当前 feature/preprocessing 一致性。

**数据/反馈是否必须新增。** MVTec AD 2 + Real-IAD 的公开 images 和 evaluator labels 足以做初步模拟 falsifier；两者都没有真实运行流中的时间、查询概率、操作员延迟/误标轨迹。若审计认为问题有 headroom，可信的人类成本/响应率验证将需要新的小型专家反馈协议。不能用合成噪声证明真实部署风险保证。

**发表潜力。** 工程：仅固定阈值/漂移触发更新。PR 方法：在 matched budget 下跨数据提升 future safety/utility，并清楚优于适配、记忆、随机审计 baselines。IJCV：还需可复用的一般问题/非平凡理论界和真实多产品验证。当前只有问题重要性，无 novelty 结论。

**四个月可行性：** 中（论文/模拟审计可做）；若需真实人类反馈和企业 shift 元数据则高。  
**短列状态：** 条件性 shortlist；置信 HYPOTHESIS/UNKNOWN。

### 2. O06：持久学习人类修订的组合/逻辑检查规则

**科学问题。** 少量人类对缺件、错数、错位或关系的纠正能否学习可修订的检查规则，并改善未见过配置的后续检查？

**为何暂留后备。** 重要工业情形，且未来效用清楚；但 MVTec LOCO 的公开规则与 PSAD、SALAD、LogiCo 等造成高基线。真正窄差异应是后续配置上经反馈修订的规则知识，而不是“加入逻辑推理”。

**最危险的竞争者。** [PSAD（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/view/28703)、[SALAD（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/papers/Fucka_SALAD_--_Semantics-Aware_Logical_Anomaly_Detection_ICCV_2025_paper.pdf)、[LogiCo（2026 预印本）](https://arxiv.org/abs/2606.28688)，以及显式组件计数/位置/关系规则。

**可能的理论创新。** 在组件观测噪声下，关系规则的可识别性、组合泛化和矛盾反馈触发的可撤销规则修订。当前仅是假设；公开 LOCO labels 不包含任意逐图人类 rule corrections。

**第一项研究设计。** 不从 LOCO test GT 蒸馏隐含规则后再把它称作在线反馈。先确定一组合法组件配置与人工可回答的规则反馈 schema；按配置留出；以固定计数/位置、PSAD/SALAD、VLM QA、nearest prototype 对照。若公开数据不含足够配置，先做数据采集方案而不是跑一个泄漏式实验。

**立即终止条件。** 对照固定 component rule 或 SALAD/LogiCo 无未来配置 headroom；要成功必须把目标配置的规则/GT 直接输入系统；真实反馈不可复现或只在五个 LOCO 类别上成立。

**可复用项目资产。** 反馈 authority/cost 文档、prequential 事件排序、future sentinel 评估理念。现有 EG-FAMAR memory architecture 不是必须资产，不能因为在仓库里而优先。

**数据/反馈是否必须新增。** LOCO 可做初始静态 sanity check，提供逻辑/结构异常与像素标签，但只有五类产品、静态样本和 category-level normality rules；它不提供逐流 rule feedback。若继续，需最小人工规则数据：多种合法产品/配置、规则例外、反馈来源及成本。

**发表潜力。** 只把 VLM rule prompt 保存下来是工程；跨配置 rule-learning 方法可能达 PR；IJCV 需要更广的组合泛化与修订理论。  
**四个月可行性：** 高，主要由专家数据决定。  
**短列状态：** 后备候选；只有拿到可用配置与人工规则数据后才值得继续；novelty UNKNOWN。

## 不进入 shortlist 的既有方向

### EISCA

**地位：** 不入 O0 shortlist；现有方向维持 HOLD。宽泛 Binary-feedback memory correction 已有直接工作，weak patch responsibility 也已由 DPB-DF、WeakREST、MIL 等覆盖。仅“B-only 历史能否识别未来有用的持久地址”仍是 UNKNOWN，需先通过 O03 的 oracle-headroom/简单基线检验才值得重开投资。

### SCAEA

**地位：** 不入 O0 shortlist；现有方向维持 HOLD。决策感知 VOI、targeted Bayesian AL、budgeted online AL 覆盖通用理论；HITL Memory Correction 的主动/被动差异不显著，质疑 query mechanism 的必要性。可保留精确应用问题 UNKNOWN，但 EFE/AIF 不是自动理论贡献。

## 研究投资比较（定性）

| 候选 | 新颖性 | 问题重要性 | 最强竞争者 | 理论空间 | 标签/数据够用度 | 最小成本 | 简单基线解释风险 | 四周负/正证据 | 四月稿件可行性 | 最大不确定性 |
|---|---|---|---|---|---|---|---|---|---|---|
| O01 安全、可撤销适配 | UNKNOWN | 高 | HITL Memory Correction / SH-HIL-PCF / RoDA / 标准 IPS | 有条件、需先查先例 | 中低；无真实 response/shift 轨迹 | 中 | 高 | 高（离线设计/小模拟可快证伪） | 中 | 是否超出普通 gate+audit |
| O06 规则反馈跨配置 | UNKNOWN | 高 | PSAD/SALAD/LogiCo + 固定 rules | 中 | 低；逐配置人工纠正规则缺失 | 高 | 高 | 中低（数据限制会很快显露） | 低–中 | 可迁移配置与人工标签来源 |
| O03 EISCA | 窄问题 UNKNOWN，宽主张 FAIL | 中高 | DPB-DF/HITL/WeakREST/MIL | 条件式可识别性 | 中（public masks evaluator-only） | 中 | 很高 | 高 | 中 | Binary responsibility 是否可识别 |
| O10 SCAEA | 通用 VOI FAIL，精确应用 UNKNOWN | 中高 | Selective Supervision/Targeted AL/HITL | 一般理论已有 | 中（行为反馈与成本模拟） | 中 | 很高 | 高 | 中 | 标签条件未来效用可否预测 |
| O02 动态反馈粒度 | 广义 FAIL | 中 | Multi-level AL/WeakREST/MultiADS | 低到中 | 中低（真实分钟与噪声缺失） | 中 | 高 | 中 | 中 | 未来 transfer 是否带来净价值 |
| O08 多视角/模态动作 | 当前动作选择已被覆盖 | 高 | ActiveInspect/CVAD | 普通 POMDP/VOI 已覆盖 | 低（没有真实动作日志/成本） | 高 | 很高 | 中 | 低 | 是否有跨对象持久学习 |
| O04 defect attribute transfer | UNKNOWN | 高 | MultiADS/DPB-DF/VLM prototypes | 中 | 低（人工属性缺失） | 高 | 高 | 低–中 | 低 | 标签语义和外观是否可迁移 |
| O05 规范 vs shift | UNKNOWN | 高 | Shift adaptation/规则配置 | 可能有识别性结果 | 很低（缺版本化政策） | 高 | 高 | 低 | 低 | 企业过程数据 |

表中是相对判断，不是数值分数或发表概率。

## 唯一建议的下一步

只推荐一个**设计/审计任务**，不推荐立即实验：

**O01-Risk-Identification-Audit（限范围文献与数据可识别性复核）**：先通读 DPB-DF 最终全文、HITL Memory Correction、SH-HIL-PCF、RoDA、OWAD，以及 selective-label/off-policy 风险估计的理论先例；再写出 O01 的完整观测变量、query propensity、噪声/延迟假设、可识别目标、至少一个不可识别反例和基于 MVTec AD 2/Real-IAD 的 no-leak 流拆分。交付为审查稿与数据字段要求，不改代码、不读数据、不做实验。只有当 O01 明确超出标准估计器且所需变量可观测，才另行请示研究设计。

## 暂定第一选择

**不指定最终 first-choice。** O01 只是下一轮审计最值得检查的对象；现有证据不足以把它定为未来论文方向。O06 仍是有条件后备。除此之外当前不要继续投入。

## 审批状态

人类审批：PENDING。  
本建议不是 EISCA/SCAEA 解冻、不是 S1/Q1/S2/Q2 授权，也不授权新实验。
