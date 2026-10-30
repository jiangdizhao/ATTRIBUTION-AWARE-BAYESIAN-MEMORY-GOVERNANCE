# N0-B — SCAEA 竞争性新颖性审计

**审计 ID：** LASTPAPER-N0-20261031  
**文献截点：** 2026-10-31（Australia/Sydney）  
**范围：** 只审计文献/理论；未运行 query-policy 实验或模型。  
**证据标记：** `VERIFIED`、`INFERRED`、`HYPOTHESIS`、`UNKNOWN`，定义见 N0-A。

## 执行结论（Executive Verdict）

**建议：HOLD。** 提出的价值函数

\[
V(x,B)=\sum_{y\in\{N,D\}}P(y\mid x,B)\,\Delta U_y(x,B)-\mathrm{QueryCost}(x)
\]

是状态条件化、对结果标签求期望的 value-of-information 目标。若 \(\Delta U_y\) 表示标签 \(y\) 引发状态更新后未来决策风险的下降，这就是已有 decision-theoretic active learning / Bayesian experimental design 的直接形式。至少自 2007 年起，决策论主动学习已明确将预期风险下降与标签成本比较；2024 年 targeted Bayesian active learning 为下游最优决策分布上的信息增益选 query；budgeted online active learning 已研究有限 horizon 与标签预算下的顺序查询。

可能的差距是一个**应用层的状态转移问题**：不同 Binary 标签会改变若干耦合部署状态（memory、correction、threshold sidecar），系统需在查询前估计 B-only feedback 对未来效用的价值，并遵循有预算的 prequential anomaly stream。检查的文献不能证明这个精确部署问题已经解决；也不能证明当前状态能稳定预测 N 与 D 的未来效用差异。两者均为 `UNKNOWN`，待 Q1 检验。

**宽泛主张“状态依赖 query value 是新理论”：FAIL。**  
**窄版“耦合状态转移与多模块未来效用”：UNKNOWN。**  
**Active Inference / EFE 是否必要：没有证据支持必要。** 先实现 cost-sensitive Bayesian VOI 和简单 state-aware 规则。只有当 EFE 建模了额外的 latent state/多步 belief-space 能力且胜过匹配的 VOI 对照，才应使用。

任务说明中的历史 BEA 结果（BA 约 0.7838，对比 Random 约 0.7095，但 Random AUROC 较高）不能支持指标无关的优势。E02-R6 使用 offline ground truth 构造了部分流顺序，因此不能验证可部署的 pre-query policy。HITL PatchCore correction 预印本则直接比较 error pool 随机查询与 threshold-distance uncertainty sampling，在 15 个 MVTec 类别均未报告显著 held-out 终点差异。上述项目历史结论来自任务说明，未在本审计中重跑。

## 科学问题比较（Scientific Problem Comparison）

### 查询动作与标签可见性比较（Query Action and Label Visibility Comparison）

| 系统/论文 | 反馈前选择什么 | 查询后可见什么 | 之后改变什么 | 与 SCAEA 重叠 |
|---|---|---|---|---|
| SCAEA 目标 | 只对当前图像 query 或 skip，使用当前可观测 (x,B) | B：仅 Binary Normal/Defect。R 是 Defect 后单独付费升级 Type+Mask；Normal 不产生 mask。 | 按协议只更新外部状态：Normal Memory、signed correction、threshold sidecar、query state。时序为 predict→query→reveal→update。 | 目标设定 |
| DPB-DF | 检查代码中是按增量类别处理带标签样本，不是 B-only 主动查询器 | 每图/文件夹标签含类别 ID，包含写入 bank 使用的 defect class | 保留 patch 写入类别 bank；调整 class 权重 | 标签更多、动作协议不同；patch filter/bank 是邻近机制 |
| WeakREST | 使用弱标签做离线训练 | 按条件可见 image defect status、box、block 或 pixel label | 训练 Swin/residual 模型，不是部署期 memory state | 弱标签定位，不是 query value |
| HITL Memory Correction | 从当前 error pool 选一条：passive random 或离 threshold 最近 | 一次 Binary Normal/Defect 判断 | FP：插入 normal patches；FN：hard memory surgery | 直接的 binary 部署反馈/持久 memory loop；未估计多模块价值 |
| Budgeted Online Active Learning（AAAI 2026） | 有限 horizon 与预算下从数据流逐步选择 | 查询目标标签（其应用是有成本的葡萄冷硬度测量） | 更新预测器；用 expert advice 和 episodic priors | 说明 online + 有限预算已研究；但不是视觉 anomaly memory 或 Binary 标签状态转移 |

Rich Type/Mask 不属于 Binary 的“隐含标签”。查询是一个动作，系统不能选择真值为 N 还是 D；只能预测两类结果的价值分布，并在当前预测后揭示被授权标签。不得把更新前已经输出的当前预测算成查询标签带来的提升。

### 最接近的竞争工作（Closest Competitive Works）

#### 1. Selective Supervision: Guiding Supervised Learning with Decision-Theoretic Active Learning

IJCAI 2007，[官方 proceedings 页面](https://ijcai.org/proceedings/2007/141)，[PDF](https://www.ijcai.org/Proceedings/07/Papers/141.pdf)。Section 3（官方 HTML 约 63–90 行）明确把标签的 value of information 定义为总预期分类风险下降减去获取标签的成本。方法考虑类别相关错误成本、标签成本和未来使用量，并比较 random、uncertainty、posterior-influence acquisition。它与 SCAEA 公式存在**最直接数学重叠**。区别是原文主要为 myopic 单次分类，而 SCAEA 目标是多个受限部署状态更新后的视觉异常检测未来效用；应用差异可以构成 gap，但不能创造新的 VOI 原理。

#### 2. Targeted Active Learning for Bayesian Decision-Making

Louis Filstroff 等，*Transactions on Machine Learning Research* 2024(6)，[最终发表记录](https://researchportal.helsinki.fi/en/publications/targeted-active-learning-for-bayesian-decision-making/)，[最终 OpenReview PDF](https://openreview.net/pdf?id=KxPjuiMgmm)。Sections 2–3 与 Eq. (8) 按下游最优决策 posterior 上的预期信息增益选 query；论文指出，普通参数不确定性 query 未必改变决策，并直接建模这一点。因此“把 query value 关联到未来决策结果”不能作为宽泛新颖性。其未涉及 B-only anomaly feedback 下的持久 memory/threshold 状态转移。

#### 3. Budgeted Online Active Learning with Expert Advice and Episodic Priors

Kristen Goebel 等，AAAI 2026，*Proceedings of the AAAI Conference on Artificial Intelligence* 40(45):38496–38504，DOI [10.1609/aaai.v40i45.41191](https://doi.org/10.1609/aaai.v40i45.41191)，[AAAI 官方记录](https://ojs.aaai.org/index.php/AAAI/article/view/41191)。arXiv 全文 Sections 3–6 与 Algorithm 1 将 stream query 表述为有限 horizon/严格预算下的顺序选择，并在 acquisitions 之间更新；使用预先存在的专家预测器与无标签历史 stream priors。它与 SCAEA 的 online budget、当前模型状态和历史条件 acquisition 重叠。其应用是农业连续值结果，未做 visual anomaly correction；但“online + budget + 当前/历史状态”不足以构成新颖性。

#### 4. Training-Free Human-in-the-Loop Anomaly Detection via Memory Bank Correction

Ayusha Abbas、Saram Abbas、Kabita Adhikari，**arXiv 预印本** arXiv:2608.17775（2026-08-18），[全文](https://arxiv.org/html/2608.17775)。Sections III-B–III-C、Algorithm 1：每个已查询错误图像只得到一个 Binary label，再更改 PatchCore memory。query 选择比较 uniform random 与 threshold 最近的 uncertainty sampling \(\arg\min_x |s(x)-\tau|\)。Sections IV-E/IV-G 报告：每类 20 splits 下，15 类均无显著 active/passive held-out 终点差异；有时 query 影响达到收益的速度，不改变最终端点。这是当前 anomaly-specific 的直接威胁，说明复杂 controller 可能不必要。它未测试 SCAEA 多模块 state-value predictor 或 BEA。

#### 辅助理论：EFE 与 selective labels

Expected free energy 通常分解为 epistemic information gain 与 preferred-outcome/pragmatic value。2016 年 “Active inference and learning”（DOI [10.1016/j.neubiorev.2016.06.022](https://doi.org/10.1016/j.neubiorev.2016.06.022)，Section 2.3、Eqs. 5–6）把 epistemic value 解释为 mutual information/expected information gain，把 extrinsic value 联系到 expected utility。Sajid 等“Active inference, Bayesian optimal design, and expected utility”([arXiv:2110.04074](https://arxiv.org/abs/2110.04074)) 说明去掉偏好后 EFE 的极限对应信息增益设计/贝叶斯决策理论。这些是理论关系，不代表 EFE 在本异常检测系统上有增益。

Selective labels 是独立评估威胁。Lakkaraju 等，KDD 2017，[DOI 10.1145/3097983.3098066](https://doi.org/10.1145/3097983.3098066)，讨论已有决策决定哪些标签被观察。Robinson 等，*Data Mining and Knowledge Discovery* 38 (2024)，[DOI 10.1007/s10618-024-01026-x](https://doi.org/10.1007/s10618-024-01026-x)，研究 active learning 中 non-response，展示顺序 query/response 机制会改变观测数据分布与之后的 query 序列。SCAEA 只在查询时见标签；只看 queried cohort 的指标不能替代生产全流，除非有随机审计或合理 propensity 校正。

## 详细新颖性矩阵（Detailed Novelty Matrix）

| 维度 | SCAEA 目标 | Decision-Theoretic AL | Targeted Bayesian AL | Budgeted Online AL | HITL Memory Correction |
|---|---|---|---|---|---|
| 科学问题 | 哪个当前图像的标签经 mutable state 更新改善未来部署决策？ | 哪个标签带来的分类风险下降超过标注成本？ | 哪个 query 降低下游最优决策 posterior 不确定性？ | 有限 stream/horizon/预算下何时查询？ | 应检查哪个 anomaly error 来改善部署 memory？ |
| 状态依赖 | Memory、signed correction、threshold、query state | 已标记数据/分类器 posterior/预期使用 | 下游决策 posterior 与模型 | Predictor/expert weights 与历史 episode | 当前 score、memory、error pool |
| 动作 | 当前样本 query/skip，另有付费 R | query 一个未标记样本 | 选样本（多决策场景也包含决策对） | 按预算顺序选择 | 从当前错误池选一个 |
| 标签可见性 | Binary N/D；Defect 后可付费 Type+Mask | 查询类别标签 | 查询目标观测 | 应用相关数值标签 | Binary N/D |
| Query utility | 标签触发状态变化后的预测未来效用 | 明确 risk reduction − label cost | 下游最优决策分布上的信息增益 | budgeted acquisition score、expert/episodic priors | 比较随机与 uncertainty；在 held-out 上评 correction |
| 多更新模块 | 目标包含 | N | N | Predictor/expert state | 主要是一个 PatchCore memory |
| 多步有限 horizon | 需要研究 | 基线为 myopic，可扩 lookahead | 顺序 acquisition | 明确包含 | 顺序 correction round |
| 选择偏差处理 | 必须设计 | VOI 本身不解决 | 非该文中心贡献 | 依各自评估协议 | held-out；标签由模拟生成，query 仍然自适应 |
| 与此处相关的实验证据 | 尚未验证 | 已有理论/方法基线 | 决策任务有实证 | 农业模拟与真实 vineyard 数据 | 本研究的 active/passive endpoint 无显著差异 |
| 新颖性影响 | 精确视觉状态转移应用仍未知 | 通用 VOI 已覆盖 | decision-aware query 已覆盖 | online 有限预算已覆盖 | 复杂 query 可能不值得，尤其全局 memory update |

## BED / AIF / VOI 理论重叠（Theoretical Overlap）

### 已属标准理论的部分

若 (T(B,x,y)) 是标签 (y) 后获准的状态转移，而 (Delta U_y) 是相同未来流在该状态下的预期决策风险下降，则

\[
\mathrm{VOI}(x,B)=\sum_y P(y\mid x,B)\,\Delta U_y(x,B)-c_q(x,B)
\]

就是 expected net value of sample information，也是 decision-theoretic active learning 的直接形式。把 (B) 改成更丰富的“state”不自动改变决策理论原则。标准 Bayesian decision formulation 可将 memory、correction、threshold 组合为状态；Bayes-adaptive MDP/POMDP 可表示顺序状态依赖动作与预算。

若要比较 query 与 skip，因 prequential 顺序为 score→predict→query→reveal→update，目标应是未来样本。可定义

\[
\Delta_y(x,B)=\mathbb{E}_{\mathcal S_{t+1:T}}\left[
R_{t+1:T}(B^{\mathrm{skip}};\mathcal S)-R_{t+1:T}(T(B,x,y);\mathcal S)
\right],
\]

于是 (V=\sum_yP(y\mid x,B)\Delta_y-c_q)。query/no-query 两臂必须用同一 candidate、future stream、support、threshold 初始化和评估集。当前图像的预测在标签到来前已完成，不得把它计为改善。

### 有证据时可能具有新意的部分

若不同标签会改变异质状态模块、状态间存在交互、写入有延迟或 collateral cost，那么**estimand 与 transition model** 可能形成有用研究。论文需说明：(i) 每种可见标签的状态转移；(ii) 未来部署效用和查询成本；(iii) 如何无泄漏估计 (P(Y\mid x,B)) 与 ΔU；(iv) 这些量为何比一步风险/coverage heuristic 和标准 VOI 多提供信息。若只是把同一 expected risk reduction 换成新符号，不能算新理论。

### EFE 是否有独立价值？

目前没有证据。EFE 把 expected information gain 与偏好/成本结合，也可描述 latent states/parameters 的 belief 和多步 policy sequence。如果 epistemic term 只是减少不确定性、pragmatic term 仍是 VOI 的未来风险目标，它只是同一 query ranking 的另一种推导或参数化。只有当 SCAEA state model 真包含潜在 dynamics/transition structure 的不确定性，或 VOI comparator 未表达的多步 belief-space policy，且 ablation 证明该部分改善 matched-budget future utility 时，EFE 才有独立价值。当前 SCAEA 定义尚未证明需要这类额外模型。

## 最强简单基线（Strongest Simple Baselines）

1. 少量预注册随机审计（random query），记录纳入概率，按实际标注分钟和预算比较。
2. 当前图像 N/D score 的 entropy/margin、anomaly novelty、threshold-distance 规则。
3. 简单 coverage/reliability priority：按 memory coverage、threshold calibration error 或其固定组合调整。
4. 精确复现旧 BEA 与 simple AIF；不预设谁胜。
5. 有显式标签后状态 replay 的 cost-sensitive decision-theoretic AL / one-step Bayesian VOI。这是最强理论基线。
6. 标签比例、query 时点与初始状态控制：匹配 N/D mix、score/threshold 距离、query schedule。它们用于解释收益，不是同一算法。
7. 只在 R tier 授权范围内比较 random rich escalation 与启发式升级，并计入新增 annotation minutes。

HITL PatchCore correction 给出了高度相关的实践对照：从当前错误池随机挑选，终点与 uncertainty sampling 在 15 类均无显著差异。项目历史中 Random BA 低于 BEA，但 AUROC 更高。两者说明应报告多个 operating metrics 与每分钟未来效用，而不是按单一指标选政策。

## 选择性标签偏差（Selective Label Bias）

SCAEA 选择哪些流样本可见标签。因此反馈分布是 (P(X,Y\mid Q=1,B))，未必等于生产分布 (P(X,Y\mid B))。状态更新又会改变后续 score 和 query probability，形成反馈环：欠查询区域模型弱、观测少、之后仍欠查询。只在 queried 样本上评估会夸大或反转结果。

Q1 及后续 controller 对照必须：

- 记录每个合格样本、pre-query state、query action、action probability/propensity、可见反馈层级和实际标注时间；
- 保留有非零纳入概率的随机审计，覆盖相关 memory-coverage/threshold 区域，避免违反 positivity；
- evaluator 标签不得进入 controller/update code，只有预先选择的查询可以揭示标签；
- 在固定 never-updated sentinel 上估计全流风险，或使用满足 positivity 的 inverse-propensity/doubly robust 估计并报告方差；
- 用同一 query items 的 replay 隔离 query-policy 与 update-rule，再做第二个 matched-budget 闭环实验；
- 将拒答、未返回标签和反馈成本作为结果报告，不可静默丢弃。

任务说明中的项目风险仍关键：BEA vs Random 指标不一致；E02-R4/R6 查询时点和标签构成变化；E02-R6 部分使用 offline labels 排流；初始状态质量可能支配结论。它们均未隔离 pre-query state-value predictor 的因果贡献。

## 对抗性反论证（Adversarial Counterarguments）

1. **“这是带 state variable 的 VOI。”** 如果 ΔU 定义为下游风险下降，该公式就是预期风险下降减成本；state conditioning 的命名不创造新理论。
2. **“有用标签不可在查询前预测。”** 若 N/D 条件更新价值不同，却不能由可见 (x,B) 预测，期望价值会退化为 prevalence、随机或覆盖规则，复杂 controller 无法识别目标。
3. **“收益来自时点/类别比例。”** 旧策略查询时间或 Normal/Defect 比例不同；匹配后若优势消失，就没有 SCAEA controller effect。
4. **“controller 利用了评估泄漏。”** offline labels、future sentinel、mask、query IDs 不得影响 policy；任务说明中的 E02-R6 排流方法不是可部署证据。
5. **“AIF 只是重命名不确定性+utility。”** 若去掉 EFE epistemic term 或换成 VOI 后动作与结果不变，EFE 不构成独立贡献。
6. **“随机审计已足够解决 blind spot。”** 若少量随机审计在同等分钟下与复杂 policy 同效，则复杂度不合理。

## 尚存科学差距（Remaining Scientific Gap）

**候选、尚未验证：** pre-query 状态 (B_t) 是否含有足够可观察信息，能预测当前样本每种可能 Binary feedback 的**因果未来效用**？控制标签比例、查询时间、初始状态和选择机制后，这种预测能否优于标准 decision-aware VOI 与简单 random/uncertainty/coverage 基线，并提升每分钟部署效用？

它不同于“label value 是否依赖状态”（状态式学习通常如此），也不同于“active learning 是否胜随机”（本任务中可能不成立）。

## Q1 状态学习价值审计方案（State Learning-Value Audit）

**先回答的问题：** 在冻结 pre-query state 与固定候选 (x) 后，两种 Binary 结果 N/D 经各自授权状态更新，会否在相同未来流上造成稳定且不同的 held-out 决策效用？这个差异能否由查询前可观测量预测，而不仅由标签频率、score/threshold 距离、memory coverage、query 时间或初始状态解释？

最小审计：

1. 冻结 stream、state snapshots、support、feature/scoring version、threshold、事件顺序、预算与 never-updated sentinel；在各独立 source-CV fold 保留预注册 random-audit rate，并记录 propensity。
2. 每个 candidate 先只用 (x,B_t) 生成 controller 预测。再由离线 evaluator clone 对实际 N/D 标签分支，执行该标签授权更新，并在同一配对 future stream 上评分。oracle 分支只供评估，不可用于 policy selection。
3. 对比 no-query 分支估计标签条件因果效用；当前图像预测固定，只评估后续样本。各政策臂匹配 query positions、标签构成与实际标注分钟。
4. Primary estimand：预注册的 cost-sensitive future risk reduction / annotation minute；若无法正当化错误成本，可用固定 threshold BA。Secondary：AUROC、FPR/FNR、校准风险、coverage、标签比例、按状态模块计的 writes。
5. 对照 random、entropy/margin、threshold reliability、memory coverage、label-prevalence、旧 BEA/simple AIF 和 one-step Bayesian VOI。utility predictor cross-fit，产品留出集保持封存。
6. 若 outcome-conditional delta 跨 fold 不稳定、查询前不可预测、控制比例/时间/初态后消失，或随机/简单规则在预注册有效界内等同最佳 controller，则否定/收窄该假设。

这是最小证伪设计建议，不是实现 Q1 的授权。

## EFE 是否具有独立价值

**当前答案：未证明，且 Q1 不需要完整 EFE。** 先实现/估计最简单 decision-aware VOI。只有在明确的 generative model 描述了不确定 state transition 或 latent operating regime，且存在一步 risk reduction 不足以表达的多步动作 horizon 时，再比较 EFE。

若要主张 EFE 贡献，需定义 observation/state/transition/preference model；与 VOI 使用同一成本尺度和 query budget；分别消融 epistemic/pragmatic 项；校准标签结果概率；匹配计算与 latency；并在未来 held-out utility 上证明统计支持的收益。若 EFE 与 VOI 代数或实证上排序等价，应删除 EFE 新颖性主张。

## SCAEA 能否与 EISCA 独立成篇？

**可以作为独立实证问题，前提是隔离。** EISCA 改变“证据写到哪里/如何写”，query events 固定；SCAEA 改变“查询哪些样本”，update mechanism 固定。两者 estimand 不同，应先各自研究。之后可用 factorial design 估计交互。若一个实验同时改 query policy 和 memory write，就无法分辨因果贡献。

独立论文需要冻结 SCAEA 的写规则；EISCA 使用共同固定 query trace；SCAEA 固定最有依据的写规则并匹配 realized query/cost budget。若 query-policy 效应被 VOI/random audit 消除，或 memory-write 收益依赖 oracle supervision，不应把两条线捆绑互相补救。

## Pattern Recognition / IJCV 贡献要求（PR / IJCV Contribution Requirements）

**Pattern Recognition 最低要求：** 定义标签可见性与 state transition；展示 pre-query calibrated predictor 预测未来 outcome-specific utility；比较 decision-theoretic VOI、budgeted online AL、random/entropy/margin/novelty/coverage、旧 BEA/simple AIF；匹配 query time、标签比例和标注分钟；用全流 future sentinel 或可辩护的 selective-label 校正；报告 operating metrics 与运行开销；证明 held-out 产品增益。新 controller 名或事后 query ordering 不够。

**IJCV 额外要求：** 给出一般 state/action/observation formulation；刻画 outcome-conditioned utility 的 pre-query 可识别条件及预测校准；处理 selective-label positivity 与状态诱发反馈偏差；跨产品/流/成本设置复现；优于 Bayes-adaptive/decision-aware baseline。除非有明确独立理论/实证能力，否则 Active Inference 可选。

## 局限与未核实主张（Limitations and Unverified Claims）

- AAAI proceedings 页面核对标题、作者、DOI、出版状态；方法细节来自关联 arXiv v1 全文。arXiv PDF venue 字段含占位符，未将其当最终书目信息。
- HITL Memory Correction 是预印本，反馈由模拟生成；query 对比仍是相关直接实验。
- Decision-theoretic AL 与 TMLR targeted AL 不是视觉 anomaly system；它们威胁的是理论新颖性，不必然解决精确部署应用。
- BEA/E02 项目指标和泄漏说明来自用户任务说明，未新跑实验或检查原始结果。
- 来源集是定向检索；未发现精确组合工作不构成不存在的证明。

## 暂定 GO / HOLD / NO-GO 建议（Provisional Recommendation）

**HOLD** SCAEA 于 N0。先做 Q1，再考虑增加 controller 复杂度。若标签条件未来效用无法查询前预测；或同分钟下随机/简单 coverage 与最佳 policy 等效；或优势经匹配查询时点、类别比例和初态后消失；或 EFE 不超越 VOI，则停止复杂 controller 主线。不要预设或强求 AIF/EFE。N0 文献结论本身不授权 Q1，也不等于科学 GO。

## 主要来源（Key Sources）

- [Selective Supervision, IJCAI 2007](https://ijcai.org/proceedings/2007/141)
- [Targeted Active Learning for Bayesian Decision-Making, TMLR 2024](https://researchportal.helsinki.fi/en/publications/targeted-active-learning-for-bayesian-decision-making/)
- [Budgeted Online Active Learning, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/41191)
- [HITL Memory Bank Correction, arXiv:2608.17775](https://arxiv.org/abs/2608.17775)
- [Active inference and learning, 2016](https://doi.org/10.1016/j.neubiorev.2016.06.022)
- [Active learning with biased non-response, 2024](https://doi.org/10.1007/s10618-024-01026-x)
