# N0-A — EISCA 竞争性新颖性审计

**审计 ID：** LASTPAPER-N0-20261031  
**文献截点：** 2026-10-31（Australia/Sydney）  
**范围：** 只审计文献与理论；未运行实验、模型或数据集，未读取图像，未修改源码。  
**证据标记：** `VERIFIED`=直接检查原始论文/代码；`INFERRED`=基于已检查来源的比较判断；`HYPOTHESIS`=待验证研究主张；`UNKNOWN`=尚无结论。

## 执行结论（Executive Verdict）

**建议：HOLD。** “稀疏图像标签可转成 patch-level responsibility，并据此持续更新异常检测记忆”这一宽泛主张已有显著重叠。最接近的工作分别覆盖了它的关键部分：DPB-DF 用正常 patch 邻域筛选候选缺陷 patch，并维护按类别划分的动态 patch bank；attention MIL 与 WeakREST 从 bag/image label 估计或学习 patch/block 分数；2026 年的 HITL Memory Correction 预印本则以二元 Normal/Defect 人类反馈直接修改 PatchCore memory。

本次检查未发现一篇论文同时包含 EISCA 的全部受限条件：冻结 DINOv2、初始仅有 B-only Binary feedback（付费 Type+Mask 单独计算）、对**持久记忆地址**而非仅 patch 异常分数给出 responsibility posterior、显式处理不确定性/拒写/隔离，以及用未来未见产品的因果效用检验更新价值。这只是候选差距，不能视作已确立的新颖性。**精确新颖性：UNKNOWN。**

可辩护的窄问题是：在明确的反馈可见性和更新授权协议下，系统能否从证据中形成校准的不确定性后验，选择性地向持久记忆地址写入，在质量、写入总量和容量匹配时，优于 hard MIL/top-K/density/prototype 基线，并降低错地址干扰？这里的 `address` 必须是操作性定义，不能未经验证就称为人类语义因子、缺陷来源、因果因子或 patch 坐标。

任务说明中两项历史结果尤其限制主张范围。Five-Branch 结果（正确地址 F1=0.6631、质量匹配 broadcast=0.4254、no-update=0.3957）使用了**已知正确地址**，说明正确路由可能有价值，不说明系统能自动推断地址。E4B 的 History Credit 仅比 Binary-only 高 0.0020（0.6402 对 0.6382），Oracle 为 0.6551。后一个设置中 History 与 Oracle 的差距仅 0.0149。以上数字来自用户任务说明，本次没有用原始结果文件重算。

## 科学问题比较（Scientific Problem Comparison）

| 维度 | EISCA 目标 | 最接近的已有工作 | 审计判断 |
|---|---|---|---|
| 科学问题 | 哪些 patch 证据应更新哪个持久地址；责任不确定时写入、隔离、保留还是拒写？ | MIL 学习实例对 bag label 的贡献；WeakREST 预测 block 异常分数；DPB-DF hard-filter 缺陷 patch；HITL correction 根据 binary feedback 编辑正常 patch bank。 | 组成机制已存在。组合后的精确问题可能仍有空间，但必须证明不是把 patch selection 和 memory editing 换了名称。 |
| 表征 | 冻结 DINOv2 / VisionAD-style features | DPB-DF 代码初始化 pretrained Wide-ResNet-50-2 并提取 layer2/3；HITL correction 使用 PatchCore 表征；WeakREST 学习 residual Swin classifier。 | 冻结表征本身不是新颖性。须在严格对照下证明该表征对任务必要或有独立收益。 |
| 反馈 | 初始 Binary Normal/Defect；Type+Mask 为 Defect 后付费升级 | HITL correction 对每个查询只用一次 Normal/Defect 判断；DPB-DF 的训练数据为每张图提供类别 ID（含 defect class）；WeakREST 还考虑 box/block 等监督。 | 不能声称首次使用 binary feedback 做 memory correction。可能的差异是：仅凭 B-only history 推断地址，并以不确定性控制未来写入。 |
| 图像标签到 patch 分配 | 对候选 patch/地址产生 responsibility posterior | DPB-DF 用局部密度 kNN 投票做 hard 筛选；MIL 提供 soft attention；WeakREST 将弱标签转为已知/未知 block 后学习伪标签。 | “隐藏的 patch responsibility”作为一般概念已有覆盖；校准的地址后验与持续写入价值仍未验证。 |
| 分配不确定性 | 不确定性应影响写入、隔离或付费升级 | DPB-DF 是 hard vote，且会降低 k 直到留下最少 patch；WeakREST 会对未知 block 估计伪标签；MIL 权重虽连续但不等于校准概率。 | 明确校准并据此延迟写入是候选差异，仍需校准与效用消融。 |
| 在线状态 | 局部持久写入并检索 | DPB-DF 把筛选 patch 追加到 class bank 并更新权重；HITL correction 对 PatchCore bank 做插入/删除；在线 prototype 是相邻监督式持续学习基线。 | 持久 patch memory 本身不新。可能的贡献是地址级写入授权与未来效用。 |
| 标签授权 | B-only 不含缺陷位置/类型；rich label 需要单独计费 | DPB-DF 获得 defect 类别标签；WeakREST 覆盖 box/block；HITL correction 是 binary。 | B-only 主张不可使用 Type/Mask/source；rich 标签必须单列为付费机制或 evaluator-only oracle。 |
| 未来泛化 | 更新后在未见样本/产品上获得效用 | DPB-DF 做 few-shot/class-incremental 检查；HITL correction 在 held-out 样本上报告收益集中于欠覆盖正常外观；WeakREST 主要研究定位性能。 | “看未来 held-out”不独特；需要证明 binary 条件下地址级写入的因果机制。 |

## 最接近的 3–5 篇论文（3–5 Closest Papers）

### 1. Dynamic Patch Bank with a Defect Filter for Cold Start in Online Engineering Inspection（DPB-DF）

Jing Chen、Qingyang Wang、Wenqiang Xu，*Engineering Applications of Artificial Intelligence*，DOI [10.1016/j.engappai.2026.116388](https://doi.org/10.1016/j.engappai.2026.116388)，article 116388，Volume 184 Part 2。出版社页面给出的期刊期次日期为 **2026-11-15**，晚于本审计截点 2026-10-31。页面在截点前已公开列为 research article，且有预览；直接打开完整 ScienceDirect 页面返回 403。页面未明确标注 accepted 或 online-first，因此截至截点的精确状态为 **UNKNOWN；页面列出未来期次日期**，不将其记作已刊入该期。官方 [GitHub 实现](https://github.com/YoungerAnonymous/DPB-DF)及原始 [`models/defect_filter.py`](https://raw.githubusercontent.com/YoungerAnonymous/DPB-DF/main/models/defect_filter.py)、[`models/dpb.py`](https://raw.githubusercontent.com/YoungerAnonymous/DPB-DF/main/models/dpb.py) 可访问。

**具体方法比较（代码已核验）：** `DPB.update` 接收 patch embeddings 和图像文件夹类别标签；对每个非正常类别做降采样、调用 `DefectFilter.run`，再把保留的 patch 写入该类别 bank（`models/dpb.py` 第 71–83 行）。Filter 将正常 patch 与候选 patch 放入近邻索引，查询 k+1 个近邻、去掉最近的自邻居、进行类别投票并保留被判为 defect 的 patch；若少于 5 个，则递减 k 后重试（`models/defect_filter.py` 第 13–34 行）。之后系统在各类别 patch bank 上按逆距离做 patch kNN 投票（`models/dpb.py` 第 106–116 行）。论文预览的 Introduction/method overview 说明其使用冻结 ImageNet-pretrained encoder、正常密度 filter、dynamic bank 与 EWF。

这是对“弱图像标签能选出缺陷 patch 并写入持久类别记忆”最直接的威胁。它并不等同于 EISCA 的 B-only 条件：实现读取每个图像/patch 的 class ID，记忆 bank 按已知 defect class 区分；代码没有校准地址后验或 uncertainty/hold/abstain 写入规则。它是 hard neighborhood vote，而非 responsibility posterior。其标签比单一 Normal/Defect 更丰富。因此差异可收窄重叠，不能消除重叠。

**核验状态：** 方法/代码 `VERIFIED`；全文和截点时的最终出版状态 `PARTIALLY_VERIFIED`（出版社全文 403、期次日期在未来）。

### 2. Training-Free Human-in-the-Loop Anomaly Detection via Memory Bank Correction

Ayusha Abbas、Saram Abbas、Kabita Adhikari，**arXiv 预印本** arXiv:2608.17775 v1，2026-08-18。[HTML 原文](https://arxiv.org/html/2608.17775)、[PDF](https://arxiv.org/pdf/2608.17775)。截至截点未核实同行评审 venue。

**方法比较：** Section III-B / Algorithm 1 对查询图像只接收一次 Binary “normal”或“defect”。false-positive normal 图像将 784 个 patch feature 写入 normal bank（Eq. 2），可经 novelty gate 筛选；false-negative defect 则按 anchor patch 做 memory surgery（Section III-B.3）。该方法无需训练，直接修改部署期记忆。它用 held-out split 避免把已更新图像纳入评估。Binary feedback 与持久 memory editing 和 EISCA 高度重叠；但它没有 structural memory address 的后验：正常更正写入整张图的 patch，缺陷更正采用 hard anchor/removal。

它还提供了反驳“复杂 query controller 必然必要”的结果：Sections III-C、IV-E、IV-G 比较 error pool 内随机查询和 threshold-distance uncertainty sampling；在 15 个 MVTec 类别上，held-out 终点差异均不显著。作者将其归因于全局 bank update 的作用。该结果不等同于 EISCA 的 query stream，但足以让“主动选择必然有用”成为待证假设。

**核验状态：** 预印本全文 `FULL_TEXT_VERIFIED`；实验是作者报告结果，feedback 由 ground truth 模拟（Abstract、Sections III-E/IV/VI）。

### 3. Weakly-Supervised Residual Transformers（WeakREST）

期刊记录：Hanxi Li 等，“Accurate Industrial Anomaly Detection and Localization Using Weakly-Supervised Residual Transformers”，*IEEE Transactions on Image Processing* 35 (2026), 1551–1566，DOI [10.1109/TIP.2026.3659337](https://doi.org/10.1109/TIP.2026.3659337)，[PubMed 书目记录](https://pubmed.ncbi.nlm.nih.gov/41637707/)。方法全文检查作者的 arXiv 前身 [arXiv:2306.03492v6](https://arxiv.org/html/2306.03492)，2025-01-15，题为“Industrial Anomaly Detection and Localization Using Weakly-Supervised Residual Transformers”；未读取最终 IEEE 全文。

**方法比较：** Sections III-C.1–III-D.2、Eq. (13) 使用可训练 block-wise Swin classifier 处理 PosFAR residual。论文涵盖 pixel/mask、旋转/轴对齐 bounding box 和 image-level defect-status 标签。image-level 标记为 defect 时，起初所有 blocks 都是 unknown；ResMixMatch 再估计 block 标签并训练网络（Sections III-D.1–D.2、Algorithm 1）。这已覆盖从弱 binary 图像标签得到局部 anomaly scores。差异在于它离线训练定位模型，不根据查询反馈授权局部持久记忆写入，也不在多个记忆地址中分配写入质量。

**核验状态：** arXiv v6 方法 `FULL_TEXT_VERIFIED`；IEEE 书目 `VERIFIED`；最终期刊版与 v6 完全一致与否 `PARTIALLY_VERIFIED`。

### 4. Attention-Based Deep Multiple Instance Learning

Maximilian Ilse、Jakub Tomczak、Max Welling，ICML 2018，PMLR 80:2127–2136。[PMLR 页面](https://proceedings.mlr.press/v80/ilse18a.html)、[PDF](https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf)。

**方法比较：** 论文以一个 bag label 监督多个实例，并提出可训练、置换不变的 attention pooling；归一化实例权重呈现各实例对 bag 输出的贡献（gated attention Eq. 9）。这是从图像级/bag label 建立潜在实例贡献的直接数学先例。权重是 learned pooling weights，不天然等于校准或因果 responsibility；该方法也不管理部署期 memory 写入。它显著削弱了“仅凭 binary bag label 估计哪些 patch 重要”这一宽泛新颖性主张。

**核验状态：** PMLR 正文及 PDF `FULL_TEXT_VERIFIED`。

## 详细新颖性矩阵（Detailed Novelty Matrix）

标记：**Y**=直接存在；**P**=部分/邻近；**N**=所检方法中未见；**?**=未独立核验。

| 维度 | EISCA 主张 | DPB-DF | WeakREST | HITL Memory Correction | Attention MIL |
|---|---|---|---|---|---|
| 科学问题 | 从稀疏反馈推断持久地址的更新责任 | P：筛选 defect patches 并按已知类别保存 | P：由弱标签估计 block anomaly | P：Binary feedback 更正部署期 normal memory | P：估计实例对 bag label 的贡献 |
| 冻结 encoder | 冻结 DINOv2 | Y：pretrained 冻结特征流程 | N：Swin 可训练 | Y/P：PatchCore feature；不在线更新 backbone | N：端到端可训练 |
| 人工反馈粒度 | B=Normal/Defect；R=付费 Type+Mask | P：image/folder class label 包含 defect class | N/P：image status、box、block 或 mask | Y：一次 Binary 判断 | Y：bag label |
| 图像标签是否含来源/类型 | B 不含 | Y：已知类别 index 用来写 bank | image-status protocol 不必有来源类型 | 不含，只有 Normal/Defect | 通常是任务类别，不是 defect-source address |
| Patch responsibility | 对候选地址的 posterior | Y/P：hard local-density kNN vote | Y：block scores 和 pseudo-labels | N：整图正常 patches 插入 / hard FN anchor | Y：learned attention contribution |
| responsibility uncertainty | 明确且能改变写入决策 | N：hard vote，递减 k 保证最少 patch | P：unknown/pseudo-label confidence，不是 write posterior | N：确定性编辑 | P：soft weight 但非校准 responsibility |
| 在线更新 | 局部持久写入/保留未决证据 | Y：增量 class-bank append 与 EWF | N：离线网络训练 | Y：直接 memory edit | N：参数训练 |
| Backbone 更新 | 否 | 否 | 是，训练 Swin | 否 | 是 |
| 局部持久 memory | 是 | Y | N | Y，共享 normal bank | N |
| 反馈/写入授权 | 显式 authority/cost | P：class label + filter，无 B/R 协议 | P：监督协议，不是部署授权 | P：binary label 触发更新，含 novelty gate | N |
| 未来未见样本效用 | 必须 | P：class incremental benchmark | P：定位测试，非持续写入效用 | Y：held-out 与多 split | P：常规 bag 泛化 |
| 跨产品迁移 | 强主张所需 | P：多 benchmark，但任务标签不同 | P：多个异常检测 benchmark | N/P：仅 MVTec AD | N：非工业 MIL |
| 强控制 | 匹配 query/write/capacity 与 oracle/shuffle | 有消融，但非 EISCA matched-B protocol | 有弱标签/模型消融 | 有 held-out、passive/active、memory/cost 对照 | 常规 MIL 比较 |
| 成本/计算 | B/R 分钟、写入与检索成本 | few-shot/增量类；coreset 降采样 | 点击成本与可训练网络 | 每轮一个 Binary；bank 增长/上限 | 训练与 bag pooling |

## 数学与机制重叠（Mathematical / Mechanistic Overlap）

1. **Bag-label 不可识别性。** 设 patches 为 (z_{1:n})，图像标签 (Y=1) 表示至少有一个 patch defect。单独观察到 (Y=1) 无法识别具体 patch/address；许多潜在 assignment 都与此观测一致。因此 (q_{ij}=P(A=j\mid z_{1:n},Y,B)) 由表征与结构先验/似然决定，不能说是单个 binary label 直接“识别”出来。须验证跨产品校准和稳定性。这是对 H-A1 的核心反证。MIL 已形式化 bag label 与 instance contribution；DPB-DF 展示基于几何的 hard allocation；WeakREST 展示弱监督定位。
2. **Hard assignment 必须当强基线。** DPB-DF filter 已按近邻投票选择 patch；top-K、max pooling、简单局部密度规则和归一化 MIL attention 可构成从简单到复杂的责任分配器。EISCA 必须在相同 B 标签、写入质量下证明 posterior 不确定性改善未来写入效用。“soft 替代 hard”或“Bayesian 替代 heuristic”本身不是科学贡献。
3. **不确定性必须改变行动。** responsibility posterior 只有在它改变授权行动（写地址、分摊质量、隔离、保留或请求付费 rich feedback）时才有价值。如果各种不确定程度都写入相同地址，它只增加报告复杂度。
4. **操作地址不等于语义/因果。** DINO feature-space cluster 或 memory partition 可有预测价值，但不必对应人类可理解的缺陷原因。source/type、appearance address、空间 patch 坐标、semantic factor 必须分别记录。B-only estimator 不得读入真实 defect-source label。

## 最强已有基线（Strongest Existing Baselines）

窄版 EISCA 至少应在同一 query IDs、反馈、总写入质量、容量、support、threshold 和 sentinel/future stream 下比较：no-update；质量匹配 global broadcast 与 shuffled address；冻结 DINO top-K/max/score-softmax；在线 class-conditional centroid/prototype；MIL/top-K responsibility；以及在同一特征上实现的 DPB-DF-style 局部密度 filter。Oracle Type/Mask/source-address 只可当**不可部署的上限**，不可包装成 B-only 证据。

最强直接威胁是 **DPB-DF 的 local-density hard filter + class-specific dynamic patch bank**，因为它已经把带粗粒度标签的图像转换成筛选 patch 并持久储存。最强 Binary/HITL 威胁是 **Memory Bank Correction**，它以 Normal feedback 扩展 normal bank、以 Defect feedback 做 hard memory edit。无新增学习的简单基线是 raw-DINO nearest-neighbor/top-K 或 prototype。任务说明称 32-D sidecars 未胜过 raw DINO，E4B history credit 仅比 binary-only 高 0.0020。

## 对抗性反论证（Adversarial Counterarguments）

1. **“这就是 MIL + PatchCore memory edit。”** 批评者可把已有 latent-instance 方法与 HITL 预印本中的 online memory edit 组合。回应必须是可测的交互效应：校准责任在固定标签/容量时，比两者及其简单组合都更能提升未来未见效用。
2. **“地址收益来自 oracle label。”** Five-Branch 的正确来源地址已知，不能证明 B-only 系统可自动推断。若需要已知来源、类型或 mask，它是 paid-R/oracle 方法。
3. **“Binary patch responsibility 不可识别。”** 若冻结特征未区分负责与不负责 patch，且没有跨图重复结构，posterior 会由 prior 驱动；表征和 history 中不存在的信息无法被算法恢复。
4. **“收益只是多写了样本/扩大了 memory。”** 必须同时匹配写入质量和容量，并加入 proposal-faithful broadcast。Five-Branch 对已知地址路由做了 source-side 控制，但未解决可部署地址推断或端到端检测收益。
5. **“Soft posterior 不等于校准概率。”** 若有 mask/source 评估标签，仅在 held-out evaluator 中用它们评估校准、selective risk/coverage 与 write-harm。可视化热图不足以证明。

## 尚存科学差距（Remaining Scientific Gap）

**候选、尚未验证：** 冻结 DINO 空间中的当前状态证据，能否仅凭 B-only history 对操作性写入地址给出可校准、可识别的 posterior？该 posterior 是否在限制持久写入、计算写错地址及 unknown defect 风险后，仍能提升未来 held-out 决策效用？

除非有干预/不变性论证，不要称其为“causal responsibility”。当前问题最多支持操作性预测责任。

## S1 最小证伪方案（S1 Minimal Falsifier Proposal）

**先回答的问题：** 冻结传感器、相同 stream 与 B-only 可见性下，自动推断地址的 posterior 是否比最强简单 patch assignment 更能改善未来 held-out anomaly decisions？在不使用 Type/Mask/source label 的情况下，可恢复多少 oracle 已知地址的 headroom？

最小设计：

1. 在看结果前冻结 source-CV 产品、feature/scoring pipeline、support IDs、threshold、stream order、query positions、容量和 future sentinel。
2. 比较 no-update、质量匹配 broadcast、shuffled-address、raw-DINO top-K/score-softmax、简单 class prototype、MIL/top-K 与 DPB-DF-style density filter。EISCA estimator 只看当前 features/state/已揭示 Binary label。
3. 单独加入 oracle-address/mask 上限。oracle 信息必须对 controller/update code 隐藏。既有 Five-Branch 结果只说明应测上限，不证明自动 attribution。
4. Primary endpoint 选固定 threshold 下 future-sentinel BA 或预先定义的 cost-sensitive risk；AUROC、unknown-defect recall、按产品配对效果、write harm、地址校准为 secondary。queried/update images 不得进入 sentinel。
5. Oracle 若不优于最强简单基线，则停止新 estimator；若有 oracle headroom 但 B-only 方法不能胜过简单规则或不校准，则收窄/删除自动责任主张；若仅 rich Type/Mask/source 有收益，则按 paid-rich 方法报告。

这是 CPU/data replay 的研究设计建议，不是执行 S1 的授权。

## Pattern Recognition / IJCV 贡献要求（PR / IJCV Contribution Requirements）

**Pattern Recognition 最低要求：** 明确 B/R 可见性契约；区分 patch location、defect type/source、appearance address、semantic factor；提出非平凡且校准的 responsibility estimator；包括 DPB-DF-style filter、MIL/top-K、prototype 强基线；先做固定 query 的因果写入审计，再做 matched-budget 闭环实验；未来产品 hold-out；报告标注成本、容量、延迟、负结果和不确定性；部署路径无 oracle 信息。新名称、branch softmax、已知标签路由不够。

**IJCV 额外要求：** 提出一般问题形式与可识别假设；说明 binary label + frozen feature 在何条件下可/不可识别地址责任；对不确定性或写错地址的干扰传播给出界限或可证伪稳定性结果；展示跨产品/数据集与 support seed 迁移；胜过 MIL、class-prototype 和 DPB-DF-style 对照。理论贡献应解释问题何时可解、更新规则能保证什么。

## 局限与未核实主张（Limitations and Unverified Claims）

- DPB-DF ScienceDirect 全文直接打开返回 403。出版社预览和官方代码足以核验关键重叠，但全文公式、全部 ablation、limitation 与精确 online protocol 为 `PARTIALLY_VERIFIED`。
- WeakREST 最终 IEEE 全文未访问。PubMed 核对书目；方法来自作者完整 arXiv v6 前身；最终版本变化未知。
- HITL correction 是 arXiv 预印本，使用 MVTec AD，feedback 由 ground truth 模拟，未报告真实操作员实验；仍直接威胁宽泛机制新颖性。
- 这是定向检索，不是弱监督定位或 continual anomaly detection 的穷尽综述。
- 这里引用的 EISCA 历史分数来自用户任务说明，未从原始输出复算。
- 未运行实验、访问 Real-IAD、调用 GPU 或判断实现正确性。

## 暂定 GO / HOLD / NO-GO 建议（Provisional Recommendation）

**HOLD** EISCA mainline 于 N0。保留上面的窄候选问题，待人类审核是否授权 S1 前不扩展算法。若收益依赖 oracle Type/Mask/source label、可由写入质量或正例数解释、不能胜过 DPB/MIL/prototype 简单对照、不迁移到未来 held-out 产品，或伤害 unknown-defect recall，应停止 EISCA 主线。文献审计通过不等于科学 GO。

## 主要来源（Key Sources）

- [DPB-DF publisher DOI](https://doi.org/10.1016/j.engappai.2026.116388) · [official filter code](https://raw.githubusercontent.com/YoungerAnonymous/DPB-DF/main/models/defect_filter.py) · [official bank code](https://raw.githubusercontent.com/YoungerAnonymous/DPB-DF/main/models/dpb.py)
- [HITL Memory Bank Correction, arXiv:2608.17775](https://arxiv.org/abs/2608.17775)
- [WeakREST method, arXiv:2306.03492](https://arxiv.org/abs/2306.03492) · [TIP DOI](https://doi.org/10.1109/TIP.2026.3659337)
- [Attention-Based Deep MIL, PMLR](https://proceedings.mlr.press/v80/ilse18a.html)
