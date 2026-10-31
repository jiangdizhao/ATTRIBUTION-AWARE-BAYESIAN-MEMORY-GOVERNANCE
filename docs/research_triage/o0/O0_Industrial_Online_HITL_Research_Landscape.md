# O0 — 工业在线人机反馈研究景观

**任务：** LASTPAPER-O0-20261031  
**检索截止：** 2026-10-31  
**范围：** 工业视觉异常检测 × 部署后/序贯学习 × 稀疏或昂贵人类反馈  
**结论状态：** 文献审计与候选筛选，不构成科学 GO、实验授权或稿件判断。

## 执行结论

六条研究线都已有强相关先例。最直接改变投资判断的是：训练自由的 PatchCore 人工记忆修正（2026 预印本）报告主动查询与被动查询在其协议下没有统计显著差异，且缺陷记忆扩展失败；DPB-DF（2026）已用粗粒度缺陷标签、局部密度过滤与动态图像块库做增量缺陷学习；ActiveInspect（2026）已把多视图/多模态证据选择表述为预算约束的序贯决策；逻辑异常检测已有 PSAD、SALAD 及多种无监督/组件级方法。由此，EISCA、SCAEA、AIF/EFE、可写外部记忆或多反馈层级都不能单凭组合成为新颖性主张。

当前最值得继续审计的问题不是某个新架构，而是：**在反馈选择偏差、反馈噪声和正常分布变化并存时，稀疏人工判断何时足以支持安全、可撤销的持久更新？** 这个问题很重要，但新颖性、识别条件和公共数据可检验性仍是 UNKNOWN。没有证据支持选出唯一获胜方向。

## F1 — 反馈类型、粒度与采集

**主流。** 主动学习按不确定性、代表性、决策风险或标签价值选择样本；弱监督视觉检测使用图像标签、框、块级标签、点或像素掩码；多级主动标注可按样本请求不同粒度。成本敏感 VOI 与“选择标签类型”本身早已建立。代表作包括 [Selective Supervision（IJCAI 2007）](https://ijcai.org/proceedings/2007/141)、[Multi-Level Active Prediction（NeurIPS 2008）](https://proceedings.neurips.cc/paper_files/paper/2008/hash/4e0928de075538c593fbdabb0c5ef2c3-Abstract.html)、[WeakREST（arXiv 前身；TIP 2026）](https://arxiv.org/abs/2306.03492)。

**尚未被证实的窄缺口。** 现有多级标注通常优化当前训练集/任务损失；它们未必估计“这次付费标签经许可的持久状态更新，对未来未见产品的增量效用”。但把目标函数从当前精度换成未来效用，仍可能只是标准决策理论的应用。必须以 cost-per-minute 对照随机、置信度/边界、覆盖、多级 AL、VOI 与无需 rich label 的简单基线。

**可证伪假设。** 在实际人时、类比例、查询数量和未来产品拆分匹配后，在线选择二元/类型/点或框/掩码层级能提高未来产品的决策效用；若随机升级或固定层级达到相同效用，停止复杂控制器。公开数据掩码只能模拟标签，不提供真实标注耗时或操作员错误率。

## F2 — 结构知识获取与持久复用

**主流。** PatchCore 类正常 patch memory、按类/缺陷存放的 prototype bank、MIL/弱监督定位和小样本缺陷分类已覆盖图像级信号到局部 patch 的多种转换。直接威胁包括 [DPB-DF（EAAI 2026）](https://doi.org/10.1016/j.engappai.2026.116388)（本次只能核对元数据、摘要及作者代码；出版社全文访问受限）、[HITL Memory Correction（arXiv:2608.17775）](https://arxiv.org/abs/2608.17775)、[WeakREST（TIP 2026）](https://doi.org/10.1109/TIP.2026.3659337) 和 [MultiADS（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/papers/Sadikaj_MultiADS_Defect-aware_Supervision_for_Multi-type_Anomaly_Detection_and_Segmentation_in_ICCV_2025_paper.pdf)。

**竞争判断。** “弱标签转 patch responsibility”“小样本写入记忆”“跨类型 defect prototype”不是空白。当前可能未解决的是：只有 B-only 标签且不能看 mask/type/source 时，能否识别对未来决策真正有用的写入地址；在什么可观测性/重复结构假设下可识别；估错地址的风险如何传到未来拒收/漏检。oracle 掩码或真实来源标签只可用于 evaluator 上限，不是部署反馈。

**可证伪假设。** 固定特征、查询集、总写入数、容量和阈值后，B-only 学到的持久结构是否胜过 raw-feature top-K、prototype、MIL/top-K、DPB-DF 式密度过滤、广播/打乱与不更新，并提高 never-updated future sentinel 的质量。既有项目报告的微小 history-credit 增益和压缩 sidecar 负结果应纳入假设先验，而不能视为自动可迁移。

## F3 — 正常变化、未知缺陷与分布变化

**主流。** 测试时/域适配、稳健分布对齐、漂移检测、正常样本持续学习、异常拒绝和开放世界学习是相邻主线。[OWAD（NDSS 2023）](https://www.ndss-symposium.org/ndss-paper/anomaly-detection-in-the-open-world-normality-shift-detection-explanation-and-adaptation/) 已针对无监督异常检测的 normality shift 检测、解释及适配；[RoDA（2025 预印本）](https://arxiv.org/abs/2503.14910) 用目标域有限样本做 memory-bank 分布对齐，并显式讨论目标域异常污染；[SH-HIL-PCF（IEEE WcCST 2026）](https://doi.org/10.1109/WcCST67302.2026.11496141) 已直接组合漂移告警、人类复核与 PatchCore memory 更新。持续异常检测基准 [Continual-MEGA（2025 预印本）](https://arxiv.org/abs/2506.00956) 与 [InCLAD（2026-10 预印本）](https://arxiv.org/abs/2610.15032) 也在快速扩展。

**尚未被证实的窄缺口。** 非平稳下区分“新但合格的正常性”和“新缺陷”非常重要；只有分布统计通常无法在没有标签、过程状态或可验证探索的情况下识别语义。公开 anomaly benchmark 主要是静态/人为构造顺序。防止 unknown defect 被吸收到 normal memory，与适配已知光照变化不是同一个问题。

**可证伪假设。** 在类别/样本泄漏受控的连续流中，保留小比例随机审计、反馈来源概率及可撤销写入能否以预先限定的误吸收率换取更快适配。若简单滑窗、固定 admission gate 或已发表 shift adaptation 达到同样风险/恢复曲线，则风险门控的新机制没有 headroom。

## F4 — 逻辑、关系与组合异常

**主流。** 组件分割、对象计数/布局/共现约束、结构与逻辑分离、关系 prototype、视觉语言解释和合成逻辑异常共同发展。数据基准 [MVTec LOCO AD](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) 有五类产品、结构和逻辑异常及异常区域像素标注；[PSAD（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/view/28703) 从少量标注与无标签图像恢复组件组合；[SALAD（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/papers/Fucka_SALAD_--_Semantics-Aware_Logical_Anomaly_Detection_ICCV_2025_paper.pdf) 建模 composition；[LogiCo（2026 预印本）](https://arxiv.org/abs/2606.28688) 统一 logical/structural 检测。

**尚未被证实的窄缺口。** 用户以稀疏反馈教系统可迁移、可修订的组合规则，并让其在后续产品/配置上改进检查，可能比单图解释更接近持续学习；但 LOCO 已公开正常性规则与逻辑标签，强静态方法充足，基准只有五类。用 GT rule 或组件标签训练后再称“在线人类规则学习”会造成监督不对等。

**可证伪假设。** 在从未见过的合法配置上，少量实际规则纠正能否让持久规则模型胜过固定组件计数、位置统计、SALAD/PSAD、VLM 问答和 nearest-prototype，并减少规则冲突导致的错误。若不能跨配置，贡献局限于工程提示。

## F5 — 主动多视角、多模态与交互诊断

**主流。** 固定多模态融合和主动选择 view、modality、crop、reference 或停止判定并行发展。Real-IAD 提供每个对象五个同步视角；Real-IAD D³ 同步 RGB、photometric-stereo pseudo-3D 与高精度点云。[CVAD（IEEE TII early access, 2026）](https://doi.org/10.1109/TII.2026.3721718) 已在 Real-IAD 上做冻结视觉基础模型的跨视角 few-shot 学习；[ActiveInspect（Sensors 2026）](https://doi.org/10.3390/s26154932) 已把 15 个候选视图/模态/裁剪等动作做成当前样本预算约束的序贯证据选择，并以观察数/准确率评估。

**尚未被证实的窄缺口。** ActiveInspect 是强直接竞争者。其公开报告的重点是当前对象的证据采集效用，不是由稀疏人工标签产生的跨对象、持久诊断策略学习。要主张后者，必须隔离同对象的多视图相关性、动作预算和当前图像收益，并在后续对象/产品上验证持久效用。没有公开数据集提供真实“操作员何时选择另一视图/何种标注”的轨迹。

**可证伪假设。** 在每对象多视角/模态访问次数匹配且按物件级分组时，学习过的反馈条件策略改善后续物件诊断，且超过固定 view 顺序、单一最优 view、贪心 VOI、ActiveInspect 类策略和随机策略。没有真实采集成本或在线可控传感器时只能做仿真，不足以支持完整工业主张。

## F6 — 噪声、矛盾、延迟与选择性反馈

**主流。** 鲁棒学习、多个标注者模型、偏差/非响应建模、有限反馈在线选择性分类、标签延迟后的 rehearsal 和反馈校正分别已有研究。[Active learning with biased non-response（DMKD 2024）](https://doi.org/10.1007/s10618-024-01026-x) 说明非响应机制会造成选择偏差；[Label Delay in Online Continual Learning（NeurIPS 2024）](https://papers.neurips.cc/paper_files/paper/2024/hash/d8f5f134febb4bd74d8f79e338de382c-Paper-Conference.pdf) 研究延迟标签；[Online Selective Classification with Limited Feedback（NeurIPS 2021）](https://papers.neurips.cc/paper_files/paper/2021/hash/79b6245ff93841eb8c120cec9bf8be14-Abstract.html) 给出有限反馈和 abstention 的形式化。

**尚未被证实的窄缺口。** 对工业异常检测的稀疏、由策略选中的 Binary 人类反馈，若错误反馈可永久改变正常 memory / threshold / correction，何种随机审计与可逆 state update 能识别并限制 write harm？现有监督理论提供先例，但不能直接证明项目机制安全。仅对被查询样本报告校准会掩盖未查询区域的漏检和选择偏差。

**可证伪假设。** 在控制查询动作、错误率、延迟、标签构成和记忆容量后，带来源记录、隔离/撤销与随机审计的更新相对简单置信度 gate、majority/relabel 和不更新，是否同时降低未来 sentinel 风险与恢复时间。若收益依赖不可观测的真实 propensity 或完美反馈，或标准逆概率加权已解释结果，则无新原理。

## 研究机会排序的初步判断

| 方向 | 当前证据判断 | 主要阻碍 |
|---|---|---|
| 稀疏/选择性反馈下的安全、可撤销在线更新 | 有条件继续审计；风险重要，但 novelty UNKNOWN | 与 memory correction、漂移适配、选择性反馈理论重叠；需要可辩护的可识别假设 |
| 二元反馈推断持久 patch/address responsibility（EISCA） | 暂不入 shortlist | DPB-DF、WeakREST、MIL 与 HITL memory 直接竞争；责任可识别性未证 |
| 查询前估计标签条件的未来效用（SCAEA） | 暂不入 shortlist | decision-theoretic VOI/Bayesian AL 已覆盖一般原理；anomaly-specific 直接结果不支持主动策略有普遍收益 |
| 跨产品学习组合规则 | 后备候选，需人工数据与配置设计 | MVTec LOCO 方法强，标签只有检测 GT；产品/规则覆盖窄 |
| 多模态证据动作的持续复用 | 暂不入 shortlist | ActiveInspect 已覆盖当前图像动作策略；公开序贯人类交互协议缺失 |
| 自适应标注粒度 | 主要是工程机会 | 多级主动标注已有理论与算法；真实成本和持续转移效用缺失 |

## 证据级别

- **VERIFIED**：官方 proceedings、期刊/数据集主页或作者原文中核对了书目和相关部分。
- **PARTIALLY VERIFIED**：摘要、出版社记录或作者代码可用，但全文/补充材料/精确协议未完全核对。
- **INFERRED**：由已核对工作的差异推断出的可能空白，不等于完成系统新颖性搜索。
- **UNKNOWN**：本轮搜索无法确认，不得写成“首个”或“无人研究”。

完整书目、验证范围和局限见 [O0_Primary_Sources.csv](O0_Primary_Sources.csv)；搜索与排除见 [O0_Search_and_Exclusion_Log.md](O0_Search_and_Exclusion_Log.md)。
