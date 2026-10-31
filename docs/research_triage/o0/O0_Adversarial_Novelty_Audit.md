# O0 — 对抗性新颖性审计

**审计 ID：** LASTPAPER-O0-20261031  
**立场：** 试图推翻而非包装候选方向。首轮后，对六个最有吸引力的方向做定向反向检索；查询记录见 [O0_Search_and_Exclusion_Log.md](O0_Search_and_Exclusion_Log.md)。

## 总体结论

没有证据支持把 EISCA、SCAEA 或其他单一候选称作明显未被研究的重要问题。宽泛说法已有直接竞争者：B-only 图像反馈改变 PatchCore memory（HITL Memory Correction, 2026 预印本）；粗粒度 defect label 转 patch bank 并接纳新缺陷类（DPB-DF, EAAI 2026）；多级标签粒度主动选择（至少 2008 年）；normality shift 适配（OWAD、RoDA、SH-HIL-PCF）；逻辑/组合异常（MVTec LOCO、PSAD、SALAD、LogiCo）；预算约束多视角/模态选择（ActiveInspect, 2026）；标签条件未来决策 VOI（decision-theoretic AL 与 targeted Bayesian AL）。

窄化后的状态转移、反馈授权、跨产品效用仍为 UNKNOWN。未查到论文不是新颖性的证明。

## A. O01 — 选择性反馈下安全、可撤销的持续适配

**可能主张：** 选择性/噪声/延迟 Binary feedback 与正常分布漂移交织时，propensity 记录和撤销更新可保护未来工业检测。

**反证来源：** [HITL Memory Correction](https://arxiv.org/abs/2608.17775) 直接用模拟 Binary feedback 修改 PatchCore memory；其预印本报告主动与被动查询在 held-out 终点无显著差异，defect-memory extension 失败。[SH-HIL-PCF](https://doi.org/10.1109/WcCST67302.2026.11496141) 已组合漂移告警、人类复核和 PatchCore 更新。[OWAD](https://www.ndss-symposium.org/ndss-paper/anomaly-detection-in-the-open-world-normality-shift-detection-explanation-and-adaptation/) 和 [RoDA](https://arxiv.org/abs/2503.14910) 已做正常性 shift 检测/适配或目标域分布对齐。有限反馈、non-response、在线风险与 propensity 加权另有一般理论。

**Reviewer 判词：** “安全漂移适配”宽泛主张不新。可继续的窄点只能是持久写入 harm 的识别条件/风险边界，且要证明不是标准随机审计、IPS、置信 gate 或 undo log。噪声上界、positivity、审计代表性很可能不符合生产环境。  
**结论：** 条件性继续审计；novelty UNKNOWN。

## B. O02 — 按未来效用购买不同反馈粒度

**反证来源：** [Multi-Level Active Prediction](https://proceedings.neurips.cc/paper_files/paper/2008/hash/4e0928de075538c593fbdabb0c5ef2c3-Abstract.html) 已按不确定性降低与相对成本选择不同图像标签粒度；[WeakREST](https://arxiv.org/abs/2306.03492) 已用 image tags/boxes/block labels 做定位；[MultiADS](https://openaccess.thecvf.com/content/ICCV2025/papers/Sadikaj_MultiADS_Defect-aware_Supervision_for_Multi-type_Anomaly_Detection_and_Segmentation_in_ICCV_2025_paper.pdf) 已结合 defect type 与 mask。

**Reviewer 判词：** “控制器选择标签名称不同”是旧问题重述。决策 VOI 已纳入标签成本与后续任务。只有真实标注分钟、同预算、未来产品留出，且优于多级 AL/VOI/随机/固定层级才有实证价值。  
**结论：** 广义 novelty FAIL；跨产品未来效用窄版 UNKNOWN；不入 shortlist。

## C. O03 — EISCA：只凭 Binary 历史推断持久 patch/address responsibility

**反证来源：** [DPB-DF](https://doi.org/10.1016/j.engappai.2026.116388) 用局部 embedding density 过滤粗粒度 defect 图像中的 patch，维护动态 patch bank 并纳入新类；其标签 authority 比 Binary 丰富，仍是直接威胁。[WeakREST](https://doi.org/10.1109/TIP.2026.3659337) 从 image tag/box 等弱标签学习 block/pixel localization。[HITL Memory Correction](https://arxiv.org/abs/2608.17775) 已用 Binary 反馈修正 normal memory。MIL、top-K、NN/prototype 是低成本强基线。

**Reviewer 判词：** 宽泛“Binary feedback 更新记忆”不新。oracle mask/source 或正确地址收益不证明 B-only responsibility 可识别；若 feature/bag history 不含该信息，posterior 只由 prior 决定。必须在完全隔离 evaluator 标签下胜过 top-K、prototype、MIL、density filter、broadcast/shuffle/no-update，且校准不确定度能改变动作。  
**结论：** 宽版 novelty FAIL，窄版 UNKNOWN；不入 shortlist。

## D. O06 — 稀疏反馈学习可迁移逻辑规则

**反证来源：** [MVTec LOCO AD](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) 已提供逻辑异常、像素 GT 与公开类别规则；[PSAD](https://ojs.aaai.org/index.php/AAAI/article/view/28703) 用 few labeled/unlabeled 图像学组件组合；[SALAD](https://openaccess.thecvf.com/content/ICCV2025/papers/Fucka_SALAD_--_Semantics-Aware_Logical_Anomaly_Detection_ICCV_2025_paper.pdf) 学 composition；[LogiCo](https://arxiv.org/abs/2606.28688) 统一逻辑与结构异常。LogicQA、LogicAD、ComAD、LogicAL 还覆盖 VLM解释、关系与逻辑异常生成。

**Reviewer 判词：** “让系统理解关系”不是缺口。公开类别规则不等于逐图人类反馈；public mask/type 不是线上监督。配置变化必须跨图、跨配置，五类 LOCO 不足以支持广泛工业主张。  
**结论：** 仅作等待人工 rule feedback 和新配置的数据后备；novelty UNKNOWN。

## E. O08 — 多模态/多视角动作与人类反馈选择

**反证来源：** [ActiveInspect](https://doi.org/10.3390/s26154932) 已将 view、modality、crop、normal reference 与 stop 纳入预算约束序贯证据选择，并用 Real-IAD D³ 等评估。[CVAD](https://doi.org/10.1109/TII.2026.3721718) 已做 Real-IAD 跨视角 few-shot 检测。Real-IAD D³ 已提供同步 RGB、pseudo-3D 和 point-cloud 数据。

**Reviewer 判词：** 换成 frozen backbone 或外部 memory 不能避开 ActiveInspect。区别必须是对象 A 的反馈改善未来对象 B 的动作策略，而非只让当前对象多看对一个视角。Real-IAD 的五个视角属于同一个物件，不能当五个独立样本。公开数据无人工动作轨迹和实测视角成本。  
**结论：** 当前不入 shortlist；persistent transfer UNKNOWN，成本数据缺失。

## F. O10 / SCAEA — 查询前预测 Binary 标签的未来效用

**反证来源：** [Selective Supervision](https://ijcai.org/proceedings/2007/141) 已用风险下降和标注成本定义 VOI；[Targeted Active Learning](https://openreview.net/forum?id=KxPjuiMgmm) 已针对下游最优决策 posterior 选 query；[Budgeted Online AL](https://ojs.aaai.org/index.php/AAAI/article/view/41191) 已研究有限 horizon 和预算。HITL Memory Correction 的主动/被动终点无差异是直接反例。

**Reviewer 判词：** 把 memory、correction、threshold 打包成 state 仍可用标准 Bayes-adaptive decision model 表示；AIF/EFE 没有自动新意。当前图像预测在 label 前已产生，不能计为 query 后收益。必须匹配 query 时点、label prevalence、support、初态、未来 sentinel、标注分钟并校正选择偏差。

**结论：** 通用公式 novelty FAIL；精确 IAD 状态转移 UNKNOWN；不入 shortlist。

## 十项 reviewer check 综合结果

| 评审问题 | 当前判断 |
|---|---|
| 是新问题还是旧算法换名？ | 大多数宽泛候选是旧问题重组；仅 O01 的风险/识别条件还值得审计。 |
| 是否给候选比基线更强的人类标签？ | 是主要混淆；Binary、Type、Mask、source oracle 必须分级，不能公平性混报。 |
| NN/prototype/rule/VOI 是否足以解决？ | 高概率；必须列为首轮基线。 |
| 是否真正持续学习？ | 只在 feedback 后的未来未见样本上评估，当前被查图不能算未来效用。 |
| 是否依赖 hidden GT？ | 策略与更新不可看未查询标签；private AD2 GT 只走官方 evaluator。 |
| 是否只是更多数据/查询/校准/support？ | 高风险；需匹配 query IDs、label prevalence、support、threshold、capacity、write count 与人时。 |
| 是否超出标准概率/continual learning？ | 尚未证明；EFE/NIG/Bayesian 名称本身不算差异。 |
| 理论保证是否平凡/假设不实？ | 高风险；positivity、有界噪声、平稳性与随机 audit 代表性待证明。 |
| 是否能独立证伪？ | O01/O03/O06/O10 可设计反例；O04/O05 需要新人工标注。 |
| 是否超出单数据集重要？ | O01/O06 可能重要，但现有数据/规则覆盖无法支持广泛结论。 |

## 对候选的共同停止规则

- policy 只看当下被授权的信息；mask/type/source/target GT 仅供独立 evaluator。
- Real-IAD 多视角按 object/sample 分组；同一对象多张视图不是独立物件。
- 固定 query replay 隔离写入因果效果；闭环策略按实际人时而非 query count 匹配。
- 报告 future sentinel 固定阈值风险、unknown-defect recall、误吸收、FPR/FNR、产品级差异、写入撤销和不确定性；AUROC 单指标不代表 safety。
- 若 simple NN/prototype/greedy bank/random audit/fixed rule/one-step VOI 胜出，则接受负结果，不以增加模块挽救主张。

## O0 shortlist 结论

只有 O01 进入条件性审计候选。O06 是等待真实规则反馈/新配置数据的后备；其余当前不建议投入。O01 仍是 UNKNOWN，本报告不授权 S1/Q1/S2/Q2 或任何新实验。
