# O0 — Research Opportunity Cards

**审计 ID：** LASTPAPER-O0-20261031  
**证据标签：** VERIFIED 表示来源/数据事实在指定来源中核过；INFERRED 表示竞争关系或缺口判断；HYPOTHESIS 表示待证伪的科学主张；UNKNOWN 表示当前检索不能判断。  
**说明：** “四个月可行性”指形成可审查的研究结果，不是保证发表。任何卡片均未授权实验。

## O01 — 选择性反馈下安全、可撤销的在线正常性适配

1. **科学问题：** 当部署流中正常外观变化与新缺陷同时可能出现，而人工二元标签仅对系统选中的样本返回且可能延迟/出错时，系统何时可以安全地把证据写入正常记忆？
2. **工业动机：** 照明、批次、夹具或工艺变动增加误报；若把未知缺陷误收为正常，后续同类缺陷可能持续漏检。
3. **最近方法：** [R05] HITL Memory Correction；[R09] SH-HIL-PCF；[R07] OWAD；[R08] RoDA。
4. **精确缺口：** 前三类分别已有人工 PatchCore 更新、漂移触发人工反馈、无监督 shift adaptation；尚 UNKNOWN 的是选择性/非响应反馈下，持久异常状态更新的可识别条件、误收风险和撤销机制是否需要超出常规随机审计/propensity 校正。不能称作无人研究。
5. **潜在理论贡献：** 假设性贡献是给出条件式可识别性边界：若某区域 query propensity 为零或反馈错误率无上界，未来效用不可从观察反馈识别；再研究最小随机审计率与有界噪声下的状态风险控制。该结果可能只是经典选择性标签/在线风险界的直接应用，需先查数学先例。
6. **最强简单基线：** 冻结 detector；固定阈值 + 最近邻 novelty gate；滑动窗口/greedy prototype；SH-HIL-PCF 风格 PSI/KS trigger；随机审计 + inverse-propensity 风险估计；HITL normal-only memory insertion。
7. **最小证伪：** 先设计同一类别内正常光照/背景变化与 held-out 缺陷的交错流，在正常反馈漏失、延迟和可控误标下比较固定 gate、随机审计和可撤销更新。若简单 gate 在同等误吸收率下恢复更快，或所谓优势依赖真实标签/未来 GT，则停止复杂机制。只需公开训练图像与 evaluator-only 标签；不执行本任务实验。
8. **数据与反馈：** MVTec AD 2 有正常训练/验证、公开测试部分像素真值与私有测试真值；其照明差异是真实场景变化但不是持续顺序。Real-IAD 有 sample/type/mask 和五视角；物件/视角需按 sample group 防泄漏。反馈可先由 GT 模拟 Binary，再独立模拟响应倾向/延迟/噪声；真实人时与错误率需人工采集。私有 AD2 GT 仅服务器 evaluator。
9. **主要风险：** MVTec AD2 的 shift 不能表示真实工艺概念漂移；随机审计/逆概率加权或固定 gate 可能完全解释收益；没有人工响应机制就无法估计真实反馈质量；过度隔离会让适配停滞。
10. **发表潜力：** 工程：若仅调门限。PR 方法：若风险匹配下有跨数据恢复收益。IJCV：仅在有非平凡且可验证的可识别性/风险界、外部流确认时。
11. **四个月可行性：** 中（只做离线模拟和两数据源）；若要求真实操作员与过程状态，高。
12. **置信与证据：** SHIFT/数据限制 VERIFIED [R12]；最近方法 VERIFIED/PARTIALLY VERIFIED [R05,R07–R09]；具体科学缺口 INFERRED；新颖性与理论 HYPOTHESIS / UNKNOWN。

## O02 — 反馈粒度是否应按未来可迁移效用付费

1. **科学问题：** 对当前对象请求 Binary、Type、Point/Box、Mask 或 Rule 中哪种反馈，能在标注分钟固定时最大化后续产品的检查效用？
2. **工业动机：** 全量像素掩码成本高；单一 Binary 无法定位；一种反馈可能服务多个产品，另一种可能只改善当前对象。
3. **最近方法：** [R01] Selective Supervision；[R02] Multi-Level Active Prediction；[R03] WeakREST；[R06] MultiADS。
4. **精确缺口：** 多级标注与成本敏感 VOI 已知；UNKNOWN 仅是工业异常检测里“标签动作→受限持久更新→跨产品未来效用”的实证优势能否在真实按分钟预算下超过多级 AL/静态层级。组合本身不构成贡献。
5. **潜在理论贡献：** 反馈种类作为不同 observation/action channel 的 Bayes-adaptive design，含显式标注成本、反馈错误与未来产品分布。数学原则已有；可能新的是可识别的 transfer utility estimand，而非 VOI 公式。
6. **最强简单基线：** 固定 Binary-only；永远请求最便宜标签；每次都请求最丰富标签；随机层级；多级 AL；每分钟标签信息增益；后验风险 VOI。
7. **最小证伪：** 在相同反馈分钟与候选序列上，对实际可见标签等级 replay。若一个固定层级或随机层级与主动规则等效，或性能收益来自更多标签/更多 defect positives，结束动态粒度主张。
8. **数据与反馈：** MVTec AD、VisA、Real-IAD 有 image labels 与 mask；Real-IAD 有 defect types。它们可做 simulated B/Type/Mask，point/box 可从 GT 自动生成但只是假设成本；rule 无现成逐样本解释。真实标注耗时须人测。
9. **主要风险：** 多级标注主动学习很早已有；真实 annotation cost 不等于几何像素数；synthetic point/box 低估标注噪声；后续效用可能等同训练数据增加。
10. **发表潜力：** 默认主要是工程贡献；严格未来效用与可靠成本模型或可达 PR 方法。IJCV 理论潜力低，除非建立新 identifiability/transfer 结果。
11. **四个月可行性：** 中。
12. **置信与证据：** 多级 AL 已有 VERIFIED [R01–R03]；persistent future-value gap INFERRED；工业新颖性 UNKNOWN。

## O03 — 仅凭 Binary 历史是否能识别持久 patch/address responsibility（EISCA）

1. **科学问题：** 冻结表示和只能使用事后 Binary N/D 的事件历史，能否可靠识别哪些 patch/地址应获得持久更新，而不读取 defect type、mask 或 source？
2. **工业动机：** 一张缺陷图通常含大量正常背景；广播写入会污染无关记忆，错误 patch assignment 可能削弱未来检测。
3. **最近方法：** [R04] DPB-DF；[R03] WeakREST；[R05] HITL Memory Correction；[R14] PSAD/MIL 方向。
4. **精确缺口：** 局部 density filter、weak labels、instance responsibility、normal memory correction 均有先例。剩余差异是持久地址的校准后验及其 B-only 跨样本识别能力，仍 UNKNOWN。若表示中没有相关信号，责任不可从 image-level label 恢复。
5. **潜在理论贡献：** 在明确的 patch exchangeability、重复结构、label-noise 与 positivity 假设下证明 responsibility 可识别/不可识别；错误地址写入如何影响 future risk。不得仅重命名分数或称“semantic factor”。
6. **最强简单基线：** raw DINO top-K/score-softmax、局部密度 hard filter、class prototype、MIL/top-K、fixed threshold、broadcast/shuffle/no-update；Type/Mask oracle 只作为不可部署上限。
7. **最小证伪：** fixed feature/query/write count/capacity/support/threshold；比较 B-only estimator 和简单分配器在未更新 future sentinel 上的成本敏感风险与地址校准。若 oracle 地址对简单基线都无 headroom，或 B-only 方法不能赢最强 simple baseline，停止。
8. **数据与反馈：** Real-IAD 或 VisA 的 public image labels/masks 可供 evaluator 校准；实时更新只给模拟 Binary。type/mask/source 必须只留 evaluator。跨产品泛化需按对象、产品类别、来源拆分；同一图里的多个 patch 不是独立样本。
9. **主要风险：** DPB-DF 与 HITL correction 已占据机制核心；宽泛 weakly-supervised localization 很旧；项目历史 E4B 小增益不支持大 headroom；结果可能由写入数或 label support 解释。
10. **发表潜力：** 未证明识别性前为工程；若有条件式识别/校准并跨产品胜过强简单方法，才可能 PR。IJCV 要求通用假设和风险结果。
11. **四个月可行性：** 中–高，主要难在因果配对、标签边界和容量匹配。
12. **置信与证据：** 竞争者 VERIFIED/PARTIALLY VERIFIED [R03–R05]；binary identifiability 为 UNKNOWN；候选 hypothesis。

## O04 — 由 sparse rich feedback 学习跨产品 defect attributes，而不是复制外观

1. **科学问题：** 少量真实缺陷的 Type/属性/局部证据能否形成可迁移的结构表示，使未见产品上的新 defect 诊断优于按类 exemplar/prototype？
2. **工业动机：** 磨损、污染、缺件等机制跨产品重复，但纹理、材质、视角和缺陷大小不同；逐类别积累样本昂贵。
3. **最近方法：** [R06] MultiADS；[R04] DPB-DF；[R03] WeakREST；预训练 VLM defect-type prompting。
4. **精确缺口：** 已有 defect-aware taxonomy/VLM 与增量类别 patch bank；尚 UNKNOWN 的是通过部署后稀疏人工反馈得到、跨对象复用且对未来二元/定位决策有增益的 attribute state。没有公开基准的“人类 attribute correction”。
5. **潜在理论贡献：** 可能是共享/产品特有 latent attributes 的可识别边界与 transfer condition；尚无理论结果，不应声称新 taxonomy。
6. **最强简单基线：** 冻结 VLM prompt、nearest labelled defect prototype、class-specific bank、MultiADS-style known taxonomy、few-shot NN。
7. **最小证伪：** Leave-product-family-out：反馈只用于少数源产品，评估目标产品；将 attribute feedback 与同成本的类别标签、image-text retrieval、prototype 对照。无跨产品收益或必须输入 defect-name GT 即终止。
8. **数据与反馈：** Real-IAD 的 defect category/type、mask 可用于 GT 评估；MultiADS 已利用这些 metadata。human text attribute/rule 不存在，需工业检查员真实标注。模拟属性模板不等价真实 feedback。
9. **主要风险：** 从数据集 defect 名称得到“transfer”可能是标签泄漏；新类别语义与视觉特征错位；VLM prompt/prototype 足够；实际属性术语依工厂/产品不同。
10. **发表潜力：** 成功的跨产品评估可达 PR 方法；IJCV 需一般化的属性学习理论与独立工业验证。
11. **四个月可行性：** 高（必须真实专家参与；仅数据模拟为中但证据弱）。
12. **置信与证据：** dataset types/masks 与 MultiADS overlap VERIFIED [R06,R17]；真实人类跨品属性学习差距 INFERRED/UNKNOWN。

## O05 — 变化的是可接受质量规范，还是成像分布？

1. **科学问题：** 如何在 deployment 中分开检测外观变化、产品配置变化和用户对可接受质量规范的显式改变，并只更新对应持久状态？
2. **工业动机：** 合格容差、材料批次或产品修订可能改变；把新规则误当纯视觉 shift 会造成旧规范下的错判，盲目适应则可能放过质量缺陷。
3. **最近方法：** [R07] OWAD；[R08] RoDA；[R09] SH-HIL-PCF；[R12] MVTec AD 2。
4. **精确缺口：** 视觉 benchmark 记录外观/照明及异常标签，但通常不记录版本化的质量政策或“允许缺陷变化”。缺少这种信息时，策略变更与分布变化可能不可识别；精确缺口属于 UNKNOWN。
5. **潜在理论贡献：** 区分观测分布 drift 与 label/acceptance concept change 的不可识别性证明，以及何种政策元数据/询问可识别。必须先检索 concept drift/statistical decision theory 前身。
6. **最强简单基线：** 固定当前规范；每次版本变更由人工配置新阈值；普通 domain adaptation；不适应并触发人工复核。
7. **最小证伪：** 构造同样 image marginal、但 quality label policy 不同的两种环境；若只有图像/旧标签的任意方法可一致判别，说明实验泄漏；检验显式版本反馈是否只需更新阈值。
8. **数据与反馈：** MVTec/Real-IAD 不含版本化可接受性政策；任何“规范改变”反馈都需人造/真实质量工程规则。当前公开 masks/type 是 evaluator labels，不是规范审定。
9. **主要风险：** 需企业级规则与审批数据；规则切换近乎常量，可被阈值配置解决；理论贡献可能只是不可识别性的直接推论。
10. **发表潜力：** 若有真实规范改变及安全结果，有 PR/IJCV 问题潜力；无企业数据则停留在工程或合成演示。
11. **四个月可行性：** 高。
12. **置信与证据：** 数据缺少规范标签 VERIFIED；跨域 gap INFERRED； novelty UNKNOWN。

## O06 — 人类纠正规则能否跨配置持久转移（逻辑异常）

1. **科学问题：** 少量人类对缺件、错数、错位/错关系的纠正，能否学到可修订的产品规则，并提高后续新配置检查，而非只修当前图像？
2. **工业动机：** 单个螺丝/连接器可视觉正常，但数量、次序、相对位置或配对不符。
3. **最近方法：** [R13] MVTec LOCO；[R14] PSAD；[R15] SALAD；[R16] LogiCo。
4. **精确缺口：** 强方法已学习/利用组件 composition、logical/structural 约束；LOCO 还公开五类产品正常性规则。缺口只可能在“反馈更新规则并转移至未见配置”，其 novelty UNKNOWN；不可称 logical AD 未解决。
5. **潜在理论贡献：** 在可观测组件抽取误差下，规则约束何时可识别、组合异常的规则覆盖/泛化界、矛盾反馈如何导致例外/规则版本更新；都只是可能方向。
6. **最强简单基线：** 固定计数/位置/关系规则、组件直方图、PSAD/SALAD/LogiCo、VLM 规则问答、最近正常 component-set。
7. **最小证伪：** 一个产品多个合法配置，训练流只在某些配置出现 Binary/规则 correction，测试未见配置。与人工预写规则和最强组件方法同等信息比较；若人工 rule 模型不超固定/视觉 baseline，或规则变化不能迁移，则终止。
8. **数据与反馈：** LOCO image-level 正常/逻辑/结构分组及异常像素 masks 已有；规则说明是类别定义，不代表逐图 annotated relation。真实纠正规则需人标。Real-IAD 可提供缺件/异物等，但无组合拓扑规则。
9. **主要风险：** 五类别基准规模小；public rule leak；可见 anomalous pixels 不代表逻辑责任区域；规则学习可能只是 prompt engineering；生产图中视角/遮挡会破坏组件抽取。
10. **发表潜力：** PR 方法潜力；IJCV 需更广跨配置/产品泛化和规则学习理论。
11. **四个月可行性：** 高；若需独立人工规则数据则高。
12. **置信与证据：** MVTec LOCO 事实与 PSAD/SALAD 方法 VERIFIED [R13–R15]；反馈 transfer 缺口 INFERRED / novelty UNKNOWN。

## O07 — 延迟、矛盾或错误的反馈是否能撤销状态更新？

1. **科学问题：** 系统收到延迟或互相矛盾的 Binary/Type/Mask 反馈时，何种 provenance、隔离、撤销和重新验证策略可限制持久记忆的未来损害？
2. **工业动机：** 操作员误判、复检结果推翻、掩码质量不足或延迟复核都可能留下错误写入。
3. **最近方法：** [R05] HITL Memory Correction；[R23] biased non-response；[R24] label delay；鲁棒在线学习/噪声标签方法。
4. **精确缺口：** 延迟和 noisy feedback 在一般在线学习有结果；异常 memory 写入与未来 defect miss 的具体影响未知。可逆 update、quarantine 或 majority vote 本身不新。
5. **潜在理论贡献：** 在反馈延迟与错误上界条件下对最大 future risk increase 给出界，或刻画“不可逆”持续状态下的最低确认成本。边界很可能能由现有 robust online learning 推导，尚需查先例。
6. **最强简单基线：** 不更新；一票确认后写入；双人多数表决；短期 quarantine；undo log；冻结状态版本。
7. **最小证伪：** 在固定错误率/延迟/label prevalence 下，注入纠正前后矛盾；若 undo log + simple majority 已达到相同未来风险/label cost，停止额外概率模型。
8. **数据与反馈：** Real-IAD/MVTec labels 作 evaluator；噪声/延迟是受控模拟；真实操作员冲突数据不可由公开 masks 推断，需要真实复核协议。
9. **主要风险：** 模拟错误过于理想；多数票增加成本；不可检出的系统性 label bias 不满足常用独立噪声假设；可能只是工程数据治理。
10. **发表潜力：** 多数是工程；若新颖且有非平凡 regret/safety bound，可到 PR。IJCV 贡献机会有限。
11. **四个月可行性：** 中。
12. **置信与证据：** 在线 delay/nonresponse 先例 VERIFIED [R23,R24]；视觉记忆撤销的精确风险界 UNKNOWN。

## O08 — 再看一个视角/模态，还是现在问人？

1. **科学问题：** 对有多视角/3D证据的对象，下一步应选另一 view、放大局部、查询 Binary 人类反馈，还是结束检测，才能提高后续对象的决策效用？
2. **工业动机：** 反光、遮挡、几何凹陷可能在某视角不可见；重新取像可能比人工掩码便宜，也可能无效。
3. **最近方法：** [R19] ActiveInspect；[R20] CVAD；[R18] Real-IAD D³；[R21] decision-aware AL。
4. **精确缺口：** ActiveInspect 已直接学习当前对象的多模态/多视角证据顺序。尚未验证的范围是反馈是否能改善未来对象的动作策略和状态；不是“首次主动看视角”。
5. **潜在理论贡献：** 不同 evidence action 的成本与状态价值构成 POMDP/Bayes-adaptive sensing；若使用普通 VOI 已能表示，应无视 AIF 标签。可能贡献是可重复的因果估计协议，不一定是新算法。
6. **最强简单基线：** 全视角；固定最佳 view；单步贪心 VOI；ActiveInspect；random action；view/crop round robin。
7. **最小证伪：** 物件分组、完整 matched observation pool、matched cost，比较主动行动策略在下一批对象上的结果。若只改当前图像但跨对象不迁移，不能叫 persistent learning。
8. **数据与反馈：** Real-IAD 每物件五视角，Real-IAD D³ 同步 RGB/pseudo-3D/3D；这些视角是同一物件的相关观测，不可当独立样本。公开数据无人工查询动作轨迹/真实视角获取成本；反馈可模拟，真实 action costs 需设备与操作员测量。
9. **主要风险：** ActiveInspect 已覆盖大部分序贯 action space；数据池式选择不等于闭环相机控制；同产品拆分泄漏；合成成本和GT策略可产生假优势。
10. **发表潜力：** 当前可能工程；跨生产流动作学习与实测资源成本或达 PR。IJCV 需要多站点 validation。
11. **四个月可行性：** 高。
12. **置信与证据：** 数据/现有方法 VERIFIED [R17–R20]；持久动作迁移 gap INFERRED，novelty UNKNOWN。

## O09 — 选择性查询是否隐藏了模型最需要的漏检区域？

1. **科学问题：** query policy 面向高分误报时，如何给低分但真实 defect/新类型保留可估计的探索概率，使未来风险在未标注流上可审计？
2. **工业动机：** 操作员往往只复核告警；静默漏检不被标注，导致模型看似持续改善却扩大盲区。
3. **最近方法：** [R22] Budgeted Online AL；[R23] Active learning with biased non-response；[R24] Label Delay；[R05] HITL memory correction。
4. **精确缺口：** selection bias/有限预算在线 AL 已研究；视觉 AD 的全流安全评估和长期阈值/记忆状态可否用小比例随机审计识别，仍需独立研究。泛化的“随机 query 有用”不是新颖性。
5. **潜在理论贡献：** 在非零 propensity、bounded loss 和固定反馈噪声条件下，对未查询风险及策略间未来效用作有限样本估计；须和 off-policy evaluation、selective labels、online bandits 比较。
6. **最强简单基线：** 固定 1–5% uniform random audit；score-stratified audit；每个区域分层随机抽样；简单 IPS/AIPW；按异常分数 uncertainty sampling。
7. **最小证伪：** 固定总查询预算与实际反馈构成，比较有/无随机审计时 future sentinel 风险置信区间覆盖与决策效果。若随机审计只增加成本且可用普通分层估计恢复覆盖，停止复杂 controller。
8. **数据与反馈：** Real-IAD/MVTec AD public test labels/masks 可作离线 evaluator，缺陷稀有率可以按类别构造；真实流中的操作员 non-response/审计机制不可见，必须另采集。AD2 private test 的GT只可经官方 evaluator。
9. **主要风险：** 任何 query propensity 为零则 IPS 不能识别风险；极低查询率造成高方差；公开 GT 模拟操作员反馈会夸大响应率；优化估计而不改变未来检测是无意义目标。
10. **发表潜力：** 若形成可信全流风险评估与低成本规则可达 PR；只证明随机抽样无偏为方法应用，期刊层级有限。
11. **四个月可行性：** 中。
12. **置信与证据：** selectivity/online AL 先例 VERIFIED [R22–R24]；真实工业视觉 propensity availability UNKNOWN。

## O10 — 查询前能否预测“这个 Binary 答案会改善以后” （SCAEA）

1. **科学问题：** 在查询前只给定图像、当前部署状态和已允许信息，能否预测 N 与 D 两种潜在标签分别触发的状态更新对未来未见对象风险的影响？
2. **工业动机：** 同一个 Binary 标签可能改变正常 memory、correction、threshold 或状态校准；按当前熵/边界选样不一定改善下游部署。
3. **最近方法：** [R01] decision-theoretic AL；[R21] Targeted Bayesian AL；[R22] budgeted online AL；[R05] HITL memory correction。
4. **精确缺口：** label-conditioned future utility under explicit authorized state transitions 的视觉 IAD 版本尚未验证；但一般 future decision VOI 已知，且当前 HITL 论文没有发现主动优于被动。不能把状态向量写得更复杂称为理论新意。
5. **潜在理论贡献：** 无泄漏估计状态条件的多步反事实效用、选择偏差校正与误差界。若所有状态可作为一个标准 Bayes-adaptive MDP，理论部分可能只是标准形式化。
6. **最强简单基线：** random query；margin/entropy；threshold-distance；coverage/reliability heuristic；one-step cost-sensitive VOI；Targeted AL；BEA 与 simple AIF 复现。
7. **最小证伪：** 在固定 candidate、support、未来 sentinel 和初态下，仅评估实际标签揭示后的未来样本；匹配 label prevalence/时间/成本。若 N/D 条件效用不能查询前预测，或 matched-budget random/simple VOI 达同效，停止 SCAEA。
8. **数据与反馈：** Real-IAD 可作序列数据、type/masks 作 evaluator；B-only 标签由GT模拟。需要独立流顺序、随机 audit、严格隐藏未查询标签。数据不提供真实查询时点或标注成本。
9. **主要风险：** VOI/decision-focused AL 已知；label outcome 和未来效用可能不可预测；history/order 可能使用 evaluator GT 泄漏；查询收益因 BA/FPR/FNR 指标冲突。
10. **发表潜力：** 若只是包装 VOI，为工程；若有可迁移 estimand 与控制反馈选择偏差的强结果，可到 PR；理论 novelty 当前低。
11. **四个月可行性：** 高（源代码已有状态可能可 replay），但可靠跨产品因果验证为高。
12. **置信与证据：** VOI 先例 VERIFIED [R01,R21,R22]；HITL active/passive 负结果 VERIFIED 为预印本 [R05]；SCAEA 差异 UNKNOWN。

## 初始排序

- **暂列条件性候选：** O01。问题最重要，且最可能促使我们放弃不安全的持久更新；只有在识别条件和数据可行性审计后才可继续。
- **需要另行专业定义的后备：** O06。组合规则重要，但须获取逐配置的人类规则纠正，并与强组件/逻辑基线比较。
- **不推荐当前投入：** O02、O03、O04、O05、O07、O08、O09、O10。它们仍是可证伪问题卡，不代表科学方向选择。
- 所有候选的支持/反证、审批与下一步见 [O0_Decision_Register.md](O0_Decision_Register.md)。
