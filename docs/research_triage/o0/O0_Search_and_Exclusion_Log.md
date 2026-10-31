# O0 — 检索与排除记录

**审计 ID：** LASTPAPER-O0-20261031  
**研究截止：** 2026-10-31  
**实际网页检索日期：** 2026-10-31  
**检索方式：** 网络语义检索、官方 proceedings/出版社/数据集页、可访问论文全文及作者代码；同时定向查找是否有先例反驳候选主张。不是系统综述，不宣称穷尽。

## 仓库与研究边界核查

- git status --short --branch：stage3-eg-famar-execution...origin/stage3-eg-famar-execution，工作区干净。
- HEAD：7e1d3f66385831f5b1857dff68aeae1ab6beeeb4。
- PowerShell 5.1；Python 命令解析至 D:\anaconda3\python.exe。
- G:\dataset\realiad 与 G:\dataset\realiad\jsons\realiad_jsons 路径存在；本任务未枚举或读取数据集。
- 项目要求的 README、Stage-3 handoff/plan 与 N0-A/N0-B/decision register 存在并被选择性查看。项目所列三个 .docx 在工作区文件清单中不可访问，记录为 NOT_ACCESSIBLE；没有据此推断其内容。
- 未查阅或读取任何 sealed GT/隐藏 Probe 或 Confirmation 标签。仅查阅公开网页、仓库 Markdown 与 Git 元数据。
- 目标目录此前不存在。本任务只新增 docs/research_triage/o0/ 中八个审计文件。

## 检索顺序与代表关键词

### 第一轮：六个研究家族的机会发现

1. F1/F2/F3：industrial anomaly detection active learning human feedback online continual anomaly detection 2024 2025；weak supervision annotation granularity point mask active learning defect inspection；concept drift open-world anomaly detection human feedback。
2. F4/F5/F6：industrial logical anomaly detection compositional relations MVTec LOCO；industrial anomaly VLM interactive rule learning；active multiview anomaly detection choose camera view；noisy labels delayed feedback human-in-the-loop continual learning。
3. 标签粒度与数据：multi-fidelity active learning annotation type selection；MVTec AD 2 hidden ground truth；Real-IAD multi-view D3 annotation masks。
4. 直接竞品：DPB-DF defect prototype bank；WeakREST；continual industrial anomaly detection benchmark。

### 第二轮：反驳前五/六个较强候选（必须的反向检索）

1. 对 EISCA/持续 patch attribution：DPB-DF、defect prototype bank、WeakREST、training-free human-in-the-loop memory bank correction。命中 DPB-DF、WeakREST 与直接 B-only PatchCore correction；memory correction 报告主动与被动查询无显著差异，缺陷记忆扩展失败。
2. 对 F3 安全适应：self-healing human-in-the-loop PatchCore drift robust；normality shift open world；robust distribution alignment industrial anomaly detection；continual anomaly detection industrial image。
3. 对 F4 规则学习：MVTec LOCO compositional logic few-shot part segmentation；SALAD semantics-aware logical anomaly detection；LogiCo logical structural；VLM logical anomaly detection。
4. 对 F5 行动策略：ActiveInspect GRPO multi-sensor evidence selection；cross-view few-shot industrial anomaly detection Real-IAD。
5. 对 SCAEA/F1：decision theoretic value of information active learning；targeted active learning Bayesian decision making；budgeted online active learning；multi-level active prediction image annotation；active learning biased non-response。
6. 对 F6：label delay online continual learning；online selective classification limited feedback；active learning noisy annotator。

第二轮显著降低了“通用 online memory”“选择最有价值反馈”“可选多传感器动作”“用人类反馈应对漂移”的广义新颖性。未找到文献不能作为空白证明。

## 主要纳入来源

本轮纳入 25 条主要来源（见 CSV）。覆盖：

- 2007–2008 决策感知与多级主动标注的理论先例；
- DPB-DF、WeakREST、PatchCore 人机修正和 MultiADS 等局部定位/持久状态竞争者；
- OWAD、RoDA、SH-HIL-PCF、Continual-MEGA、InCLAD 和 MVTec AD 2 的漂移/连续评估；
- MVTec LOCO、PSAD、SALAD、LogiCo 的组合异常工作；
- Real-IAD、Real-IAD D³、CVAD、ActiveInspect 的多视角/多模态工作；
- 非响应偏差、标签延迟与在线有限反馈来源。

选择标准是它是否改变候选问题的新颖性、理论边界或数据可行性，而不是引用数或标题相关度。来源验证级别详见 CSV。

## 排除或仅作背景

| 作品/类别 | 处理 | 理由 |
|---|---|---|
| 单纯静态无监督异常检测排行榜论文 | 大多不收入 CSV | 可作为实验基线，但不直接检验在线人类反馈、持久状态或候选 novelty |
| 纯合成缺陷方法 | 作为 F4/F2 竞争背景 | 合成异常可提高任务效果，但不能证明人类反馈学习或持续收益 |
| 自动驾驶/医学图像的 active segmentation | 不作为领域先例，只作方法类比 | 标注成本方法可迁移，但传感器、目标损失与工业部署协议不同 |
| 仅网络/时序漂移方法 | 部分纳入 OWAD/在线学习理论 | 对 shift/feedback 形式有启发，不等于视觉质量检验竞争方法 |
| 搜索聚合站、排行榜、论坛和二手新闻 | 不作为原始证据 | 仅用于发现论文；最终以官方论文、出版社、会议、数据集页或作者代码核验 |
| 未能核实摘要/书目或同名不明结果 | 排除或标 UNKNOWN | 不把检索片段拼接成未经验证的主张 |

## 覆盖和访问限制

- DPB-DF 的 DOI、作者、摘要记录和作者官方代码可核对；期刊全文在出版社侧无法访问。本报告对其具体机制作 PARTIALLY VERIFIED，不推断未见的 ablation/限制。
- WeakREST 以完整 arXiv v6 核查方法；最终 TIP 全文未取得，最终版本相对预印本的改动未知。
- HITL Memory Correction、LogiCo、InCLAD、Continual-MEGA、RoDA 为预印本或本轮只能核对预印本；发布状态可能在之后变化。
- ActiveInspect 已读到出版社开放页面摘要/部分正文；作者数据声明为按请求提供，数据不可公开获取。
- CVAD 的 IEEE 记录摘要、期刊、日期和 DOI 可核对，IEEE 页面阻止进一步打开；完整正文/作者列表未能在本次直接页面验证。
- MVTec LOCO、MVTec AD 2、Real-IAD、Real-IAD D³、VisA 的数据能力以官方数据页或原始论文核对。没有下载数据、检查代码包或推断项目本地文件标注内容。
- 本次未做引文网络/数据库的可复现 PRISMA 检索，未逐一搜索所有别名、专利、非英语来源或 2026 年下半年未索引工作。所有新颖性结论保留 UNKNOWN，不可宣称穷尽。

## 下一轮搜索应优先解决

1. 找到并通读 DPB-DF 最终全文及补充材料，核对其 online 事件顺序、反馈授权、类别更新和未来样本隔离。
2. 对 O01 比较已发表人机记忆修正、SH-HIL-PCF、RoDA、OWAD 与选择性标签/在线 bandit 的机制级重叠。
3. 找出存在真实时间/产品/工况元数据的可公开工业视觉数据，或明确未来必须向企业采集的最小反馈集。
4. 检查 O01 所需的随机审计、非响应 propensity、错误标签率与可撤销更新假设是否能在目标部署中观测；若不能，应停止理论保证主张。

**人类审批：PENDING。** 本日志不授权后续研究阶段或实验。
