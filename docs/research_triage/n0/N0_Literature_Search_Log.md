# N0 文献检索日志

**审计 ID：** LASTPAPER-N0-20261031  
**检索/访问日期：** 2026-10-31（Australia/Sydney）  
**检索类型：** EISCA 与 SCAEA 定向竞争性新颖性审计。优先查看原始方法文本、出版社/会议官方记录、arXiv 正文、作者公开实现；不是穷尽性系统综述。

## Repository 与证据边界

- 工作目录：`E:\ATTRIBUTION-AWARE-BAYESIAN-MEMORY-GOVERNANCE`
- Branch：`stage3-eg-famar-execution`
- HEAD：`ad6ae3dd0010e68343e376170b0539515051ecce`
- 初始 `git status --short`：干净（无输出）。
- `python -c "import sys; print(sys.executable)"` 实际为 `D:\anaconda3\python.exe`，与任务历史记录的 `D:\anaconda3\envs\torch\python.exe` 不一致。
- 按任务范围选择性阅读 README、Stage-3 handoff/plan、feedback authority/cost、E2/E4 experiment notes、five-branch structural-credit audit；未通读 Python 实现或数据集。
- 用户给定的项目指标及因果限制作为历史背景，未独立重算。仓库 handoff 指出 Stage-3 mechanisms 未运行/验证，既有 structural-credit 只证明已知来源地址的路由，不是自动 attribution。
- 未使用 GPU、训练模型、访问 Real-IAD 图像/隐藏标签、安装依赖、克隆外部完整仓库、commit 或 push。

## 检索渠道

- 出版社/官方：ScienceDirect、PubMed、IEEE 书目记录、AAAI Proceedings、IJCAI Proceedings、PMLR、TMLR/OpenReview。
- 预印本：arXiv abstract、HTML、PDF。
- 实现：DPB-DF 作者链接的公开 GitHub 与 raw source files。
- 书目/作者研究门户：University of Helsinki research portal（TMLR 最终记录）。

## 英文检索关键词

### EISCA

- `"10.1016/j.engappai.2026.116388"`
- `"Dynamic Patch Bank with a Defect Filter"`
- `"Training-Free Human-in-the-Loop Anomaly Detection via Memory Bank Correction" arxiv`
- `"Industrial Anomaly Detection and Localization using Weakly-Supervised Residual Transformers"`
- `"local embedding density" defect filter DPB-DF patch`
- `"WeakREST" block-wise anomaly detection localization paper`
- `PMLR Attention-based Deep Multiple Instance Learning Ilse Tomczak Welling 2018`
- `online class-conditional prototype learning from weak labels patch-based visual defect detection dynamic memory bank`
- `Multiple instance learning weakly supervised patch localization image level labels`

### SCAEA / 理论

- `"Budgeted Online Active Learning with Expert Advice and Episodic Priors"`
- `decision theoretic active learning expected risk reduction value of information active learning primary paper`
- `Bayesian experimental design expected utility reduction decision risk information value primary source`
- `"Targeted Active Learning for Bayesian Decision-Making" journal venue published`
- `expected free energy Bayesian decision theory active learning information gain expected utility paper`
- `selective labels problem machine learning selective labels paper counterfactual risk evaluation`
- `active learning selective label bias query selection propensity weighting paper`
- `feedback loops selective labels prediction systems ML paper primary source`
- `"Active learning with biased non-response to label requests" authors venue DOI`

## 纳入候选与原因

| 工作 | 方向 | 纳入理由 |
|---|---|---|
| DPB-DF，DOI 10.1016/j.engappai.2026.116388 | EISCA | 最接近“弱/图像标签→局部 defect-patch filter→持久类别 patch bank”；官方代码可精确核验 hard-vote 方法。 |
| Training-Free Human-in-the-Loop Anomaly Detection via Memory Bank Correction，arXiv:2608.17775 | EISCA + SCAEA | 直接研究 binary 部署反馈与 memory edit；包含 passive-vs-active 查询对照和 held-out 评估。 |
| WeakREST，arXiv:2306.03492v6；期刊 DOI 10.1109/TIP.2026.3659337 | EISCA | 弱 image label、unknown blocks、学习型 block anomaly scores 与定位。 |
| Attention-Based Deep Multiple Instance Learning，ICML 2018 | EISCA | bag-level label 下 instance contribution 的基础数学先例。 |
| Selective Supervision，IJCAI 2007 | SCAEA | 直接的 expected future risk reduction 减 annotation cost / VOI 基线。 |
| Targeted Active Learning for Bayesian Decision-Making，TMLR 2024(6) | SCAEA | 直接以 downstream optimal decision 上的预期信息增益选择 query。 |
| Budgeted Online Active Learning with Expert Advice and Episodic Priors，AAAI 2026 | SCAEA | 有限 horizon、顺序 query budget、当前预测器状态、专家建议和流历史 prior。 |
| Active inference and learning，DOI 10.1016/j.neubiorev.2016.06.022；Sajid et al. arXiv:2110.04074 | SCAEA | EFE 的 epistemic/pragmatic 分解及其和信息设计/贝叶斯决策关系。 |
| Lakkaraju et al. KDD 2017；Robinson et al. DMKD 2024 | SCAEA | 查询选择造成的 selective-label 与 biased non-response 问题。 |

## 筛选与排除

- **Memory Pollution in Multi-Product Visual Anomaly Detection (2026)** 是产品级 memory routing 邻近工作。已见预览研究共享 memory 干扰与 routing，但未显示稀疏人工反馈或 feedback-authorized address writes；未纳入 EISCA 3–5 篇核心对手。不能据此说跨产品 attribution 已解决。
- **Online Prototype Learning for Online Continual Learning (ICCV 2023)** 是可用的监督持续学习 baseline。它从有标签流更新 class prototypes，但已检查记录未显示 binary-only 人工反馈或 patch responsibility；只列为 baseline 家族，不列核心竞品。
- 未因标题相似把通用 anomaly detection、continual learning 或 uncertainty sampling 升为最接近论文。各 audit 另列具体 baseline。
- 不主张已穷尽 MIL、弱监督定位、在线 class-conditional prototype 或 query selection 领域。

## 访问失败与核验状态

- 初次直接打开 ScienceDirect DPB-DF 页返回 **403 Forbidden**。出版社搜索结果/页面预览披露摘要、贡献描述和 Introduction；官方 GitHub raw 源码可访问。全文其他章节/消融及截至审计日的最终出版状态为 `PARTIALLY_VERIFIED`。
- DPB-DF publisher record 给出 **2026-11-15** issue date，晚于 2026-10-31 截点。页面没有明确标注 accepted 或 online-first，故记录为“精确状态 UNKNOWN，显示未来期次日期”，不记为已进入该期。
- WeakREST 最终 IEEE 全文未访问。PubMed 核验期刊标题、年份、卷、页码和 DOI；方法按作者完整 arXiv 前身 v6（2025-01-15）检查。最终版差异为 `PARTIALLY_VERIFIED`。
- HITL Memory Correction arXiv v1 正文可访问；方法和结果作为预印本作者主张标注 `FULL_TEXT_VERIFIED`，不是独立复现。
- AAAI 官方页核验最终 proceedings 书目/状态；详细方法读关联 arXiv v1 全文。arXiv PDF 的会议信息含占位符，未当作权威书目。
- 其余来源没有阻断窄范围结论的原文访问失败。

## 待人类审阅的问题

1. 研究计划与潜在论文中，`memory address`、`responsibility`、`source` 和 `factor` 是否分别操作化并保持一致？
2. 用户提供的 Five-Branch F1 与 E4B Binary/History/Oracle 是否来自相同任务、表征和 outer folds？本审计未查原始输出。
3. SCAEA 的 primary estimand 应是固定 threshold BA、cost-weighted risk、recall constraint，还是 utility/minute？
4. query cost 与付费 R 标注时长是否有实测，还是协议假设？
5. 目前是否有足够的非 oracle B-only 重复反馈用于校准 patch/address responsibility，还是只有来源/类型标签支持既有路由结果？
6. DPB-DF 的 future-issue 论文在截点时是否已正式 online-first？最终全文是否超出预览和官方代码披露的内容？

## 解释纪律

`VERIFIED` 只表示原文/代码直接支持某事实，不代表论文作者的宽泛新颖性结论已成立。`INFERRED` 为本审计的比较判断。项目历史指标标注为用户提供，非新验证。未获得全文的来源绝不标为 `FULL_TEXT_VERIFIED`。本轮没有实验，因此所有研究建议仍待人类审核。
