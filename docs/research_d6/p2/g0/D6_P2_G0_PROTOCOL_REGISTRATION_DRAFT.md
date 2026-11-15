# LP-D6-P2-G0 — 协议注册草案

**版本：** G0 / D6-P2 v0.3 运行前注册
**准备日期：** 2026-11-15
**研究方向：** A_CONTINUE_FEASIBILITY
**状态：** 供研究负责人审阅；不授权 E1/E2 执行

## 1. 目的、状态词与边界

本草案把已批准的 D6-P2 v0.3 拆成可复核的执行契约。它不改变研究方向、既有 D6-P1 结果或批准的总体模型族。

本 gate 不扩展至 A1/A2/A3 或 ABMG；B3 Context-aware Region Prototype 与 CB2M 不属于 v0.3 的首批 E1/E2 方法族，仍延后。

| 标记 | 含义 |
|---|---|
| **VERIFIED** | 由本地已授权记录、报告或只读运行环境检查直接核对。 |
| **PROPOSED** | 为下一次实验注册而提出，尚未由研究负责人冻结。 |
| **UNKNOWN** | 当前证据无法确定。 |
| **RESEARCHER_DECISION_REQUIRED** | 运行前必须由研究负责人明确选择或批准。 |

本轮只读了已授权 CSV、历史报告、特征来源清单、代码/依赖元数据和硬件状态；没有读取图像或 mask，没有加载权重，没有推理、训练、更新、下载、安装或创建新人工标签。D-conf、sealed/HOLD、Probe、Mint/PCB 及受保护数据均未访问。

## 2. 研究对象和数据边界

- 属性是图像视角级 `surface_abrasion_visible`，定义沿用已冻结的 Attribute Definition v1.1。
- **VERIFIED：** D-fit 54 张已标注图像；52 条为二元可评估标签（27 Present、25 Absent），另有 2 条 Uncertain。D-fit 物理对象身份未知。
- **VERIFIED：** D-dev 是 48 个已标注 Real-IAD 视角，属于 24 个注册的 `(product, sample_key)` 操作组，6 个产品各 4 组；不能把这些组称为已证明互相独立的物理对象。
- 只使用首条人工标注；盲重复不增加样本、反馈事件或训练权重。`unobservable_or_uncertain`、`not_applicable`、未标注均从二元目标、二元损失和 BA/AP 中排除，但在覆盖率中分别保留。
- Real-IAD 缺陷码、正常/异常字段、产品、视角、VLM 标签、mask 和未来标签均不提供给学习器。来源字段仅可由 evaluator 在预测冻结后用于诊断。
- 所有 54 张 D-fit、Real-IAD 已曝光的 120 张图像及可识别近重复都排除于未来 D-conf。D-conf 未定义、未访问、未授权；历史 sealed/HOLD 文件级重叠仍为 UNKNOWN。

## 3. 固定的反馈和评价顺序

保持 P1 已冻结的 W/X tape、每个协议 12 个事件和原有角色，不重新抽样。

1. 每个反馈事件先对查询视角预测并保存分数、阈值输出和状态摘要。
2. 再揭示该事件已有的合法人工属性状态。
3. Present/Absent 才进入二元训练历史；Uncertain/N-A 仍消耗一个历史事件成本单位，但不产生二元更新。
4. 更新后对固定未来集评分；未来标签仅供离线 evaluator，永不影响查询、更新、超参数、模型或初始化选择。
5. 一个反馈样本所属的整个 sample-key 组不进入同一协议的独立未来评价。

**X 为首要开发诊断：** 反馈产品与未来产品分离；12 个反馈事件实际为 7 Present / 5 Absent；未来 24 个候选视角中 21 个为二元有效标签，3 个 Uncertain，三个未来产品均有正反例。

**W 为次要诊断：** 同产品内按组分开；12 个反馈事件为 5 Present / 6 Absent / 1 Uncertain；未来 24 个候选视角中 14 Present / 7 Absent / 3 Uncertain，6 个产品中 4 个产品有两类标签，2 个产品只有 Present，不能定义 BA。

以上 W/X 均是回顾性开发切分，不是盲测或严格未见产品测试。

## 4. E1 — 学习机制 headroom

**VERIFIED** P1-B 已在同一全局 DINOv2 CLS 特征、W/X tape 和 evaluator 下运行 B0、SGD+Replay、完整历史 Logistic Refit、Recursive Ridge。P1-B/R1 原输出只作为既有控制，不重跑。

**PROPOSED** 追加一个 `Proximal-LBFGS` 历史保持族：

\[
J_t(w,b)=\frac{1}{n_t}\sum_{i\in D_{fit}\cup F_{1:t}^{bin}}\ell_{log}(y_i,w^Tx_i+b)+\frac{\lambda}{2}\lVert w\rVert_2^2
\]

\[
J_t^{prox}(w,b)=J_t(w,b)+\frac{\gamma}{2}\lVert w-w_0\rVert_2^2.
\]

其中 `w0` 只用 52 条 D-fit 二元标签训练；锚点固定，不随反馈漂移。偏置 `b` 不作锚定且不作 L2 正则。`γ=0` 与相同 λ 的完整历史 Refit 是同一目标，应复用现有 Refit 作为对照。

待注册参数建议：固定 `λ=1e-4`，`γ ∈ {0, 1e-4, 1e-3}`。这复用 P1-B 已声明的 Logistic 正则强度候选中间值，不从 D-dev 未来表现挑选。该选择仍需负责人确认。优化器草案、样本数和失败处理见 [MIL 与优化器契约](D6_P2_G0_MIL_AND_OPTIMIZER_CONTRACT.md)。

## 5. E2 — 表征与 MIL headroom

两个冻结特征提取器：DINOv2 ViT-L/14 Register 与 DINOv3 ViT-L/16 web-pretrained。五种表征/读出族：

| 族 | 输入读出 | 头部 |
|---|---|---|
| G-CLS | 全局 CLS | Logistic |
| G-Mean | 有效 patch 的均值 | Logistic |
| G-Concat | CLS 与 patch 均值拼接 | Logistic |
| M-BCE | patch bag | 线性 patch scorer + log-mean-exp bag score + bag BCE |
| M-Rank | 与 M-BCE 相同的 patch 输入与 scorer | BCE 加已批准的正负 bag hinge ranking 项；平滑替代尚未批准 |

全都使用相同 D-fit、合法反馈历史、W/X 事件顺序和未来集。没有新增 mask、defect type 或点位监督；属性正例的所有 patch 不可自动标成正 patch。DINOv2 的 P1 block-18 CLS/392 crop 缓存保持原样，E2 使用独立 448 输入缓存。具体预处理、token、缓存和模型就绪情况见 [Backbone 与运行环境](D6_P2_G0_BACKBONE_AND_RUNTIME_READINESS.md)。

## 6. 评价和比较顺序

- 主要：X 协议 checkpoint 12 的产品宏平均 Balanced Accuracy，相对各自匹配 checkpoint-0 冻结 B0 的变化。
- 次要：产品宏平均 AP、逐产品正负样本数、ROC-AUC 分数诊断、全部 checkpoint 轨迹、正确→错误与错误→正确翻转、正预测率、参数/截距变化、运行时间与状态大小。
- 单独报告 checkpoint 0 表征性能和在线增益。W 保留作次要 same-product 诊断，不把未定义 BA 补零，也不重抽有利切分。
- 比较分层：同表征换更新机制；同优化器换全局/空间表征；同 patch scorer/聚合器换 bag loss；等条件换 backbone。任何不同时改变两项以上组件的结论不得写成单因素因果效应。
- 不用 D-dev 未来标签选择参数、阈值、checkpoint、模型变体、MIL seed 或 winner。

## 7. 已知风险与执行闸门

P1-B/R1 显示 W 的 SGD 在 checkpoint 0 的 21 个二元视角全部判为 Present，前三个事件后全部转为 Absent，同时产生明显分数排序反转；X 的 AP 增益主要由 toothbrush 一类贡献，BA 未变。D-fit 的 18 条 scratch 来源记录全部为人工 Present，24 条 good 来源记录中 21 条 Absent；D-dev 有部分解耦但仍是富集小样本。这些事实要求把 E1/E2 定位为 headroom 与失败机制筛查，不代表复杂模型有优势。

**G0 建议状态：`DOCUMENTATION_READY`。** 这仅表示可供独立审阅；E1/E2 仍 `NOT_AUTHORIZED`。DINOv3 未就绪，MIL hinge/softplus 及 E1 参数须研究负责人签字选择。见 [Review and GO request](D6_P2_G0_REVIEW_AND_GO_REQUEST.md)。

### 依据

- 本地计划：`docs/Last_Paper_Four_Month_Research_Execution_Plan_2026-11_to_2027-02_v1_1.md`
- D6-P0/P1/P1-A/P1-B/P1-B-R1 审核文件：同目录 `docs/research_d6/`
- 官方模型资料：[DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)、[DINOv3 README](https://github.com/facebookresearch/dinov3/blob/main/README.md)、[DINOv3 ViT backbone definitions](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py)、[DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
