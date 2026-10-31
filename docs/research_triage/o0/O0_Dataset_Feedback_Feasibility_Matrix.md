# O0 — Shortlist 数据与反馈可行性矩阵

**范围：** 仅映射两个有条件候选：O01 安全/可撤销适配；O06 组合规则反馈。没有扫描或下载数据集。事实来自数据集主页/原始论文；任务划分与反馈方案是研究设计，不代表数据自带此能力。

## 数据集事实

| 数据集 | 已核实公开数据/标注 | 多视角/模态 | 逻辑/组合信息 | 未核实/缺失 |
|---|---|---|---|---|
| MVTec AD 2 | 8 个场景、8,000+ 高分辨率图；每场景 defect-free train/validation；公开测试部分的异常有 pixel-precise GT；第二测试部分 GT non-public，只能官方 evaluation server 评分。[官方页](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) | 单视图 RGB | 非逻辑专用 | defect-type、point/box/scribble/rule labels 未在官方说明中提供；无部署时间/操作员事件流 |
| Real-IAD | 30 产品类别；image-level normal/anomaly；异常有 pixel masks；论文说明有多 defect types。标签由人工确认及 mask annotation。[CVPR 2024 原文](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Real-IAD_A_Real-World_Multi-View_Dataset_for_Benchmarking_Versatile_Industrial_Anomaly_CVPR_2024_paper.pdf) | 同一物件 5 个同步 camera views，不是 5 个独立样本 | 有缺件/异物等类别，不是通用 rule correction data | sequence timestamp、human query history、label cost、point/box/rule feedback 未核实/不提供 |
| VisA | 10,821 images、12 objects，image 与 pixel-level labels。[原论文](https://arxiv.org/abs/2207.14315) | 单视图 RGB | 不是逻辑专用 | 原文摘要没有确证逐图 defect-type/point/box/rule 标注，按 UNKNOWN 处理 |
| MVTec LOCO AD | 3,644 images、5 个产品类别；结构+逻辑异常；每个异常区域像素 GT；官方页公开类别正常性约束示例。[官方页](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) | 单视图 RGB | 是，正常性含数量、位置、组合、对应关系；annotation 是 benchmark GT/rules，不是部署中专家逐次 correction | 无多视角、在线顺序、操作员 feedback/成本；逐图 rule explanation label 未核实 |
| Real-IAD D³ | 8,450 samples、20 categories、69 defect types；官方论文说明 pixel-level accurate annotation。使用权限按官方 Hugging Face access 条件。[CVPR 2025 项目页](https://realiad4ad.github.io/Real-IAD_D3/) | 同步 RGB、photometric-stereo pseudo-3D、point cloud | 多模态几何，不是 logical-rule benchmark | 人工 query/action logs、成本和在线时序未核实 |

**监督权威分类：**

- **公开 evaluator GT：** 数据集提供的 image label、mask、category/type（仅在源明确支持时使用）。
- **Oracle/evaluator-only：** 未查询样本的 GT、source/type/mask、private AD2 test 指标。任何候选 policy、controller、memory update、query selection 均不得读取。
- **模拟付费人类反馈：** 从 public GT 生成 Binary/Type/Mask/point/box feedback 并给定假设成本；必须明确它是 simulation，不能证明真实人时/错误率。
- **实际人类 annotation：** O06 的逐配置规则/例外；O01 的实际响应概率、复核延迟、标注错误与分钟成本。这些信息公开数据集没有。

## O01 — 选择性反馈下安全/可撤销适配

| 要素 | 可行性判断 |
|---|---|
| 首选数据 | MVTec AD 2 用于正常训练与光照条件变化；Real-IAD 用于多类产品、缺陷类型与像素 mask evaluator。VisA 可做第二个静态 cross-dataset sanity check。 |
| defect category / image labels | MVTec AD2 normal/anomaly 已核实；Real-IAD image labels 和缺陷类型已由论文核实；VisA image labels 已核实。 |
| anomaly masks | AD2 公开测试子集有 pixel GT，private split 不公开；Real-IAD 异常有人工 pixel masks；VisA 有 pixel labels。 |
| point/box/scribble/rule | 上述数据页/原文未提供部署反馈协议；Point/box 可从公开 masks 派生为模拟标签，但必须标为 derived simulation。scribble/rule 不可默认存在。 |
| 单视图/多视图 | AD2/VisA 单视图；Real-IAD 的五视图属于同一物件且同步采集，必须 group split。 |
| shift 是否真实 | AD2 有训练未必包含的 lighting conditions；这是测试分布差异，不是部署时间记录。Real-IAD 跨类别/材质差异不自动等于同一生产线时间漂移。其余 temporal drift 需显式设计/模拟。 |
| sequential/online | 公共文件可人为排序做 prequential replay；无可验证生产时间戳或真实发生次序，因此只能称模拟 stream。 |
| leakage 风险 | 将 Real-IAD 同一对象的五张图分到不同 fold/query/sentinel 会泄漏物件身份；AD2 同一场景/条件也需先按源文件 split，禁止用 private GT 调序/调参。 |
| feedback/cost | Binary 可从 queried image label 模拟；type/mask 可模拟 rich tier。真实 operator response propensity、延迟、错误与分钟成本需实际标注。 |
| held-out confirmation | Real-IAD→VisA/AD2 或反向 source-to-target 做 external confirmation；domain/scorer 差异需固定/报告，不得把类别身份当作自由可见信息。AD2 private test 只能官方 evaluator。 |
| 结论 | 离线 falsifier 可在现有公开图像上模拟；支持安全保证/工业结论的人类反馈数据仍缺。公共数据本身无法检验真实 feedback selection。 |

## O06 — 反馈学习跨配置逻辑规则

| 要素 | 可行性判断 |
|---|---|
| 首选数据 | MVTec LOCO AD，用来验证 logical/structural 类别、异常 masks 和类别层正常约束；Real-IAD 仅可做缺件/异物的邻近 transfer，不可冒充逻辑规则数据。 |
| defect category / image labels | LOCO 有 normal/logical/structural test group；Real-IAD 有产品/缺陷 type；这些是 benchmark 标签，不是 query history。 |
| masks | LOCO 异常区域像素 GT；用于定位评价，不代表逻辑因果组件标注。Real-IAD 有像素 mask。 |
| point/box/scribble/rule | LOCO 官方页面给类别层正常性规则文本示例；没有核实任意逐图的人工 rule feedback、例外 label、point/box/scribble。 |
| 单/多视角与组合覆盖 | LOCO 单视角、五类产品；同产品里的变体有限，不能假设覆盖未见配置。Real-IAD 多视图可检表面缺陷，但不是明确计数/配对 benchmark。 |
| distribution shift | LOCO 不是随时间或规则修订数据；需要按合法配置设计 held-out shift，若 dataset 缺少新配置，则必须采集数据。 |
| sequential/online | 可构造每条规则先验、纠正、后续新配置的模拟事件；仅以同一测试标签顺序不能作为在线学习证据。 |
| leakage 风险 | 若由 public logical labels/GT masks 推导出测试 rule，再提供给 learner，就把 evaluator oracle 当 feedback；需配置级留出并冻结规则词表。 |
| feedback/cost | Binary 与 GT 逻辑标签可模拟；自然语言规则/例外的成本与歧义只能由实际 inspectors 估计。需同时记录是否能回答、是否需重复查看。 |
| held-out confirmation | LOCO 只能验证初始 benchmark；Real-IAD、AD2、VisA 没有经核实的 logical-rule labels，不能做同任务独立确认。需一组可更换配置的实拍/渲染物件和规则标注。 |
| 最小数据采集提议 | 先与一位质量工程师定义 3 个具备合法变体的物件、每类至少 4 种配置/规则、真实/人为规则例外、图像/组件真值和逐条 feedback 时间；再由独立检查员复标一部分以估计歧义。规模只是设计起点，需另行审批。 |
| 结论 | Public LOCO 可做静态 baseline sanity check；对于“人类教会规则并迁移”这一核心主张，最小真实规则反馈数据是必要条件。 |

## 数据获取与评估门槛

1. 本任务只核对公开网页和论文描述，不下载、不扫描数据，不访问隐藏标签。
2. 所有随机 stream、synthetic corruption、feedback noise 和 paid cost 都需在后续 protocol 中明示，不许写成“来自数据集”。
3. 训练、query、support、never-updated sentinel 的划分须在对象/产品 group 层面执行；同一 object 的多视角不能跨组。
4. O01 若用 AD2 private test，只能通过官方评估服务器获得分数；不得访问或推断其隐藏真值。
5. 若无法获得真实人类响应概率/规则反馈，结论须限于仿真而不能推断生产安全保证。

**状态：** 数据文件未读取；O01/O06 的评估仅为文档级可行性评估；human approval PENDING。
