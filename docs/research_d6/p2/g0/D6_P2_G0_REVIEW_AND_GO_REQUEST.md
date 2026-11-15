# LP-D6-P2-G0 — 审阅摘要与执行申请

**日期：** 2026-11-15
**G0 状态：** `DOCUMENTATION_READY`
**E1/E2 状态：** `NOT_AUTHORIZED`
**DINOv3 状态：** `E2_BACKBONE_NOT_READY`

## 一页摘要

G0 已把 D6-P2 v0.3 写成可独立审核的执行草案。数据审计确认历史反馈事件身份一致；唯一发现是旧 Gate0 对 X 的文字计数过期。冻结 tape、first annotation 和所有 P1-B 日志副本均为 X=7P/5A、W=5P/6A/1U。P1-A 特征索引、P1-B 输出与 102 项身份 join 的已有 fingerprint 可追溯。

现有 D-dev 仍是 24 个操作性 sample-key 组、48 个视角，物理独立性未证实，标签已被研究组看过。W 有两类 BA 不可定义的产品；X 支持三产品 BA，但总计只有 21 个二元未来视角。D-fit 的 scratch 和 good 来源标签与人工属性高度关联；D-dev 有有限解耦实例但不是完整去混淆证据。故下一步若获批也仅为探索性 headroom，不构成独立产品泛化或确认。

DINOv2 checkpoint 和 P1 CLS cache 可追溯，但 E2 要 448 全视场及空间 tokens，必须重新处理已授权的 102 个源图像。DINOv3 的精确本地源码/权重未找到，且当前 HF Transformers 版本不满足官方 README 所述 4.56.0 支持下限；license 也须独立审阅。Milestone 不满足双 backbone E2 执行条件。MIL 排序项保留已批准硬 hinge，求解器和是否另批 smooth 变体仍待负责人决策。

## 建议的有界实验矩阵（仅估算）

| 阶段 | 方法与范围 | 估算/已知工作量 | 本 G0 是否运行 |
|---|---|---:|---|
| E1 基线复用 | P1-B B0、SGD+Replay、完整 Refit、Recursive Ridge；同 DINOv2 CLS、W/X 和原 evaluator | 只复用已封存结果；不重跑 | 否 |
| E1 新增 Proximal | λ=1e-4；γ={1e-4,1e-3}；γ=0 复用对应 Refit；W/X | 50 次 full-history fit；可选 GD/Newton-CG 交叉核对最多 16 次 | 否 |
| E2 特征 | DINOv2 与 DINOv3，D-fit 54 + D-dev 48；448 letterbox；CLS+spatial patch | 204 个 backbone-image 编码记录；FP32 缓存约 721.74 MiB；GPU 用时 UNKNOWN | 否 |
| E2 主矩阵 | 两 backbone × W/X × 三 global Logistic readouts，各 1 条确定性轨迹；MIL BCE/Rank 每族最多 2 paired seeds，主要 ρ=0.1 | 每条轨迹 13 个 X / 12 个 W 成本 checkpoint；约 350 次 full-history fitting solve（global 150 + MIL 200），加最多 20 个匹配初始/无更新评估状态 | 否 |
| E2 敏感性 | 额外 ρ=1 M-Rank，两 seeds，W/X、两 backbone | 再加最多 8 条轨迹、200 次 solve | 仅列候选，不作为主矩阵默认执行 |

计数按每个二元 feedback checkpoint 都重优化一次计算；W 的 uncertain event 只增加成本 checkpoint，不重复求解相同训练集。E2 的 solver/runtime 尚无实测，以上只作上界式 planning count。若把 G 家族也加 seed、把三种 ρ 全纳入主矩阵，或重跑 P1-B，矩阵会膨胀；建议先固定上述精简主矩阵、只保留一组 ρ=1 敏感性。12 GPU 小时和不超过两 backbone 是预算上限，不代表本次授权消费。新人工标签预算 ≤8 小时是计划边界，本次没有申请或读取新标注包。

## 运行前必须由负责人逐项冻结

| 决策项 | G0 提案 | 当前状态 |
|---|---|---|
| E1 λ / γ | λ=1e-4；γ={0,1e-4,1e-3}；中间项复用 P1-B Refit | 待批准 |
| L-BFGS-B | SciPy；ftol 1e-12、gtol 1e-8、maxiter 10,000、maxls 50、maxcor 10；梯度阈值和失败不自动修复 | 待批准；未做适配基准 |
| M-Rank | 保留批准的硬 hinge，确定 nonsmooth solver；或另行批准 M-Rank-Softplus(κ=0.1) | `RESEARCHER_DECISION_REQUIRED` |
| MIL 标量 | τ=1、margin=1、ρ={0,0.1,1}；0.1 主、1 敏感性；seed 0/1 配对 | 待批准；不看 future 调参 |
| E2 输入 | 448 全视场、保持比例 letterbox；均值色 pad；patch-center 有效规则；FP32 特征 | 待批准并需模型 smoke |
| DINOv3 | 取得/核验本地官方源与全 checkpoint SHA，确认 license 与环境 | 未就绪；未下载/安装 |
| 资源与数据 | 只使用历史授权 102 图；预算硬限制 12 GPU 小时；不访问新图像/labels | 将来单独执行授权 |

任何负责人选择均不能打开 D-conf、sealed/HOLD、Probe、Mint/PCB 或未选视角；不解除已有 HOLD。

计划中可选的新开发标注包（最多 64 组/128 个新标签、8 个产品、8 人工标注小时）不在本次授权内；G0 未查找、枚举或访问该包。若之后需要，须另行申请。

## 建议的唯一下一步

**请研究负责人审阅并在上表一次性登记 E1 λ/γ、MIL hinge/softplus、E2 输入与 backbone 路线后，再发出单独的 E1/E2 执行授权。** 在负责人作出该决定前维持 `NOT_AUTHORIZED`，尤其不开始下载/安装 DINOv3 或运行任何 smoke。若希望先做 E1-only，须明确将其作为另一项有界授权提交；本 G0 不自动拆分或启动实验。

## 执行记录与文件

- 起始 HEAD：`faffeb350bbcc7d1a7ce32d6e444ae7b40de10d1`，branch `stage3-eg-famar-execution`。
- 原始未跟踪 researcher DOCX、`docs/research_triage/h1/` 和 `research_pilot/` 在任务开始前已存在；本次未修改。
- 本次新建且只新建以下五份受控协议文件：
  1. `D6_P2_G0_PROTOCOL_REGISTRATION_DRAFT.md`
  2. `D6_P2_G0_DATA_AND_FEEDBACK_AUDIT.md`
  3. `D6_P2_G0_BACKBONE_AND_RUNTIME_READINESS.md`
  4. `D6_P2_G0_MIL_AND_OPTIMIZER_CONTRACT.md`
  5. `D6_P2_G0_REVIEW_AND_GO_REQUEST.md`
- 未生成 ignored 逐样本交叉表或新的 outputs 文件；受控行级资料沿用现有 ignored artifacts。
- 未改任何 P1 文件、原标注、tape、历史输出、算法文件；未提交/推送 Git。
