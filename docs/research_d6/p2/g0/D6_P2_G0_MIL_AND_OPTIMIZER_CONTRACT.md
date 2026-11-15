# LP-D6-P2-G0 — MIL 与优化器契约

**状态：** 运行前提案。没有训练任何头部或优化任何参数。

## 1. E1 完整历史 Logistic 与 Proximal 目标

设 `D_t = D-fit` 的 52 条 binary-valid 记录，加上到 checkpoint `t` 已揭示的 Present/Absent 反馈。令 `n_t=|D_t|`，属性目标 `y∈{0,1}`，权重 `w∈R^d`，截距 `b∈R`：

\[
J_t(w,b)=\frac{1}{n_t}\sum_{(x_i,y_i)\in D_t}
\left[-y_i\log\sigma(s_i)-(1-y_i)\log(1-\sigma(s_i))\right]
+\frac{\lambda}{2}\lVert w\rVert_2^2,\quad s_i=w^Tx_i+b.
\]

`b` 不正则化。Proximal 版本为：

\[
J_t^{prox}(w,b)=J_t(w,b)+\frac{\gamma}{2}\lVert w-w_0\rVert_2^2.
\]

`w0` 是只用 D-fit 52 binary-valid 标签、固定 λ、同一特征和预注册求解器优化得到的系数向量；反馈后永不更新锚点。初始截距 `b0` 仅作为共同初始化/配对状态，不进入 proximal penalty。

### 与 P1-B LogisticRegression 的 C 对应

P1-B 的完整历史 Refit 已注册并核对 `C=1/(alpha*n_binary_training_rows)`，对应平均 log loss 加 `alpha/2 ||w||²`、截距无惩罚。因此本契约取 `lambda=alpha`，每个 checkpoint 的 sklearn 对照映射应为 `C_t=1/(lambda*n_t)`。X 从 `n0=52` 到 `n12=64`；W 从 52 到 63（二元反馈 11 条，其中一个 uncertain event 不更新）。不要让固定 C 在样本数变化时悄悄改变 λ。

### 配置和计算顺序（待注册）

- 固定 `λ=1e-4`；`γ∈{0,1e-4,1e-3}`，即 `{0,λ,10λ}`。其中 `γ=0` 与已有 `B1-Refit, alpha=1e-4` 同一优化问题，应从保存的 P1-B checkpoints 复用，不另外计作新机制。
- 只新增两种 `γ>0` 轨迹，分别跑 W/X；每条 trajectory 记录 cost checkpoint 0–12。X 每个 checkpoint 有一个新的二元样本；W 的 uncertain event 增加成本索引但不改变训练集，状态可复用，不另做同目标数值拟合。
- 新 proximal 优化最多 `2 gamma × (X 13 + W 12)=50` 次 full-history fit。没有 D-dev 参数搜索。
- 交叉检查可选：在 checkpoint `{0,3,6,12}`，以 full-batch GD 或 Newton-CG 独立最小化同一 objective，比较 `w,b,J`；两个正 gamma × 两协议 × 四 checkpoint，最多 16 次附加数值验证。此项只验证优化器，不是新 learner 家族。
- B0 是一份不变的初始化状态 `(w0,b0)`；每个在线方法的 checkpoint-0 必须匹配同一登记权重和截距。不允许把读出方法不同的模型混成一个 B0。

### L-BFGS-B 参数与失败记录提案

使用 SciPy `minimize(method="L-BFGS-B")`，full-batch FP64 objective/gradient；CPU 执行，建议将 BLAS/optimizer 线程固定为 2（与已存 P1 refit 审计相同），不需要 GPU。固定 `ftol=1e-12`、`gtol=1e-8`、`maxiter=10,000`、`maxls=50`、`maxcor=10`。记录 solver 返回码、迭代数、函数调用、objective、`||∇J||∞`、每项正则值、运行时间和完整输入 fingerprint。成功要求返回 success、参数/objective/gradient 全有限，并 `||∇J||∞≤1e-6`；仅由 function tolerance 退出但梯度较大的状态标为 `FTOL_ONLY_NOT_CONVERGED`，不得作为有效结果。达到上限或 line-search 失败时停止该配置并报告；不自动加大容差、更换 solver、改初始化或挑另一 seed。

E1 当前最多 64×1024 个 feature values；单份 FP64 设计矩阵约 0.5 MiB，参数/锚点状态各约 8 KiB（另有 SciPy 工作区）。这不是完整峰值 RSS 测量；保守规划内存上限建议 <1 GiB，实测峰值未知。该容差组合是**建议固定值**，在 D6 目标维度和数据分布上的耗时/成功率尚未实测。验证只使用合法训练历史；future 标签不传给 optimizer。

### 区分截距漂移与表示方向变化

每个 checkpoint 相对初始化记录：`Δb=b_t-b0`、`||w_t-w0||₂`、`cos(w_t,w0)`、`||w_t||/max(||w0||,eps)`、沿 `w0` 平行投影与正交分量范数。同步保留预测前后分数和正/负翻转，但不因这些诊断改阈值或模型。P1-B-R1 已说明 W 现象同时有阈值 crossing 与排序变化，故不能把所有效果归结为截距。

## 2. E2 MIL 的固定数学约束

对视角图像 bag `B={z_p}`，仅含有效空间 patch；共享线性 scorer：

\[
a_p=w^Tz_p+b,\qquad S_\tau(B)=\tau\left[\operatorname{logsumexp}_{p\in B}(a_p/\tau)-\log|B|\right],\qquad q(B)=\sigma(S_\tau(B)).
\]

`M-BCE`：在 D-fit 与已揭示二元 feedback bags 上用平均 bag-level binary cross entropy，加 `λ/2 ||w||²`。`b` 不作 L2。未标注视角、uncertain、N-A 不进入 loss。

`M-Rank` 按 v0.3 已批准的硬 hinge 写作：

\[
J_{rank}=J_{BCE}+\rho L_{rank},\qquad
L_{rank}=\frac{1}{|P_t||N_t|}\sum_{i\in P_t}\sum_{j\in N_t}\max(0,m-S_\tau(B_i)+S_\tau(B_j)).
\]

其中 `P_t`/`N_t` 仅是 D-fit 或已付费/已合法揭示的当前训练视角；每对 positive-negative bags 等权、全笛卡尔积一次，不按 patch 数加权，不重复盲呈现、不按 view augmentation 复制样本。每次反馈后根据合法的当前训练历史重建 pair set；未来标签永不参加。负 bag 可以来自其他异常，不等同于“所有 patch 都正常”。不产生 patch 级伪标签，不加 spatial sparsity/smoothness。

### 预注册标量建议

建议固定 `τ=1.0` logit 单位、`m=1.0`、`ρ∈{0,0.1,1.0}`，最多三种设置且包含 `ρ=0`。`ρ=0` 就是 M-BCE；首要比较可预先定为 `ρ=0.1`，`ρ=1` 只作为已登记敏感性，不以 future score 决定执行或报告。所有的 `τ/m/ρ` 都须负责人批准；不在 D-dev 搜索。

M-BCE/M-Rank 对应配对运行使用完全相同的 patch 输入、bag 划分、初始 `w,b`、seed、λ、事件历史和 checkpoint。最多两个预声明初始化 seed（建议 0、1）；每个 seed 的 M-BCE 与每个 ρ>0 的 M-Rank 共享初始 scorer。seed 不得按 D-dev 结果筛选。

## 3. Hinge、平滑和 optimizer 决策

硬 hinge 在 `m-S_i+S_j=0` 不可微；虽然可指定一个次梯度，但 L-BFGS 的线搜索和收敛判据以平滑局部模型为基础。**不得为了让 L-BFGS 运行而静默换公式。**

**建议给研究负责人的选择：** 若首要目标是严格执行已批准 v0.3，保留硬 hinge，并显式注册一个确定性的 full-batch 次梯度求解方案、零点次梯度约定和收敛容差；对 M-BCE 使用相同的优化流程以保持 MIL 比较一致。若首要目标是让同一 smooth L-BFGS 求解器干净地比较 BCE 与排序项，则可以批准一个单独命名的 `M-Rank-Softplus`：把每个 hinge `max(0,u)` 换成 `κ log(1+exp(u/κ))`，例如预注册 `κ=0.1`；它是已批准硬 hinge 的目标函数变更，不能冒称原 M-Rank，也不能在 G0 自动采纳。

**决定状态：`RESEARCHER_DECISION_REQUIRED`。** 负责人必须确认硬 hinge + nonsmooth solver，或批准 softplus 变体及 κ。两种都未确认前，不实现/运行 MIL。

排序项只基于分数差，公共加性偏置 `b` 在 `S_i-S_j` 中相消；ranking alone 不能识别或校准 bag classifier 的整体偏置。因此保留 BCE 项、固定报告固定阈值预测和 AP；不把 ranking 变好解释成校准变好。

## 4. 存储、开销与不允许事项

- E1 Logistic/Proximal 权重 1024 维 FP64 加截距，每状态约 8 KiB；w0 再约 8 KiB。G-Concat 权重约 16 KiB。SGD bounded replay 最多 12 个 1024 维 FP32 特征（约 48 KiB 加元数据）。Ridge 的 augmented inverse matrix `1025×1025` FP64 约 8.02 MiB。
- E2 frozen features 是主要存储项；两个 backbone 102 图像的 CLS+spatial FP32 约 721.74 MiB，见 backbone readiness 文档。
- 运行开销尚未实测。优化数据只有最多 64 个二元 view labels（D-fit 52 + X 12，W 11）；但 patch MIL 每次目标/梯度要对多个 bags 中全部 patch 求分数，实际 CPU/GPU 时间 UNKNOWN。授权后先做明确批准的小型 smoke/profile；12 GPU-hour 是硬上限，不是本轮许可。
- 仅已有合法属性反馈可用于训练；原始 Real-IAD defect type 绝不可作为标签或输入。无额外 region point、mask、product ID、sample-key 或 future label。
