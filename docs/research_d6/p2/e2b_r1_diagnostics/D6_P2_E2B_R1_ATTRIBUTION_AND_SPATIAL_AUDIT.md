# D6-P2-E2-B-R1-D2 — 在线更新归因与 MIL 空间贡献审计

日期：2026-11-16
审计状态：`D2_PASS`
研究结果状态仍为：`E2B_R1_PARTIAL`；本审计不改变原 R1 结果状态。

## 1. 身份、输入与边界

当前仓库分支/HEAD 为 `stage3-eg-famar-execution` / `cd1d611bac860a5c82b014ffc518a6afa3a0f500`。被审 R1 运行记录的 HEAD 为 `072e11adaffd85cf64b85b4708befe45cef92018`。R1 run identity SHA-256 为 `ea1608c59c7070ace67d9db58d24ca19e77fc306569aa57db607ae9ac40bb10a`，聚合 manifest SHA-256 为 `72e60d1fc22edf2e5928204737faf90aa2d113e70348a39533caf983a5bdb378`；manifest 中 **10/10** 聚合文件哈希匹配。

E2-A1 登记输入为 102 项（54 D-fit、48 D-dev）。输入 manifest、canonical feature manifest 的 SHA-256 分别为 `820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63` 和 `0fca669b4f69ed2071b4f4c0ce8f60728c6bd0e6efe126be580f23037b748133`。DINOv2 与 DINOv3 feature manifest 的 SHA-256 分别为 `3588bab021a9cd8043d663c1d6ce46f366f89f89802fb1ac928dfae3360efddf`、`20e100671344422779d848f7664cdb6290d7fe4cd0e1ed1ea09f108fe0a01887`；缓存项、数组文件及数组内容摘要均逐项通过核对。冻结标注、队列、tape 与 P1 特征清单的 R1 输入哈希也全部匹配。

使用的是已缓存的冻结特征和 R1 参数：DINOv2 ViT-L/14 register（权重摘要 `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`）、DINOv3 ViT-L/16（`dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`）。未读图像像素或原始缺陷 mask；MIL 仅使用 E2-A1 缓存中的 `valid_patch_mask`。标签来自冻结人工属性首标；未使用 source defect type 作为属性真值。

已阅读 E2-B-R1 五份归档报告、上一轮 `D6_P2_E2B_R1_MECHANISM_AUDIT.md`、原始 E2-B `model.py`/`evaluator.py` 与 R1 `r1_core.py`/`run_r1.py`。没有调用旧 runner、没有拟合/优化、没有新增图像或特征，也没有接触 D-conf、sealed/HOLD 数据。

## 2. 已保存模型分数重建与冻结控制

| 核验项 | 实际核验 | 结果 |
|---|---:|---|
| checkpoint-12 完整 learner | 27 条，逐条重建 cp0 与 cp12 | 27/27 通过，0 条阻断 |
| 唯一失败 learner | 真实存在的 cp0–cp9 | 10/10 checkpoint 分数重建；未构造 cp10–12 |
| 冻结 B0 | 28 条 × 13 checkpoints | 364/364 参数与 future score 均逐状态等于 cp0 |
| B0 初态配对 | 28 个 learner/control 对 | cp0 参数与 scores 完全一致；B0 cp0 另行重算通过 |
| 重建分数最大绝对差 | 全部上述有效重建 | `2.66e-15`，远小于冻结 `1e-8` 容差 |

失败轨迹仍是 DINOv2/G-CLS/X，最后有效 checkpoint 为 9；没有将其纳入 27 条 checkpoint-12 完整轨迹统计。原始 R1、E2-A1、E1、P1 代码/结果未写入或覆盖；新脚本、测试、报告及详细结果均为新增文件。

## 3. 在线 BA 增益的权重/截距 Shapley 分解

对每条完整轨迹构造 `F00=(w0,b0)`、`F01=(w0,b12)`、`F10=(w12,b0)`、`F11=(w12,b12)`。Global 用原线性分数；MIL 用原 LogMeanExp（τ=1）。每条轨迹都检查 Shapley 权重项与截距项之和等于 F11−F00；BA、配对 AP、ROC-AUC 的最大求和残差均为 0。截距项对 AP/AUC 的 Shapley 贡献在全部 27 条轨迹上均为 0。

| Protocol | 完整轨迹数 | 平均 BA F00 | 平均 BA F11 | 平均总 BA 增益 | 平均权重 Shapley | 平均截距 Shapley | 权重/截距幅度主导轨迹数 | 方向相反数 | 纠正/有害二元翻转总数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W | 14 | 0.5446 | 0.6607 | +0.1161 | +0.1071 | +0.0089 | 14 / 0 | 1 | 66 / 13 |
| X | 13 | 0.5880 | 0.6897 | +0.1017 | +0.0853 | +0.0165 | 6 / 1（另 6 条幅度相近） | 7 | 42 / 10 |

这里的“主导”按 `|phi_weight|` 与 `|phi_bias|` 直接比较，不除以可能接近零的净增益。W 的 14 条 BA 提升都以权重贡献幅度为主。X 的平均净增益也主要由权重贡献，但逐轨迹更不一致：6 条权重项幅度更大、1 条截距项更大、6 条接近相同；7 条两项方向相反，6 条净 BA 增益在数值精度范围内为零。不能把 X 归纳成单一、稳定的因果机制。

AP/AUC 的变化来自排序（权重）项：单独移动截距不会改变 AP/AUC。BA 则同时受零阈值下的判别权重和整体校准影响。归因不是训练干预，也不证明因果性；其完整轨迹级以及 backbone×方法×protocol×seed 聚合均保存在私有结果中。

## 4. MIL 空间修正项及 patch 权重

逐 bag 计算：

`S_full = b + tau * logmeanexp(u_p/tau)`
`S_mean = b + mean(u_p)`
`Delta_spatial = S_full - S_mean`

每个 bag 均使用自身有效 patch mask 与有效数量 P。共得到 2,432 条 bag-state 记录：D-fit 每个 checkpoint 832 条，future 每个 checkpoint 384 条。重复模型状态使用了同一批图像，以下行数是 bag-state 观测数，不是独立图像/物件数。

| 数据角色 | Checkpoint | Delta 均值 | 中位数 | 标准差 | 最小–最大 |
|---|---:|---:|---:|---:|---:|
| D-fit 二元 bag | 0 | 5.712 | 6.169 | 3.547 | 0.969–11.380 |
| D-fit 二元 bag | 12 | 5.854 | 6.008 | 3.550 | 1.101–13.129 |
| W/X future bag | 0 | 3.431 | 2.913 | 1.625 | 1.252–9.510 |
| W/X future bag | 12 | 4.473 | 3.990 | 2.177 | 1.058–10.233 |

`Delta_spatial` 不是近似常数。future 中 Present 的均值为 cp0 `4.144`、cp12 `5.544`；可信 Absent 分别为 `2.539`、`3.108`；Uncertain 单独保留，分别为 `2.720`、`3.469`。D-fit Present/Absent 的对应均值为 `8.927/2.240`（cp0）与 `9.026/2.427`（cp12）。这些都是重复模型状态下的描述性关联，不是属性定位验证。各 protocol 下产品级 Delta 均值范围也较宽：W 在各配置间约 `1.14–7.80`，X 约 `1.53–7.92`。

同一 future bag 配对 cp0→cp12 后，平均 Delta 增量在 backbone/protocol 配置间约为 `+0.56` 至 `+1.64`；D-fit 的对应均值约为 `+0.08` 至 `+0.21`。空间修正会随模型参数及样本改变。

LogMeanExp 导数权重 `alpha=softmax(u/tau)` 明显非均匀：

| Backbone | Checkpoint | 平均 N_eff/P | 平均 N_eff | Top-1 权重和 | Top-10 权重和 |
|---|---:|---:|---:|---:|---:|
| DINOv2 | 0 | 0.0812 | 83.1 | 0.304 | 0.632 |
| DINOv2 | 12 | 0.0642 | 65.7 | 0.315 | 0.672 |
| DINOv3 | 0 | 0.0517 | 40.6 | 0.369 | 0.688 |
| DINOv3 | 12 | 0.0244 | 19.1 | 0.460 | 0.797 |

DINOv2、DINOv3 的平均有效 patch 数均以每图实际有效 P 归一化；缓存中的典型 P 分别为 1,024 与 784。future Present 的权重也更集中（cp0/cp12 平均 N_eff/P `0.034/0.020`，Top-1 `0.455/0.527`）而 Absent 为 `0.106/0.071`、Top-1 `0.185/0.227`。这说明 scorer 把分数更多集中到部分 patch；alpha 不是训练出的 attention 网络，且没有 abrasion localization ground truth，不能据此声称定位准确。

数值恒等式最大重建误差为 `1.78e-15`，最小 Delta 为 `0.969`（超过仅允许的 `−1e-12` 浮点容差），Jensen 非负约束通过。

## 5. 同一 MIL 参数下的 S_full / S_mean 反事实

S_mean 使用同一 MIL 的训练权重、bias 和未归一化 patch 均值；没有重新拟合，也不是重新归一化、独立训练的 G-Mean baseline。每个 protocol/checkpoint 包含 8 个 MIL 模型状态；W 有 21 个二元 future views、4 个可计算两类指标的产品，X 有 21 个二元 future views、3 个可计算产品。

| Protocol / cp | BA Full / Mean（差） | Paired AP Full / Mean（差） | AUC Full / Mean（差） | 正预测率 Full / Mean | 正负样本对顺序/tie 变化 |
|---|---|---|---|---|---:|
| W / 0 | 0.5625 / 0.5000（+0.0625） | 0.8542 / 0.8542（0） | 0.7500 / 0.8125（−0.0625） | 0.2143 / 0 | 284 / 784 |
| W / 12 | 0.6250 / 0.5000（+0.1250） | 0.8542 / 0.8750（−0.0208） | 0.7500 / 0.8750（−0.1250） | 0.4048 / 0 | 228 / 784 |
| X / 0 | 0.6389 / 0.5000（+0.1389） | 0.9130 / 0.9647（−0.0518） | 0.8500 / 0.9028（−0.0528） | 0.2143 / 0 | 252 / 864 |
| X / 12 | 0.7778 / 0.5000（+0.2778） | 0.9046 / 0.9647（−0.0601） | 0.8222 / 0.9028（−0.0806） | 0.4286 / 0 | 216 / 864 |

四种条件的 S_mean 零阈值正预测率都是 0；完整 S_full 加回正的 Delta 后会产生正预测，BA 因而提高，但 AP/AUC 多数下降或不变。把 Full 替换成 Mean 的全部二元阈值翻转中，纠正 8 次、损害 204 次（按模型状态累加）。这提示 BA 增益主要包含固定阈值下的整体分数校准成分；Delta 虽然样本依赖并能改变排序，但当前排序结果没有显示稳定的 AP/AUC 收益。样本对统计包含 tie 状态变化。

### DINOv3 / X

cp12 的 DINOv3/X MIL：平均 Delta `5.177`，N_eff/P `0.0276`，Top-1 `0.402`；DINOv2/X 对应约 `3.886`、`0.0718`、`0.258`。DINOv3/X 的 Full−Mean BA 差为 `+0.3611`，但 paired AP 差 `−0.0393`、AUC 差 `−0.0556`；正预测率从 Mean 的 0 上升到 Full 的 `0.5714`。M-BCE 与 M-Rank-Softplus 在该比较中的指标相同。因此可以说 DINOv3/X 同时呈现更大空间修正、更集中的 patch 权重和更大的固定阈值 BA 差；不能说这验证了 M-BCE 独有的局部空间机制，或空间聚合改善了排序。

## 6. 对后续研究的含义与限制

- **普通校准足以解释的现象：** 截距是共同加性项，不改变 AP/AUC 排序；S_full 的正空间修正能把原本全负的 S_mean 分数推过固定零阈值，从而提高 BA。故 BA 单独上升不能证明局部辨别力增强。
- **涉及判别权重变化的现象：** W 中的 BA Shapley 几乎全部来自权重；X 的平均也以权重项较大，但逐轨迹有方向抵消和零净增益。普通在线分类器权重更新已经是合理解释，不需引入结构化信用分配才能解释观察结果。
- **可能涉及局部空间证据的现象：** Delta 在样本、标签、产品和 checkpoint 间变化，patch 权重非均匀，S_full/S_mean 改变了部分正负样本排序。这验证了 LogMeanExp 的数学空间依赖性；然而当前 AP/AUC 对照不支持它在这些 future views 上带来稳定排序收益，且没有定位真值。
- **仍完全未测的假设：** 因子级记忆归属、跨属性/产品的信用分配、冲突反馈治理、长期保持与遗忘，以及局部区域属性的可信因果定位均未由此实验检验。当前证据不足以声称复杂 factor-addressable Bayesian memory/ABMG 已有性能或机制优势，也不足以据此停止该研究问题。

## 7. 失败与复现边界

- DINOv2/G-CLS/X 的 cp12 不存在，严格保留至 cp9；不计入 27 条完整轨迹。
- 历史 `E2B_R1_PARTIAL` 状态不变；本报告的 `D2_PASS` 仅表示指定身份、数值、Shapley 和空间恒等式检查通过。
- 本次没有重训、优化、调参、读新图像或原始 mask；没有执行 E2-G1。R1/E2-A1/E1/P1 原文件未修改，也未执行 Git commit/push。
- 逐轨迹、逐产品、逐 bag、标签、完整参数与分数明细仅存放在 Git 忽略的 `outputs/research_d6/p2/e2b_r1_diagnostics/attribution_spatial_v1/`。它们重复使用同一组 D-fit/future 图像和模型状态，不应当作独立样本做显著性推断。

私有运行摘要记录：27/27 完整 learner 通过端点归因；0 条因分数差异阻断；28 个 B0 的 364 个状态不变；失败轨迹 cp0–9 十个分数均重建；32 个 MIL 状态反事实与 2,432 条空间 bag-state 诊断完成。总审计状态：**`D2_PASS`**。
