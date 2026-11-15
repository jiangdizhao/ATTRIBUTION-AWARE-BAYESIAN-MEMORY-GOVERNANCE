# D6-P2-E2B-R1 — 表征与 MIL Headroom

## 观察到的信号

1. **在线反馈的收益主要出现在 W。** W 的 14 条 learner 轨迹都比各自冻结 B0 提高 checkpoint-12 BA，平均差 `+0.1161`；AP 平均差 `+0.0491`，ROC-AUC 平均差 `+0.0670`。W 是同产品组间适应，且只有 4/6 个未来产品有正、负两类。
2. **X 有正向平均 BA，但轨迹并不一致。** 13 条成功到达 checkpoint 12 的配对中，BA 平均差 `+0.1017`，但只有 6 条提升、6 条持平、1 条下降。AP 与 AUC 平均差均约 `+0.03`，各自只有 3 条提升、6 条持平、4 条下降。
3. **MIL headroom 取决于 backbone 与协议。** X 中 DINOv3/M-BCE 的 BA 明显高于 G-Mean，但 AP 低；DINOv2/M-BCE 的 BA 与 G-Mean 接近。W 中 G-Mean 在两种 backbone 上都高于 M-BCE。不能从一个条件外推 MIL 一般优越。
4. **M-Rank-Softplus 未超过 M-BCE。** 同 seed 配对的 24 个未来类别预测全部相同，BA/AP/AUC 聚合无差异；仅原始分数存在小幅差异。
5. **初始化本身存在显著方法/表征差别。** DINOv2 的 MIL 在 X checkpoint 0 的 BA 为 `0.694`，高于 G-Mean 的 `0.500`；DINOv3 两者分别 `0.583` 与 `0.589`。这些是反馈前差别，不能写成在线学习收益。

## 解释边界

- D-fit 仅 54 张、D-dev 仅 48 张且来自 24 个操作性 sample-key；每个协议未来集只有 24 个视角。不能把视角数当作独立物件数。
- D-dev 在来源 defect type 上富集，且 source type 与人工属性仍相关。尤其 D-fit 的 18 个 scratch 来源样本全部为 Present。虽然存在非 scratch 的 Present、good/OK 的 Present 和 CH 的 Absent，但这不足以估计开放世界泛化。
- X 产品隔离仅在已曝光的六个开发产品之间实现：反馈产品为 porcelain_doll、u_block、usb_adaptor；未来产品为 sim_card_set、toothbrush、transistor1。它不是 D-conf，也不是严格未见数据集泛化证据。
- 反馈事件、未来评价、产品与标签均是预注册开发协议；没有独立确认集。显著性检验或复杂统计建模不会弥补这一点。
- 评价标签按视角定义；同一 sample-key 的其他视角没有作为独立 future 评价物。Uncertain/Not applicable 不进入二元拟合或 BA/AP/AUC 正负支持数。

## Headroom 判断

本矩阵提供了“合法属性反馈可以改变未来视角分类”的初步证据，X 的跨产品平均 BA 有正增益，W 的增益更稳定；同时反馈也造成伤害：跨 13/14 learner 的轨迹-视角终点变化中，X 有 10 次错误新增，W 有 13 次错误新增。它不能证明属性学习超越普通在线属性分类、空间机制有效或真正跨产品持久迁移。M-BCE、M-Rank-Softplus 与全局读出的差异混合了表示、初始化、优化目标与很小的产品/样本数。

后续可以把“是否值得继续验证稀疏属性反馈”作为单独研究问题；不能以本轮结果直接宣称新算法 headroom，也不能自动启动 E2-G1。
