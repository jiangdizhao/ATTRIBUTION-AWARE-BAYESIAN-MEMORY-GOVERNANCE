# LP-D6-P2-G0 — 数据与反馈完整性审计

**审计日期：** 2026-11-15
**结论：** `DOCUMENTATION_DISCREPANCY_RESOLVED`；数据仅支持开发性复现，不构成独立确认

## 1. 审计范围和依据

只读核对 D4-Lite/D4B 的首次标注、D5R-P2 V2 annotation queue 与 first annotation、V2 selection/exposure manifest、P1 W/X tape、P1-B 已保存的 evaluator join 与 feedback event log，以及 P0/P1 既有报告。没有读取图像、mask、特征数值或模型状态，没有访问保护数据。审计结果在此仅以汇总形式呈现，逐图像身份、sample-key 对应表和逐条标签不写入 Git 文档。

## 2. 必须解决的 X/W 反馈数差异

P1 Gate0 的旧文字把 X 写作 6 Present / 6 Absent。独立复算结果如下：

| 来源 | X Present / Absent / Uncertain / N-A | W Present / Absent / Uncertain / N-A | 说明 |
|---|---:|---:|---|
| 冻结的 P1 W/X tape 与 V2 首次人工标注连接 | 7 / 5 / 0 / 0 | 5 / 6 / 1 / 0 | 每个 protocol 的 12 个 feedback event；重复呈现按 image 合并，未标注视角不进入。 |
| P1-B `feedback_event_log.csv`（每个 event 去重；按方法/配置的日志副本合并） | 7 / 5 / 0 / 0 | 5 / 6 / 1 / 0 | 每协议 288 行事件副本、12 个唯一事件；所有副本的揭示状态一致，并与 tape join 完全一致。 |
| 历史 P1 Gate0 描述 | 6 / 6 / 0 / 0 | 5 / 6 / 1 / 0 | W 正确；X 的文字与冻结事件及执行日志不符。 |

**判定：** 这是历史描述的 X 计数过期，而不是身份或揭示标签冲突。冻结 tape 的 12 个 X event 与首次人工标签和所有 P1-B event 副本逐项一致，最可靠的事实为 **X=7P/5A**。旧 Gate0 文件未改动；本报告作为更正记录。若未来发现旧文档之外的真实事件身份变化，需另行重新审计。

## 3. 分区、样本角色和标签状态

| 项目 | 独立复核结果 |
|---|---|
| D-fit | 54 条：27 Present、25 Absent、2 Uncertain；52 条进入二元监督。 |
| D-dev V2 | 58 次呈现对应 48 个不同视角；首次标签为 25 Present、19 Absent、4 Uncertain、0 N-A。10 次重复没有生成新 item。 |
| W tape | 12 个 query group / event；24 个 future-evaluation candidate 视角；future 为 14P/7A/3U，21 条二元有效。 |
| X tape | 12 个 query group / event；24 个 future-evaluation candidate 视角；future 为 12P/9A/3U，21 条二元有效。 |
| W 指标定义性 | 4/6 future 产品有 Present 与 Absent；2/6 只有 Present，因此逐产品 BA 未定义。后两类产品 AP 虽可算但全正例，不能代表辨别能力。 |
| X 指标定义性 | 三个 future 产品均有正反例；产品 BA/AP 可算，但样本支持很小。 |

V2 24 组、6 产品各 4 组是操作性分组。五视角来自同一 sample-key 的视角映射可在 ledger 中检查，但真实物理工件独立性没有外部身份凭证，仍为 **INFERRED**。

### 角色分离核验

- W：反馈组与 future-eval 组不重叠；同一产品可同时有不同的反馈组与评价组，故 product overlap 是该协议设计的一部分。
- X：反馈和 future 的 `(product, sample_key)` 组不重叠，反馈产品集合与 future 产品集合也不重叠。
- 各协议中，每个 query event image 唯一，且没有 query image 出现在 future-eval candidate；没有一组 sample-key 同时跨反馈/未来角色。
- tape 每协议有 120 行，是全 exposure ledger 五视角乘角色记录；其中严格只有 12 行 `feedback_query` 和 24 行 `future_eval_candidate` 用于反馈/指标。36 个同组未查询视角被排除，另有 36 个未标注视角明确标为不评价。没有将 72 个未标注视角当成 Absent，也未偷偷纳入 future set。
- Uncertain/N-A 与 Absent 分开。W 的一个 uncertain query 记一个事件成本单位，但不写入 Logistic/Ridge 二元训练历史；未来 uncertain 视角只计 coverage，不进 BA/AP。

## 4. 来源类型混淆和诊断边界

以下是来源元数据与首次人工状态的 evaluator-only 汇总；来源类别从未转换成 abrasion 真值。细粒度的产品×类别行保留在本报告的合法汇总范围内，不公开逐样本映射。

| D-fit 来源类型 | Present | Absent | Uncertain | 总数 |
|---|---:|---:|---:|---:|
| scratch | 18 | 0 | 0 | 18 |
| good | 2 | 21 | 1 | 24 |
| 其他类型 | 7 | 4 | 1 | 12 |

| D-dev 来源类型 | Present | Absent | Uncertain | 总数 |
|---|---:|---:|---:|---:|
| CH | 5 | 1 | 2 | 8 |
| HS | 5 | 0 | 0 | 5 |
| OK | 4 | 15 | 1 | 20 |
| 其他代码 | 11 | 3 | 1 | 15 |

D-dev 已包含一些解耦实例（OK 中有人标 Present、CH 中有人标 Absent/Uncertain），但 HS 五例全部 Present。该富集试点既不能证明类型与属性完全等价，也不能证明完整 Real-IAD 上已经去混淆；产品/材质/视角仍可能成为捷径。D-dev 全在原始 Real-IAD test 划分；角色是团队已接触全部试点标签之后按身份规则回顾性分配。它不是 prospective blind 或 strict zero-shot evidence。

## 5. 近重复、曝光和 D-conf 排除

- P0/P1 的 source path 与 SHA 检查记录 D4-Lite/D4B 共 54 图和 120 个 Real-IAD 曝光图之间无精确文件重叠。
- 复核 D4B 已记录 dHash 元数据：18 张 D4B 图像的 153 对中，4 对距离 ≤4；全部位于 D-fit 同一训练角色，仍应在独立性敏感性分析中成组处理。未访问图像像素。
- 没有 D4-Lite/D4B 与 Real-IAD 之间的统一感知哈希审计；跨源近重复是 UNKNOWN。
- exposure ledger 对 120 张 Real-IAD 原视角全标 `NEW_STUDY_PILOT_EXPOSED`，包括 48 个已标注视角以外的 72 个未标注视角。所有 120 张及已知/可识别近重复均排除于 D-conf。
- D-conf 尚未指定或授权。历史 sealed/HOLD 文件级重叠仍为 UNKNOWN；本审计未打开保护清单或样本。

## 6. 完整性指纹

| 文件/对象 | SHA-256 |
|---|---|
| W/X tape（240 行） | `13561809074aad72e2f13289723ccba8addc4768395480c50d3173abf66e8cc1` |
| V2 selection manifest | `c0def1f598d38ea68c78da028958f7ee030a7bdf21528d5a249fe8f8a6f81b7f` |
| V2 annotation queue（58 行） | `23f3d1084006fd4f1e5f221225ec77862be3facc8713e3bf3592ce448ff3476e` |
| V2 original annotations（58 行） | `1d138ff1a420d6134ad7fc1f2830074cc3d41fda5a8d789a81dd55b0f0e4d9a4` |
| P1-A input manifest（102 行） | `820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63` |
| P1-A feature index（102 条） | `ee0232b29f64f9fd4f10198ba3ef20da89fa85b2083a389b774f6a9723bbb99d` |
| P1-B joined human annotation view（102 行，ignored） | `c814ee3fb8a91ae845c6d7d5d43bdf102b8f30712ff41a0c0fa57d3a0fe96f4b` |
| P1-B event log（576 行，ignored） | `ef35b495d7012514188adecb97eacc412a0f51ee2f41e30661ee8a87f2ca086f` |

受控逐行 join 与 P1-B 输出均在 Git-ignored outputs/pilot 区域；本 G0 未新建行级审计附件，避免复制身份或标签。
