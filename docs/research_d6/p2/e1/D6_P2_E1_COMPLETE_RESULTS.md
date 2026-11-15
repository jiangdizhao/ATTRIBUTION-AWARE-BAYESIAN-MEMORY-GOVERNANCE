# D6-P2-E1 完整结果

**状态：** `BLOCKED_BEFORE_FITTING`  
**结果范围：** 仅报告闸门前已核验的输入与初始化；本文件不把缺失结果填成零或估算值。

## E1 执行覆盖

| 协议 | gamma | 计划来源 | E1 实际状态 |
|---|---:|---|---|
| X | 0 | 复用 P1-B full-history Refit | 未进入 E1 汇总；原 P1-B 结果保持原样 |
| X | 1e-4 | 新 Proximal | 未拟合 |
| X | 1e-3 | 新 Proximal | 未拟合 |
| W | 0 | 复用 P1-B full-history Refit | 未进入 E1 汇总；原 P1-B 结果保持原样 |
| W | 1e-4 | 新 Proximal | 未拟合 |
| W | 1e-3 | 新 Proximal | 未拟合 |

新增 Proximal 拟合：**0/46**。gamma=0 重拟合：**0**。Newton-CG 核验：**0/16**。因此没有 E1 的 gamma 对比、完整新轨迹、harmful/corrected flips、更新时延或参数漂移结果。

## 完整性闸门结果

| 检查项 | 结果 |
|---|---|
| 缓存记录与角色 | PASS：102 条，54 D-fit / 48 D-dev |
| 二元 D-fit 监督 | PASS：52 条，27 Present / 25 Absent；2 Uncertain 排除 |
| W/X 反馈与 future 支持 | PASS：各 12 事件、24 future 视角、21 个二元 future 视角 |
| X 反馈状态 | PASS：7 Present / 5 Absent |
| W 反馈状态 | PASS：5 Present / 6 Absent / 1 Uncertain |
| feedback/future 图像及组隔离 | PASS：无交叉 |
| 既有 P1-B 初始状态及 B0 数组 | PASS：逐值匹配 |
| checkpoint-0 分数复核 | PASS：四种 W/X、B0/ReFit 比较的最大绝对差均为 `3.55e-15` |
| P1-B 15 个原始输出 SHA-256 | PASS：15/15 |
| P1-A 输入快照前后 | PASS：指纹相同 |
| P1-A/P1-B/G0 原始目录 | PASS：审计前后指纹不变 |
| 历史初始参数在注册目标上的梯度 | **FAIL**：`1.7226304473e-5`，门槛 `1e-6` |
| 模型拟合 | 未开始 |

## 保留的 gamma=0 历史参照

下表只转述原 P1-B 已发布的 alpha=1e-4 `B1-Refit` 结果，不是本轮重跑或 E1 新发现：

| 协议 | 原 P1-B BA@0 → BA@12 | 原 P1-B paired AP@0 → AP@12 | 原 P1-B harmful transitions |
|---|---:|---:|---:|
| X | 0.667 → 0.667 | 0.689 → 0.826 | 7 |
| W | 0.688 → 0.812 | 0.750 → 0.875 | 16 |

停止之后对同一份已归档 P1-B gamma=0 日志与初/末状态做了只读汇总；没有拟合、没有重新计算 future 分数。它补足单个 alpha 的翻转与端点参数诊断：

| 协议 | 逐检查点 harmful / corrected transitions | 最终相对 checkpoint 0 独立视角 harmed / corrected | `||w0|| → ||w12||` | `||w12-w0||` | cosine(`w12`,`w0`) | `b12-b0` |
|---|---:|---:|---:|---:|---:|---:|
| X | 7 / 7 | 0 / 0 | 46.504 → 45.547 | 14.660 | 0.9495 | +0.2858 |
| W | 16 / 15 | 4 / 3 | 46.504 → 46.516 | 21.496 | 0.8932 | -0.7371 |

X 有 12 次二元更新，P1-B 日志中的 Refit update latency 合计 191.47 ms（均值 15.96 ms）；W 有 11 次二元更新，合计 152.74 ms（均值 13.89 ms）。这是历史 P1-B 计时，不是本次运行计时。归档模型仅保存 checkpoint 0 和 12 参数，所以没有 P1-B 中间 checkpoint 的权重诊断。

Future 支持来自同一已登记评价集，不随方法变化：

| 协议 | 产品 | Present / Absent / Uncertain | BA 状态 |
|---|---|---:|---|
| X | sim_card_set | 6 / 2 / 0 | 可定义 |
| X | toothbrush | 3 / 2 / 3 | 可定义 |
| X | transistor1 | 3 / 5 / 0 | 可定义 |
| W | porcelain_doll | 4 / 0 / 0 | 缺 Absent |
| W | sim_card_set | 4 / 0 / 0 | 缺 Absent |
| W | toothbrush | 1 / 1 / 2 | 可定义 |
| W | transistor1 | 2 / 2 / 0 | 可定义 |
| W | u_block | 2 / 2 / 0 | 可定义 |
| W | usb_adaptor | 1 / 2 / 1 | 可定义 |

只读汇总的可复核输出为 `archived_gamma0_readonly_diagnostic.json` 与 `archived_gamma0_product_support.csv`，均位于 Git-ignored E1 run 目录。

历史对照可继续追溯，但不满足本轮要求的共同初始化数值兼容闸门，故不能据此宣称已完成 proximal 配对比较。W 的两类 future 产品缺少负例，BA 不定义；X 的三个 future 产品均有正反例，但总体仅 21 个二元视角。这些限制仍然适用。

## 未产生的 E1 科学量

由于停止发生在任何 E1 fitting 之前，gamma=1e-4/1e-3 的 checkpoint 0–12 分数、BA/AP/AUC、更新翻转、权重余弦/范数、截距移动、objective/梯度轨迹和 solver 迭代数均为 **未测量**。不得把它们解释为 0，也不得从 P1-B 结果推断 Proximal 的表现。

历史 D-dev 仍是开发性、已曝光的小样本；不构成独立跨产品确认。D-conf、sealed/HOLD、mask、新图像与 E2 均未访问。
