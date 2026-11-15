# D6-P2-E1-R1 匹配基线结果

**状态：`PARTIAL`。** 新初态通过；六个预注册轨迹中五个完整，一个轨迹按数值门槛停止。所有结果仍是已曝光 D-dev 上的小样本探索性 headroom 检查。

## 匹配设置与覆盖

所有配置使用 [数值初态报告](D6_P2_E1_R1_NUMERICAL_INITIALIZATION.md) 中同一 `(w0_new,b0_new)`、`λ=1e-4`、零 logit 阈值和完整合法历史。γ=0 是本轮新跑的 full-history Logistic 对照；旧 P1-B gamma=0 仅作为 `HISTORICAL_REFERENCE_ONLY`。W/X 事件顺序、可见监督、图像/组隔离未改变。X 有 12 个二元更新；W 有 12 个事件，其中 1 个 Uncertain 只计事件、不更新。

| 协议 | γ | 结果 | 成功二元更新 / 预期 | 每次成功更新平均时间 | 成功解最大梯度无穷范数 |
|---|---:|---|---:|---:|---:|
| X | 0 | PASS | 12 / 12 | 16.91 ms | 3.06e-7 |
| X | 1e-4 | PASS | 12 / 12 | 14.08 ms | 3.69e-7 |
| X | 1e-3 | `FTOL_ONLY_NOT_CONVERGED` | 7 / 12 | 6.24 ms（7 个成功更新） | 2.43e-7（成功解） |
| W | 0 | PASS | 11 / 11 | 21.40 ms | 2.47e-7 |
| W | 1e-4 | PASS | 11 / 11 | 14.09 ms | 7.85e-7 |
| W | 1e-3 | PASS | 11 / 11 | 8.29 ms | 2.12e-7 |

X/γ=1e-3 在事件 8（`X-FB-008`）的更新被接受门槛拒绝：L-BFGS-B 报 relative function reduction，梯度无穷范数 `1.2826408917e-6`，超过 `1e-6`。该失败尝试的状态前后摘要一致；配置立即停止，事件 9–12 不再进入该配置。没有重试、放宽容差、更换优化器或修改 λ。其最后完整评分点为 checkpoint 7。

## 最后可用 checkpoint

BA、AP、AUC 是预注册固定 future set 的产品宏平均；AP 使用成对类别产品。W 的 porcelain_doll 与 sim_card_set 只有 Present，BA 未定义，不补零；AP 为退化全正例值。

| 协议 | γ | 最后 checkpoint | BA | Paired AP | AUC | 正预测比例 | 相对 cp0 净 harmful / corrected |
|---|---:|---:|---:|---:|---:|---:|---:|
| X | 0 | 12 | 0.6667 | 0.8259 | 0.6667 | 0.9048 | 0 / 0 |
| X | 1e-4 | 12 | 0.6667 | 0.7796 | 0.6778 | 0.9048 | 0 / 0 |
| X | 1e-3 | 7 | 0.6667 | 0.6889 | 0.6111 | 0.9048 | 0 / 0 |
| W | 0 | 12 | 0.8125 | 0.8750 | 0.7500 | 0.5714 | 4 / 3 |
| W | 1e-4 | 12 | 0.8125 | 0.8750 | 0.7500 | 0.5714 | 4 / 3 |
| W | 1e-3 | 12 | 0.7500 | 0.8750 | 0.7500 | 0.6190 | 3 / 1 |

X 的主诊断中，完整 gamma=1e-4 与 gamma=0 的 BA 相同，paired AP 低 `0.0463`，AUC 高 `0.0111`；γ=1e-3 轨迹未完成，不能给 checkpoint-12 结论。W 是次要 same-product 诊断：γ=1e-4 与 gamma=0 的 checkpoint-12 BA/AP/AUC 相同；γ=1e-3 的 BA 低 `0.0625`。这些小样本描述不能证明 Proximal 机制优势或劣势。

## future 支持与数据边界

| 协议 | 产品 | Present / Absent / Uncertain | BA |
|---|---|---:|---|
| X | sim_card_set | 6 / 2 / 0 | 可定义 |
| X | toothbrush | 3 / 2 / 3 | 可定义 |
| X | transistor1 | 3 / 5 / 0 | 可定义 |
| W | porcelain_doll | 4 / 0 / 0 | 未定义 |
| W | sim_card_set | 4 / 0 / 0 | 未定义 |
| W | toothbrush | 1 / 1 / 2 | 可定义 |
| W | transistor1 | 2 / 2 / 0 | 可定义 |
| W | u_block | 2 / 2 / 0 | 可定义 |
| W | usb_adaptor | 1 / 2 / 1 | 可定义 |

这 24 个 sample-key 是操作性分组，不是已证明相互独立的物理对象。D-dev 已曝光，不能作为独立泛化确认。缺陷类型未输入模型，也没有把 defect type 转作属性标签。

## 历史 P1-B 参照（不作为匹配对照）

在 gamma=0 与归档 P1-B B1-Refit 的 26 个共同 W/X checkpoint 上，BA 和 paired AP 数值均相同；原始分数仍有小差异，最大绝对分差 `0.0026295`（W checkpoint 2）。因此旧曲线只标为历史参照。P1-B `converged=True` 代表 `n_iter < max_iter`，并不等价于本轮的梯度验收。

逐 checkpoint 数据、产品指标、逐事件记录和参数状态位于 Git-ignored `outputs/research_d6/p2/e1/run_20261115_r1_01/`：`checkpoint_metrics.csv`、`product_metrics.csv`、`feedback_event_log.csv`、`future_predictions.csv`、`model_states.npz`。失败尝试与额外调用数的核算见 `run_accounting_reconciliation.json`。这些本地逐样本输出未复制进受版本控制文档。
