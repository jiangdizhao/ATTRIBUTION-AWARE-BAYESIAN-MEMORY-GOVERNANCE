# D6-P2-E2B-R1 — 执行与数值审计

日期：2026-11-16  
运行：`r1_run_20261116_01`  
结果：`E2B_R1_PARTIAL`

## 执行范围

本轮执行已注册的 28 条 learner 轨迹及其 28 条冻结 B0 控制。模型、损失、优化器、特征、反馈 tape、种子、阈值和评价规则均沿用注册版本。没有启动 E2-G1，没有训练 A1/A2/A3，没有读取新图像或 mask，也没有提取新特征或加载 DINO 权重。

运行身份为分支 `stage3-eg-famar-execution`、HEAD `072e11adaffd85cf64b85b4708befe45cef92018`。固定解释器为 `D:\anaconda3\envs\torch\python.exe`；PyTorch 环境下强制 CPU、最多两个线程。完整结果保存在 Git-ignored 的 `outputs/research_d6/p2/e2b/r1_run_20261116_01/`。

## 训练前验证与输入身份

- 原 15 项数值测试与 7 项 R1 恢复/身份测试合计 **22/22 通过**；日志中有原命令输出。
- 102 项冻结特征缓存身份通过；D-fit 54 项、D-dev 48 项。特征缓存、源输入清单及 W/X tape 的 SHA 与注册身份相符。
- tape 校验通过：每个协议 12 个反馈事件、24 个未来视角；反馈与未来图像及 sample-key 不交叉；X 的反馈产品与未来产品不交叉。
- `future_labels_passed_to_learner=false`，`source_metadata_used_by_learner=false`。缺陷类型只用于 evaluator-only 的聚合诊断。
- D-fit 首次人工标注分布为 Present 27、Absent 25、Uncertain 2；52 个二元标签进入拟合，未知状态不作为负例。
- 旧 profile JSON/NPZ SHA 分别为 `4f9181ec…55e3`、`ecafa2e4…13b1a`，本轮逐字节身份一致。profile 目标值和梯度重算差均为 0，梯度无穷范数 `9.0234e-09`。原 profile 状态仍为 `BLOCKED / OVER_BUDGET`，没有被改写为 PASS。
- 七个冻结的 E2-B 科学代码/测试文件与起始 HEAD blob 哈希一致。R1 收据报告输入哈希及保护资产树哈希未改变；R0 结果及旧 profile 均保留。

## 执行状态与失败轨迹

| 项目 | 结果 |
|---|---:|
| Learner 全部完成 | 27/28 |
| B0 控制完成 | 28/28 |
| 持久化 checkpoint 指标行 | 725 |
| 预期总 checkpoint 行 | 728 |
| D-fit 初始化 | 14 条，其中 13 次新求解、1 次按批准复用 profile |
| learner 在线拟合尝试 | 320 次；未知反馈不拟合 |
| solver 日志 | 334 条（14 次初始化 + 320 次在线拟合） |

唯一失败为 DINOv2 / G-CLS / X learner 的第 10 个反馈事件。最后有效状态是 checkpoint 9；checkpoint 10–12 未产生。L-BFGS-B 返回 `success=True`，但仅以相对目标函数变化停止；目标有限，梯度无穷范数为 `1.1482185e-06`，超过冻结接受门槛 `1e-6`。执行器因此拒绝候选并把轨迹记为 `SOLVER_FAILED / FTOL_ONLY_NOT_CONVERGED`，没有替换优化器、重试或补造后续状态。其他注册轨迹继续完成。

全任务用时为 **611.41 秒墙钟、882.02 秒进程 CPU**，峰值 RSS **1,379,172,352 字节（约 1.28 GiB）**；未触及 7200 秒 CPU/墙钟上限或 6 GiB 软门槛。

## 审计收据

- Run identity SHA-256：`ea1608c59c7070ace67d9db58d24ca19e77fc306569aa57db607ae9ac40bb10a`
- Run result SHA-256：`676266d083d85f8906a6e12db0089dc77f980620789dc3368db64d5d1f283327`
- Aggregate manifest SHA-256：`72e60d1fc22edf2e5928204737faf90aa2d113e70348a39533caf983a5bdb378`
- R1 code fingerprint：`1d7b7cc14e910559201a51f0284f511e667375a6d3bcd758b04585e1b6a6abd6`
- 聚合文件哈希均与 aggregate manifest 一致；28 组 learner/B0 初态匹配与冻结控制检查均通过。

逐图像预测、图像 ID、人工标签、模型参数与状态只保存在被忽略的输出中。本报告仅披露聚合结果。
