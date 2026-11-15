# D6-P2-E1-R2 限定数值裁决

**结果状态：`NUMERICALLY_ROBUST_BUT_PROTOCOL_PARTIAL`**  
**审查范围：** 已归档 E1-R1 数值差异的影响裁决；不是新模型实验。  
**执行分支 / HEAD：** `stage3-eg-famar-execution` / `88868d4b0b18ffd6bb03e3d94b49e67904df044a`

## 1. 范围与完整性

本轮未重跑六条轨迹、未搜索超参数、未运行 E2。只使用 E1-R1 已登记的 102 条特征缓存、W/X tape、模型状态、未来 logits 和核验记录；未解码图像像素、未读取 mask、未加载 DINO/Torch、未访问 D-conf 或 sealed/HOLD 数据。监督边界保持为 D-fit 与已揭示的合法反馈；未来标签在三次新增求解结束后才由评价步骤载入。

R1 的四份报告和实现均已检查。R1 仍为 `PARTIAL`。R1 实际执行代码与归档哈希一致：`run_e1.py` `e0d198f4239e07b0a0e6729b0e48cb3a186204d8e07de53fd0ebd230a5fe7a9d`，`proximal_logistic.py` `6fc1a5765de46dfb3a51a4b2be8ee3205bbee480fd84b494352be877b971b2df`。R1 provenance 哈希为 `9d75358a3b9306d4ee1780051ec2be4afce57c597251ff7d98c7abc0f5ba9834`。

| 身份项 | 核验结果 |
|---|---|
| P1-A 输入快照 | 与 R1 一致：`580940b5bc9b204d6bded6a548e1cc76fdba65e8b05be1f8bfbbb8fc35d8b0c7`；执行前后未变 |
| 输入 manifest / feature manifest | `820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63` / `ee0232b29f64f9fd4f10198ba3ef20da89fa85b2083a389b774f6a9723bbb99d` |
| W/X tape | `13561809074aad72e2f13289723ccba8addc4768395480c50d3173abf66e8cc1` |
| 注册初态 | `86e8e8868e7d869ccc2430555db314014daf0f7befd47c0a529e12fa3057c6f2` |
| 原 P1-B 输出 | 15/15 文件哈希复核一致 |
| R1 输出 | 11 个文件的大小和 SHA-256 执行前后完全相同；清单见本地 ignored 输出中的 `r1_output_sha256_manifest.json` |
| 运行环境 | R1 原 Conda 解释器 `D:\anaconda3\envs\torch\python.exe`；Python 3.11.14、NumPy 1.24.1、SciPy 1.11.2；CPU-only、线程上限 2、Torch 未导入 |

新诊断代码位于 `scripts/research_d6/p2/e1_r2/numerical_adjudication.py`。数值日志位于 Git-ignored `outputs/research_d6/p2/e1_r2/`。该目录有三次求解的逐次调用账本；运行器拒绝覆盖已有目录。R1 的所有验收状态和结果文件保持原样。

## 2. 15 条既有 Newton-CG 差异的界限分析

对每条既有核验记录，记 R1 的 L-BFGS-B logit 为 `s_i`，已记录的未来最大绝对 score 差为 `ε = max_i |s_i - s'_i|`。零阈值预测只有在 `min_i |s_i| > ε` 时才可由该界限严格保护。对同一产品的正负样本分数差，扰动至多为 `2ε`；只有最小正负间隔 `> 2ε` 时，才可排除该产品的正负排序变化，因此也保护 paired AP 与产品 AUC 的排序输入。表中 margin 均严格超过相应界限。

| 协议 | γ | checkpoint | ε（R1 最大分差） | 最小零阈值距离 | 最小产品内正负间隔 | 预测改变 | 排序改变 |
|---|---:|---:|---:|---:|---:|---|---|
| X | 1e-4 | 0 | 1.1500e-4 | 0.236127 | 0.051604 | 严格排除 | 严格排除 |
| X | 1e-4 | 3 | 2.7201e-5 | 0.097792 | 0.116293 | 严格排除 | 严格排除 |
| X | 1e-4 | 6 | 4.2579e-5 | 0.049824 | 0.019883 | 严格排除 | 严格排除 |
| X | 1e-4 | 12 | 7.0090e-5 | 0.087230 | 0.014109 | 严格排除 | 严格排除 |
| X | 1e-3 | 0 | 2.1803e-6 | 0.236127 | 0.051604 | 严格排除 | 严格排除 |
| X | 1e-3 | 3 | 6.3274e-6 | 0.053169 | 0.017177 | 严格排除 | 严格排除 |
| X | 1e-3 | 6 | 1.0930e-5 | 0.087823 | 0.006971 | 严格排除 | 严格排除 |
| W | 1e-4 | 0 | 2.9196e-4 | 0.068325 | 0.044854 | 严格排除 | 严格排除 |
| W | 1e-4 | 3 | 1.8156e-4 | 0.104028 | 0.035287 | 严格排除 | 严格排除 |
| W | 1e-4 | 6 | 3.9728e-5 | 0.138476 | 0.465419 | 严格排除 | 严格排除 |
| W | 1e-4 | 12 | 5.5141e-5 | 0.049767 | 0.183370 | 严格排除 | 严格排除 |
| W | 1e-3 | 0 | 1.8891e-6 | 0.068325 | 0.044854 | 严格排除 | 严格排除 |
| W | 1e-3 | 3 | 1.4726e-5 | 0.171329 | 0.165777 | 严格排除 | 严格排除 |
| W | 1e-3 | 6 | 8.4469e-6 | 0.190785 | 0.234589 | 严格排除 | 严格排除 |
| W | 1e-3 | 12 | 1.6311e-5 | 0.024243 | 0.307011 | 严格排除 | 严格排除 |

**裁决：** R1 的 11/15 超过 `1e-5` score 容差仍是正式核验不通过，不能改标 PASS；但在这 15 个已核验 checkpoint 上，已记录的最大 score 差不足以改变任一二元未来视角的零阈值分类，也不足以改变任一产品的正负排序。由此可严格排除这些数值差异改变 BA、paired AP 或 ROC-AUC 的可能性（以 R1 已保存的 logits、标签与记录的 ε 为界）。本结论不覆盖缺失的 X/γ=1e-3 checkpoint 12，也不把 score 容差放宽。

## 3. 预选最大差异 checkpoint 的独立求解

事前机械规则为：按 15 条记录的 `max_future_score_abs_difference` 降序取前两名；同值时按 protocol、数值 γ、checkpoint 排序。所选为 W/γ=1e-4/checkpoint 0 和 checkpoint 3。选择只看已有数值差，不看未来标签、指标或新模型表现。

| checkpoint | Newton 目标差 | Newton 梯度无穷范数 | 最大未来分差 | 21 项预测一致 | BA（LBFGS / Newton） | paired AP（LBFGS / Newton） | ROC-AUC（LBFGS / Newton） | 权重 L2 差 | 截距差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W / 1e-4 / 0 | 1.4907e-9 | 2.6127e-7 | 2.9196e-4 | 21/21 | 0.6875 / 0.6875 | 0.7500 / 0.7500 | 0.6250 / 0.6250 | 0.0033325 | 0.0022156 |
| W / 1e-4 / 3 | 3.0988e-10 | 5.3483e-8 | 1.8156e-4 | 21/21 | 0.4375 / 0.4375 | 0.7083 / 0.7083 | 0.5625 / 0.5625 | 0.0014088 | 0.0003607 |

两次 Newton-CG 均 success，且梯度和目标差均满足原门槛；最大 score 差仍未满足 `1e-5`。两 checkpoint 的产品级 BA、AP 与 AUC 逐项一致。R1 保存参数重算出的 logits 与 R1 预测文件完全一致（最大差为 0）。这验证 score 容差失败反映的是数值解的参数/分数小差异，而非本次所检 checkpoint 的分类或排序结论改变。

## 4. X / γ=1e-3 / 事件 8 单点诊断

按注册目标使用相同 D-fit、X 事件 1–8 合法二元反馈、`λ=1e-4`、`γ=1e-3` 和冻结 `w0_new`；以已验收事件 7 参数为 Newton-CG 初值。求解器成功，6 次迭代、6 次函数评价；目标值 `0.3904840981592111`，梯度无穷范数 `1.0716192932e-7`，通过原 `1e-6` 梯度门槛。

被拒绝的 L-BFGS-B 候选记录目标为 `0.39048409814383367`、梯度无穷范数为 `1.2826408917e-6`。两目标仅差 `1.5377e-11`，而梯度门槛只有 L-BFGS-B 候选未满足。被拒候选系数没有保存在 R1 输出中，因此其参数距离为 `UNKNOWN_NOT_ARCHIVED`；本任务不重构该候选参数。

此结果更符合 L-BFGS-B 因相对目标下降停止而未达到注册梯度门槛的单点数值停止条件问题。Newton-CG 单点成功不追认原更新，不把事件 8 写回 R1，也不继续事件 9–12；原 X/γ=1e-3 轨迹继续保持 `FTOL_ONLY_NOT_CONVERGED` 和 `PARTIAL`。

## 5. 实现与监督边界检查

- R1 已有 23/23 checkpoint objective/解析梯度检查通过；初始化的 12 个方向有限差分最大误差为 `1.2168e-10`。R2 没有发现解析梯度公式错误的新证据。
- 两个重算 checkpoint 的持久反馈特征/标签与其合法 tape 前缀及已保存状态一致；事件 8 的 checkpoint-7 状态也与事件 1–7 一致。初始 `w0_new` 哈希与冻结记录一致。
- D-fit 与反馈数据只用于目标函数；未来属性标签在三次 Newton-CG 求解结束后才读取，用于边界、排序与性能诊断。没有使用 defect type、未来标签或图像像素参与更新。
- 没有新证据指向模型状态损坏、反馈顺序错误或监督越界。

## 6. E1 结果可支持的结论边界

现有 E1 可支持：在当前注册特征、稀疏反馈 tape 和小型已曝光 D-dev 上，5 条轨迹按各自门槛完整运行，且已记录 Newton/L-BFGS-B score 不一致的 15 个可用 checkpoint 均有足够分类与排序余量，评价指标不会被该差异改变；失败事件 8 的独立 Newton-CG 解满足原梯度门槛，提示单点 L-BFGS-B 停止条件导致数值拒绝。

现有 E1 **不能**支持：六条轨迹全部完成；X/γ=1e-3 的 12 事件结果；proximal 正则优于/劣于 γ=0；对真实产品总体的泛化、持久迁移或独立确认。当前结果仍只覆盖已曝光、操作性分组的 24 个 sample-key 和 21 个二元未来视角，并非独立物理对象或 D-conf 证据。

## 7. 可复核产物

Git-ignored 数值日志：`outputs/research_d6/p2/e1_r2/`

- `checkpoint_bound_analysis.csv`：15 个 checkpoint 的边界与排序上界。
- `selected_checkpoint_newton_rechecks.csv`、`selected_checkpoint_product_metrics.csv`：两次重算的整体及产品级指标。
- `event8_newton_diagnostic.json`：事件 8 数值诊断。
- `solver_attempt_ledger.json`：恰好三次新增 Newton-CG 调用的调用账本。
- `r1_output_sha256_manifest.json`、`run_provenance.json`：输入、运行与 R1 输出身份摘要。

在授权 R1 环境中运行：

```powershell
& 'D:\anaconda3\envs\torch\python.exe' scripts/research_d6/p2/e1_r2/numerical_adjudication.py
```

R2 诊断已执行完毕，未提交 Git。原 R1 结论保持 `PARTIAL`。
