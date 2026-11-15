# D6-P2-E1 协议与实现记录

**日期：** 2026-11-15  
**执行状态：** `BLOCKED_BEFORE_FITTING`  
**授权范围：** LP-D6-P2-E1；不含 E2、D-conf、新特征或新数据访问。

## 结论摘要

P1-A 输入完整性门控通过，W/X tape、首标人工状态、反馈/未来分组及 P1-B 保存的 Logistic 状态均可追溯。但保存的 P1-B `alpha=1e-4` 初始参数在本任务注册的平均 log-loss 目标上，梯度无穷范数为 `1.7226304473e-5`，高于 `1e-6` 门槛。按授权要求，在任何 Proximal 或 Newton-CG 拟合前停止。

未替换初始状态、未调整容差、未运行 gamma=0 新拟合；没有新增 E1 学习结果。唯一运行期机读诊断文件为受控输出中的 `anchor_compatibility_gate.json`，其中没有图像 ID、逐条标签或权重数组。

## 冻结协议

- 特征：复用 P1-A 既有 102 条 FP32、1024 维、L2 归一化 CLS 缓存；不加载 DINO、不提取新特征。
- 初始信息：D-fit 54 条首标记录中，Present 27、Absent 25、Uncertain 2；仅 52 条二元记录进入损失。
- 目标：

  `mean(logaddexp(0, Xw+b) - y*(Xw+b)) + λ/2 ||w||² + γ/2 ||w-w0||²`

  `λ=1e-4`，`γ∈{0,1e-4,1e-3}`；截距不正则化也不锚定。`w0` 在整条轨迹中固定。
- 更新顺序：预测查询视角 → 揭示其已登记的合法首标 → 二元标签才加入完整历史并优化 → 评分固定 future 集。Uncertain/N-A 消耗事件成本单位，不优化、不改变状态。
- gamma=0：预定复用 P1-B alpha=1e-4 的已保存 Refit 轨迹；不重跑原实验。
- gamma>0：CPU FP64、SciPy L-BFGS-B，固定 `ftol=1e-12`、`gtol=1e-8`、`maxiter=10000`、`maxls=50`、`maxcor=10`；成功还要求梯度无穷范数不超过 `1e-6`。
- 数值交叉核验：W/X 各在 `{0,3,6,12}`、两个正 gamma 上做 Newton-CG，最多 16 次；它们不得用于选择参数。

## 输入完整性与分组核验

P1-B 的 ancestry-aware preflight 在当前执行 HEAD 上通过。缓存为 102 条：54 D-fit、48 D-dev。注册输入指纹如下：

| 资产 | SHA-256 |
|---|---|
| P1-A input manifest | `820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63` |
| P1-A feature index | `ee0232b29f64f9fd4f10198ba3ef20da89fa85b2083a389b774f6a9723bbb99d` |
| W/X feedback tape | `13561809074aad72e2f13289723ccba8addc4768395480c50d3173abf66e8cc1` |

预检按既有白名单读取授权源图像字节，仅用于 SHA-256 身份校验；没有解码像素。固定 W/X tape 各有 12 个反馈事件、24 个 future 视角、21 个二元可评价 future 视角。X 反馈为 7 Present / 5 Absent；W 为 5 Present / 6 Absent / 1 Uncertain。反馈与 future 的图像及 `(product, sample-key)` 组均不交叉。Sample-key 仍只是操作性分组，不证明物理物件独立。future 标签只用于评价支持是否可定义的核验，没有进入拟合，也不是本次停止原因。

## 初始化核验及停止点

历史 Logistic 的 `C=1/(alpha*n)=192.3076923` 与注册的 `λ=alpha=1e-4` 目标映射一致。缓存中的 52 条 D-fit 特征和标签逐数组匹配 P1-B 保存的初始拟合历史。归档 B0 系数、截距与该初始 Refit 状态逐值一致；其 W/X checkpoint-0 future 分数与保存的 P1-B B0/ReFit 分数最大绝对差均为 `3.55e-15`。P1-B 初始和最终状态摘要也与索引及 checkpoint 日志匹配；原始 P1-B 15 个输出文件的 SHA-256 全部匹配既有清单。

但同一保存初始参数在注册目标处的梯度门槛失败。因此脚本在状态复制、分数一致性和完整性核验后停止，未执行任何模型拟合或独立求解器拟合。没有读取 future 真值用于优化或停止决策。

## 实现与运行

- `scripts/research_d6/p2/e1/proximal_logistic.py`：FP64 目标、解析梯度、Hessian-vector product、固定 L-BFGS-B 契约、失败拒绝逻辑、不可变锚点及状态摘要。
- `scripts/research_d6/p2/e1/run_e1.py`：P1 输入门控、P1-B 初始化复用、W/X 角色检查、分数核验、fail-closed 闸门及后续轨迹实现。由于闸门失败，后续轨迹代码未在真实研究输入上运行。
- `tests/research_d6/test_p2_e1.py`：10 项合成与只读完整性测试，全部通过。
- 运行解释器：`D:\anaconda3\envs\torch\python.exe`；CUDA 已屏蔽，线程上限为 2，未导入 Torch。
- 原始 P1-A/P1-B/G0 目录和文件保持不变；没有提交或推送 Git。

源文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `scripts/research_d6/p2/e1/proximal_logistic.py` | `415e6f83d4b06bd1bd11a3a531f6a65b1f9a9a4fa4339f30d673f860be2dbcb1` |
| `scripts/research_d6/p2/e1/run_e1.py` | `6360bd737003c1f8ce373cbc3ac0a11e0e6cbd9eedc291d584449855222ec209` |
| `tests/research_d6/test_p2_e1.py` | `53a62ea86dd985afbb88c89cbc301864ade31946b1546511999bfc02b2949e47` |

## 解释边界

本状态只表示历史初始化没有满足这次预注册的梯度门槛。它不代表 Proximal 优于或劣于普通 Refit，也不代表 D6-P2 headroom 已测量，更不构成方向 A 的失败结论。
