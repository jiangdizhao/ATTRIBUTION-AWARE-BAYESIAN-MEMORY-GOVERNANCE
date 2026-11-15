# LP-D6-P2-E2-B 实现与数值审计

**日期：** 2026-11-16  
**状态：** profile 完成；主矩阵因预先登记的资源门槛停止。  
**运行目录：** `outputs/research_d6/p2/e2b/`（Git-ignored）

## 输入与权限核验

分支及 HEAD 与任务要求一致：`stage3-eg-famar-execution` / `1e6a602cb8b788bca03a61773ef442a55f3fa7e2`。

Profile 首先运行了 E2-B 合成测试，然后执行 E2-A1 缓存核验与 W/X tape 角色核验。缓存身份为：

| 清单 | SHA-256 | 结果 |
|---|---|---|
| Canonical | `0fca669b4f69ed2071b4f4c0ce8f60728c6bd0e6efe126be580f23037b748133` | 通过 |
| DINOv2 feature manifest | `3588bab021a9cd8043d663c1d6ce46f366f89f89802fb1ac928dfae3360efddf` | 通过 |
| DINOv3 feature manifest | `20e100671344422779d848f7664cdb6290d7fe4cd0e1ed1ea09f108fe0a01887` | 通过 |
| P1-A input manifest | `820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63` | 通过 |
| P1-A feature manifest | `ee0232b29f64f9fd4f10198ba3ef20da89fa85b2083a389b774f6a9723bbb99d` | 通过 |
| W/X tape | `13561809074aad72e2f13289723ccba8addc4768395480c50d3173abf66e8cc1` | 通过 |

核验覆盖 102 张注册图像（54 D-fit、48 D-dev）、两个 backbone 的顺序与来源 SHA 对应关系，以及 E2-A1 对缓存数组 SHA、shape、dtype、有限值、单位范数、mask 几何和 source join 的独立检查。tape 共有 240 条角色记录；每协议 12 feedback、24 future、84 excluded。排除视角不要求缓存连接，也未用于 profile。

Profile 只读取 D4-Lite/D4B 的 D-fit 首次人工标签：54 条中 27 present、25 absent、2 uncertain；训练使用 52 条二元标签。代码路径没有读取 D-dev annotation values、future truth、原始图像、mask 或 DINO 权重。CUDA 被屏蔽；没有特征提取、GPU 推理、软件安装或 protected 输出写入。

## 合成数值验证

15/15 项通过（执行环境：`D:\anaconda3\envs\torch\python.exe`，Python 3.11.14、PyTorch 2.0.1+cu117、NumPy 1.24.1、SciPy 1.11.2、scikit-learn 1.3.0；本轮 CUDA 不可见、线程上限 2）。验证覆盖：全局与 MIL 解析梯度有限差分、Softplus 配对梯度、`rho=0` 等价性、无效 patch 排除、非二元/单类拒绝、配对数及权重、bias 不变性、极端数值、seed 配对、收敛门槛、未来标签反事实、先预测后反馈、uncertain 不更新、B0 不变和重复反馈拒绝。

Profile 前额外修正了 solver 日志 `checkpoint` 字段：现由在线 runner 传入真实 `stream_order`，不再错误记录全局求解行序号。该修改发生在第一次实验性标签读取前，并包含在下方代码指纹内；不改变优化目标或更新语义。

## 限量 D-fit profile

唯一执行配置：DINOv2 / M-BCE / seed 0 / 52 个二元 D-fit bags / 固定 L-BFGS-B 设置。

| 项目 | 实测 |
|---|---:|
| 求解器 | PASS；SciPy success，有限参数/目标/梯度 |
| Objective | 0.09816280474733305 |
| 梯度无穷范数 | `9.023425575531777e-09` |
| 迭代 / 函数评估 | 101 / 117 |
| solver elapsed wall | 3.5131 s |
| profile process CPU | 5.9219 s |
| 采样峰值 RSS | 751,214,592 bytes（约 716 MiB） |
| 全矩阵唯一拟合数（含可复用 profile） | 336（global 144、MIL 192） |
| 预登记保守投影 | 3,979.5 s（66.3 min） |
| 计划窗口 | 1,800 s（30 min） |

`bounded_profile.json` 标记 `status=BLOCKED`、`resource_gate=OVER_BUDGET`；profile 的局部求解本身通过。代码按预登记公式 `process_cpu × 336 × 2` 计算投影。未启动主矩阵；`run_20261116_01/` 不存在。

Profile receipt 指纹：`2a665ce974a942107acddc9c62c30ebdbe50cf879c8d815a65d70d6bec81a304`。关键代码 SHA：

- `config.py`: `02a482bd9b325e6efb58870485d546f57f98321ffe8289f24e953e64fba9fe14`
- `data.py`: `ea1f6c13cabd944ca25d13ad2603b23711118c2b13c9db15f18784809b769973`
- `evaluator.py`: `db2625d95e4e4b71cecbf94a805e2607261c05dc92612dc973e0e5378af72ab9`
- `model.py`: `26d2b24f5d94181a2a6aad431bd5703935fb351f541fa95116c93e2a159304f2`
- `pipeline.py`: `372c0f2fc43ce5b9b3f8b0b31ccc4caebbc07a0299cbc1f6ea272db14dc1351a`
- `run_e2b.py`: `a40e173258254f4ac85bdcf0cb6cd2d45b9e826a8bca5bc386adb6fbb7cacb7b`
- `test_e2b_numerics.py`: `a00ca902cbff97d865ac7c3a76220849400310c913da53b9373cc2cbaa8bd35d`

审计说明：协议 Markdown 在 profile 后做了 UTF-8 可读性整理；损失值、优化设置、30 分钟窗口未改。投影公式 `2 × process CPU × 336` 已存在于 profile 前的 `run_e2b.py`，其 SHA 被本次 profile receipt 记录；不是看到未来性能后追加的停止条件。

忽略目录中保留 `bounded_profile.json` 与 `profile_initialization.npz`；后者 SHA-256 为 `ecafa2e417aa9532ce9b655c50dcb841df7a75bb1791d8528f5d0e3bbb613b1a`，只包含 D-fit 的初始化和拟合参数。所有原始模型结果及缓存保持只读。
