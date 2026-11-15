# LP-D6-P2-E2-B-R1 资源修订与执行协议

**日期：** 2026-11-16  
**授权：** `E2B_R1_RESOURCE_OVERRIDE_APPROVED`  
**起始身份：** `stage3-eg-famar-execution` / `072e11adaffd85cf64b85b4708befe45cef92018`

## 资源决定

R0 的 `E2B_BLOCKED / OVER_BUDGET` 状态保留。原 profile 的 JSON 与 NPZ 不修改。R1 的新预算为 **7200 秒 wall time** 与 **7200 秒 process CPU time**，最多 2 个计算线程、CPU-only；峰值 RSS 6 GiB 为软停止线。原预测 `2 × 5.9219 × 336 = 3979.5` process-CPU 秒（约 66.3 分钟）仅是估算，不代表完整矩阵实测耗时。

如果 R1 预算耗尽或 RSS 越线，保留已提交 checkpoint 与最后有效状态；不删配置、不减 seed、不变更 optimizer/精度/门槛。每个监督进程会记录 wall、process CPU、峰值 RSS 和停止原因；多次恢复的资源消耗累计到同一授权预算。

## R0 profile 复用门槛

执行器读取 `outputs/research_d6/p2/e2b/bounded_profile.json` 和 `profile_initialization.npz`，记录两者 SHA，并核验旧状态、旧 code hashes、数据与缓存身份、seed 0 初态、solver 成功与梯度门槛。使用 D-fit 的 52 个二元 bag 重新计算已保存参数的 objective/gradient；不重新拟合 profile，不读取 D-dev future truth。旧 profile 继续保持 `BLOCKED / OVER_BUDGET`。

## 固定实验矩阵

- Global：2 backbone × 3 readout（G-CLS、G-Mean、G-Concat）× X/W = 12 条 learner trajectories。
- MIL：2 backbone × 2 loss（M-BCE、M-Rank-Softplus）× X/W × seeds 0/1 = 16 条 learner trajectories。
- 每条 learner trajectory 配对一个独立可核验的 frozen B0 control，共 28 controls。
- 共 336 次注册的完整历史拟合（含可复用 profile 初始化）；R1 可复用 DINOv2/M-BCE/seed-0 D-fit 参数。其余初始化按原方法、固定顺序执行。
- 维持 R0 注册的特征、标签、损失、初始化、L-BFGS-B 配置、`lambda=1e-4`、`tau=1`、`margin=1`、`kappa=0.1`、`rho=0.1`、阈值 0、X/W tape、指标定义及失败门槛。R1 不新增方法或超参数。
- 每个事件先记录查询分数和参数 SHA，再持久化反馈揭示，之后才允许更新。Uncertain/N-A 不进二元损失；未来真值只交给 evaluator。

## 执行完整性

七个 R0 科学文件按起始 HEAD 逐字节核验并保持只读。R1 执行器在新进程中固定使用 `D:\anaconda3\envs\torch\python.exe`，禁止 CUDA。训练前运行原 15 项测试及 R1 合成测试；恢复仅接受 HEAD、R1 代码指纹、缓存/tape/标注输入哈希、checkpoint 顺序、状态参数 SHA 与文件 SHA 全部匹配的完成记录。

每个 checkpoint 连同 learner/B0 状态通过临时目录原子发布，包含训练历史、预测、evaluator 指标、有效参数、参数 SHA、solver 结果和输入/代码身份。已提交记录只读；异常或失败记录不会被覆盖。D-dev 已曝光，样本规模小，来源 defect type 可能与人工擦伤标签混杂，且没有属性级定位真值。结果只作为探索性开发证据。
