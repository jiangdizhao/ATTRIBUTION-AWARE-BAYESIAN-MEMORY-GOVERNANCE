# LP-D6-P2-E2-A0 — 双 Backbone GPU Smoke 结果

**执行日期：** 2026-11-15  
**结论：** `E2A0_PASS`（仅工程 smoke）  
**Git 身份：** `stage3-eg-famar-execution` / `2d33f899c7c17157376b8be1bf9c311114ff2344`

## 1. 环境、模型和 GPU

| 项目 | DINOv2 | DINOv3 |
|---|---|---|
| 独立解释器 | `D:\anaconda3\envs\torch\python.exe` | `D:\anaconda3\envs\dinov3\python.exe` |
| Python | 3.11.14 | 3.11.18 |
| PyTorch | 2.0.1+cu117 | 2.14.1+cu126 |
| 相关依赖 | torchvision 0.15.2+cu117；NumPy 1.24.1 | Transformers 5.23.0；NumPy 2.4.6 |
| 模型 | `dinov2_vitl14_reg` | `facebook/dinov3-vitl16-pretrain-lvd1689m` |
| 身份 | checkpoint SHA `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`；源码 tree SHA `7dd1641727d0b7b0c58f803906d03da05b8660ba8b72da5b3c5705cb9a1b4b8c`（174 files） | 本地 revision `ea8dc2863c51be0a264bab82070e3e8836b02d51`；`model.safetensors` SHA `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`；config SHA `135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5` |
| GPU | RTX 3090 Ti，24,564 MiB；启动时空闲约 21.5 GiB | 同卡，独立顺序运行；启动时空闲 21,516 MiB |
| 精度 | FP32；autocast/TF32 关闭 | FP32；autocast/TF32 关闭 |

启动时 `nvidia-smi` 可见 27 个 compute process 条目，但逐进程显存为 N/A；没有终止任何进程。每个 smoke 的 PyTorch allocator cap 为 12 GiB，均未触及。两个环境和 CUDA/PyTorch 栈不同，下面延迟只作各自资源参考，不能据此比较 backbone 效率。

## 2. 前向形状、数值与资源

两个 backbone 都在同一 canonical NPY 张量上各执行 CPU 前向、2 次 GPU warm-up 和 5 次同步计时前向。输出均有限，形状、patch 顺序和 mask 数均符合注册契约。

| 指标 | DINOv2 | DINOv3 |
|---|---:|---:|
| CLS / registers / patches | `(1,1024)` / `(1,4,1024)` / `(1,1024,1024)` | `(1,1024)` / `(1,4,1024)` / `(1,784,1024)` |
| 有效 patches | 768 / 1,024 | 588 / 784 |
| CPU forward | 6.265 s | 3.807 s |
| GPU forward 中位数（5 次） | 0.04846 s | 0.03710 s |
| GPU forward min–max | 0.04835–0.05131 s | 0.03682–0.03862 s |
| 峰值已分配显存 | 1,338,310,656 bytes（约 1.25 GiB） | 1,264,184,320 bytes（约 1.18 GiB） |
| CPU/GPU 原始 max-abs：CLS / reg / patch | `1.64e-5 / 2.19e-5 / 1.03e-4` | `6.69e-6 / 1.06e-5 / 8.90e-6` |
| 单位 token 后 max-abs（全组最大） | `4.43e-6` | `1.11e-6` |
| CLS cosine / 有效 patch 最小 cosine | `1.000000 / 0.99999982` | `1.000000 / 0.99999976` |
| GPU 重复运行 | 4 次对比；归一化 max-abs `1.74e-6`，有效 patch 最小 cosine `0.99999982` | 4 次对比；保存的 recovery 中原始 token 完全相同，归一化最小 cosine `0.99999976` |

上述 CPU/GPU 差异均满足预注册 smoke 标准：单位 token max-abs `≤1e-3`、CLS cosine `≥0.9999`、有效 patch 最小 cosine `≥0.999`。它们只验证数值执行路径，不证明两 backbone 的任务性能等价。

## 3. 失败记录、警告与审计

- DINOv2 输出 `xFormers is not available` 警告；官方实现使用可用 fallback，前向和全部数值检查通过，未安装依赖。
- DINOv3 第一次真实前向已完成并保存 tokens，但结果 JSON 序列化遇到 Transformers `SizeDict` 类型错误。该失败记录保留在 `dinov3_smoke_results_final.json`；随后只将 processor 尺寸转换为字符串并用新的文件名做一次同参数 recovery。恢复结果为 `E2A0_PASS`，未更改输入、权重、精度、阈值、warm-up 或重复次数。旧记录未覆盖。
- recovery 原始 JSON 的 `sequence_length` 汇总误将 CLS 特征维度当作 token 数写为 1812。独立核验后，正确值为 `1+4+784=789`；代码已修正供未来运行，原始 recovery JSON 保留，元数据校正版另存为 `dinov3_smoke_results_adjudicated.json`。这不影响前向张量或任一通过判据，也未为修正而重跑模型。
- DINOv3 processor 默认尺寸为 224×224；mean/std 和 channels-first 与 G0 相符。本轮按统一预处理契约绕过 processor 尺寸变换，直接向官方动态 RoPE 路径传入 448×448；已通过前向。这是明确记录的适配选择。
- 模型只从本地缓存加载，HF 离线模式启用，未下载权重或安装/升级依赖。
- D4/D6 图像、mask、属性标签和受保护集合未访问；没有创建 102 张正式图像特征，没有训练模型，也没有运行 E2-B。
- `protected_assets_before.json` 与 recovery 后重算清单完全一致；两套 backbone 权重及源码指纹也与开始前一致。结果写入 Git 忽略的 `outputs/research_d6/p2/e2a0/`。

最终补充裁决见 `outputs/research_d6/p2/e2a0/e2a0_final_adjudication.json`。原始 orchestrator 的 `run_gpu_smoke_summary.json` 保持 `E2A0_PARTIAL`，反映首次 DINOv3 序列化失败；最终裁决采用独立新增结果，不覆写该历史记录。
