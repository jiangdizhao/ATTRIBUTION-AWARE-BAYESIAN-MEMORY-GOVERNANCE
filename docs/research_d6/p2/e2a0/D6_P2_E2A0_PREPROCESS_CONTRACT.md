# LP-D6-P2-E2-A0 — 统一预处理与空间特征接口

**执行日期：** 2026-11-15  
**协议版本：** `d6.e2a0.canonical_input.v1` / `d6.e2.shared_spatial_tokens.v1`  
**状态：** 工程 smoke 已核验；不包含研究图像或属性性能结果。

## 1. 统一输入张量

本轮只使用程序生成的 320×240 RGB 合成图，未读取任何研究图像。唯一预处理进程生成共享文件，两个模型进程随后读取同一份数组：

| 项目 | 冻结实现与核验值 |
|---|---|
| 源图 | 320×240、RGB、固定背景与线条；`synthetic_only_no_research_images` |
| 缩放 | Pillow 9.3.0 双线性插值，保持比例，最长边 448；结果 448×336 |
| 补边 | 居中至 448×448；上下各 56 px，不裁剪；有效矩形 `[0,56,448,392)` |
| 归一化 | RGB FP32 `[0,1]`，按 ImageNet mean/std `(0.485,0.456,0.406)/(0.229,0.224,0.225)` 归一化 |
| padding | 归一化后填 0；padding token 仍参与 backbone attention，mask 只控制后续空间聚合 |
| 张量 | `(1,3,448,448)`、FP32、NCHW、RGB |
| canonical 数组 SHA-256 | `275ceca051759eec7cf8b0cebdda8faa7a242732f0cec7196c4c6d91cf798337` |
| NPY 文件 SHA-256 | `492541ce3f2e86d39451cf40cd80c81f46556a53462f24a6cf0b08619bbe990d` |
| 预处理脚本 SHA-256 | `8841d39d8d8218a94e8cc3eda3b8a6426773a858d8f5746d42d396a0361d0496` |

三个 Conda 环境没有各自重做 resize/normalize。DINOv3 官方 processor 的 mean/std 和 channels-first 布局与本契约一致；processor 默认尺寸为 224×224。本轮按注册要求绕过 processor 的 resize/crop，直接输入同一 448 张量，DINOv3 官方动态 2D RoPE 前向成功处理 28×28 patch 网格。此 224 默认尺寸差异已记录，不将其隐藏为完全相同的 processor 配置。

## 2. Patch 有效区域

按 patch 中心是否落入半开有效图像矩形决定有效性，行优先展平：

| Backbone | 网格 | Patch token 数 | 有效 mask 数 | Mask 文件 SHA-256 |
|---|---:|---:|---:|---|
| DINOv2 ViT-L/14 reg | 32×32 | 1,024 | 768 | `6f0ba6d1175ed4493bd213239232adaa8ff1fbf1d6701b9f836e34558f334b1d` |
| DINOv3 ViT-L/16 | 28×28 | 784 | 588 | `c2cb860a99db0015fac57b03e40ec749b2747956a222db64812507d9aec8077b` |

Mask 只用于后续 mean/MIL 空间聚合，不表示 padding token 未影响有效 token 的 attention。

## 3. Backbone 原生输出与共享特征接口

两者均保留各自官方最终层输出，然后适配到共同 batch-first schema。最终层归一化输出的具体实现仍按各 backbone 区分；共同接口不表示原始向量尺度或内部定义相同。

| Backbone | 原生输出来源 | CLS | Registers | Spatial patches | 原生特征语义 |
|---|---|---:|---:|---:|---|
| DINOv2 | 官方 `forward_features`: `x_norm_clstoken`, `x_norm_regtokens`, `x_norm_patchtokens` | `(B,1024)` | `(B,4,1024)` | `(B,1024,1024)` | 官方最终 LayerNorm 输出；不是逐 token 单位范数 |
| DINOv3 | HF `last_hidden_state`，顺序 CLS、4 registers、行优先 patches | `(B,1024)` | `(B,4,1024)` | `(B,784,1024)` | 模型最终 LayerNorm 输出；不是逐 token 单位范数 |

`feature_interface.py` 提供适配器，将两种原生输出统一成 `d6.e2.shared_spatial_tokens.v1`：

- `cls_unit`: `(B,D)`；
- `register_unit`: `(B,4,D)`；
- `patch_unit`: `(B,N,D)`；
- `valid_patch_mask`: `(B,N)`；
- 对每个 CLS、register、patch token 独立做 FP32 L2 归一化。

掩码不删除输出 token，仅供后续聚合。按 G0 定义，G-Mean 使用有效单位 patch 的算术平均后再次单位化；register tokens 不作为空间 patch 使用。数值转换已在保存的两组实际合成 smoke 输出上复核：单位范数最大误差不超过 `1.2e-7`；CPU/GPU 共享表示 max-abs 均低于 `4.5e-6`，最小 token cosine 均不低于 `0.9999997`。另有 6 个无模型单元测试。

此接口不读取图像、不加载模型、不产生人工属性标签。正式特征文件应继续遵照 G0 的 opaque ID、图像 SHA、输入/模型/接口版本 manifest 与独立 SHA 保存规则；本轮未建立正式研究特征缓存。
