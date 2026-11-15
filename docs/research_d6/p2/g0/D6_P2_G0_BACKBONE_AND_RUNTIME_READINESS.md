# LP-D6-P2-G0 — Backbone 与运行环境准备度

**只读核对日期：** 2026-11-15
**执行状态：** 未加载模型、未读 checkpoint 张量、未跑推理或 GPU kernel

## 1. 当前运行环境

| 检查项 | 实测/元数据结果 | 含义 |
|---|---|---|
| 建议 Python | `D:\anaconda3\envs\torch\python.exe` | 直接使用绝对解释器，不依赖 PowerShell 激活状态。 |
| Python | 3.11.14 | 来自该解释器。 |
| Conda 环境变量 | `CONDA_DEFAULT_ENV=base` | 与解释器所在的 `envs\torch` 路径冲突，说明是父 shell 继承值；不能把它当作解释器身份。 |
| PyTorch / torchvision | 2.0.1+cu117 / 0.15.2+cu117 | 通过 package metadata 查询；没有 import torch。 |
| NumPy / pandas / Pillow / scikit-learn / SciPy | 1.24.1 / 2.1.0 / 9.3.0 / 1.3.0 / 1.11.2 | 已安装版本。 |
| Transformers / xFormers / dinov3 | 4.38.1 / 未安装 / 未安装 | Transformers 低于 DINOv3 官方 README 所述支持版本 4.56.0；无 DINOv3 包。 |
| GPU | NVIDIA GeForce RTX 3090 Ti；总显存 24,564 MiB；查询时空闲 21,355 MiB；利用率 3% | `nvidia-smi` 单次非侵入查询；空闲显存随时间变，不能据此固定 batch size 或承诺余量。 |
| RAM | 63.8 GB 总量、约 36.8 GB 可用 | 先前同一 G0 只读系统查询；不是任务运行峰值。 |

本轮只读取 Python 包元数据和 `nvidia-smi`；未导入 torch，因此 CUDA 可用性没有用 PyTorch API 复测。P1-A 已在同一 `torch` 环境用 CPU/FP32 完成 DINOv2 feature extraction。若以后获批 GPU 提取，必须在开始时重新测空闲显存；不可停止其他项目进程。

## 2. Backbone 本地就绪度

| 属性 | DINOv2 ViT-L/14 Register | DINOv3 ViT-L/16 web pretrained |
|---|---|---|
| 指定模型 | `dinov2_vitl14_reg` | `dinov3_vitl16`, weights `LVD1689M`；HF 名 `facebook/dinov3-vitl16-pretrain-lvd1689m` |
| 本地源码 | `C:\Users\rico\.cache\torch\hub\facebookresearch_dinov2_main` 存在；无 `.git` 元数据。P1-A 源树 174 文件、2,654,538 bytes，tree SHA `7dd1641727d0b7b0c58f803906d03da05b8660ba8b72da5b3c5705cb9a1b4b8c`。 | 未找到 `C:\Users\rico\.cache\torch\hub\facebookresearch_dinov3_main`；本环境无 `dinov3` 包。其他未核验位置为 UNKNOWN。 |
| 本地权重 | `C:\Users\rico\.cache\torch\hub\checkpoints\dinov2_vitl14_reg4_pretrain.pth` 存在，1,217,607,321 bytes；P1-A 已验证 SHA-256 `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`；本轮只核对了存在及大小，没有重新哈希或加载。 | 未找到 `C:\Users\rico\.cache\torch\hub\checkpoints\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`，也未找到 `C:\Users\rico\.cache\huggingface\hub\models--facebook--dinov3-vitl16-pretrain-lvd1689m`；没有本地 SHA-256。官方 Hub short hash `8aa4cbdd` 不是 SHA-256。 |
| 输入/特征依据 | P1-A 的缓存是 448 resize 后 center crop 392、block-18 CLS、L2、FP32；不能当作 E2 letterbox/patch cache。 | 官方代码标识 ViT-L/16；448/16=28，每图空间 patch 784。官方实现定义 1024 维、4 个 storage tokens。 |
| 输出 tokens | ViT-L 特征维度 1024，patch 14；有 CLS、空间 patch 和 4 个 register tokens。 | 1024 维，patch 16，4 个 storage tokens；空间 pooling 应只使用 patch tokens。 |
| License | 官方 model card 标为 Apache 2.0。 | 独立 DINOv3 License Agreement；使用/分发受其条款约束，研究发表须致谢使用，且需符合贸易管制等条款。不是 Apache 2.0。 |
| 环境适配 | P1-A 推理已证实此 checkpoint/环境可 CPU 运行；E2 的 letterbox 448 和 spatial-token 输出仍需单独 smoke test。 | PyTorch 已安装，但本地源码/权重没有核验；HF Transformers 4.38.1 低于 README 4.56.0 支持版本。现为 **`E2_BACKBONE_NOT_READY`**。 |

DINOv3 官方 repo 提供 ViT-L/16 构造与本地权重路径；配置中的 short suffix 不能替代 checkpoint 完整 SHA。**不下载、不替换 backbone，也不自动转用 Transformers 或其他 DINO 版本。** 若 DINOv3 仍不可用，E2 的双 backbone 比较应保持阻塞，不能静默缩成单 backbone 后声称符合 v0.3。

官方参考：[DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)、[DINOv3 README](https://github.com/facebookresearch/dinov3/blob/main/README.md)、[DINOv3 backbone 定义](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py)、[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)。DINOv3 官方 README 称加载模型所需依赖为 PyTorch，并建议安装 CUDA 版；README 的 HF Transformers 集成从 4.56.0 起支持。License 要求发表研究时承认其使用，使用时受适用法律与贸易控制要求约束；本文件不是许可法律意见。

## 3. E2 拟注册的共同输入和编码规则

以下均是**待批准的工程提案**，不表示已实现/验证。

1. 只使用已授权的同一 102 张历史 source-side 图像（54 D-fit + 48 D-dev）；每个 backbone 单独建立缓存。P1 缓存不改、不复用为 448 特征。无 mask、未选视角或新图像访问。
2. RGB 转 float `[0,1]`。保持宽高比，把最长边缩放到 448、使用 antialias；置中补到 448×448，奇数差值多出的像素固定放右侧/下侧。不裁切对象。
3. 用官方 web-image ImageNet mean/std `(0.485,0.456,0.406)/(0.229,0.224,0.225)` 归一化。归一化后用零值补边（等价于补原始 mean 色）；如模型官方特定预处理在冻结实现中不同，必须先登记并公开差异。
4. DINOv2 网格是 32×32；DINOv3 网格是 28×28。位置编码按各自官方实现：DINOv2 对绝对位置 embedding 做对应网格插值；DINOv3 用官方 2D RoPE 坐标路径。虽然输入尺寸可整除 patch size，必须由 E2-A smoke 证实两个实际 checkpoint 都接收 448，并锁定插值/坐标细节。
5. pooling 有效 patch 规则：以 patch 中心是否落在 resize 后的有效图像矩形内决定。中心在内的部分补边 patch 计入，中心在外的 patch 排除。排除补边 patch 只约束最终 pooling，不消除这些 token 在 transformer attention 中与有效 token 的交互；报告中必须保留此限制。
6. 从模型最终层官方归一化输出取 CLS 与空间 patch token；排除 4 个 register/storage token。转存 FP32。对 CLS 和每个 patch 做逐 token L2 单位化；G-Mean 为有效单位 patch 的算术均值并再次单位化；G-Concat 拼接单位 CLS 与单位 patch 均值后整体单位化。所有单位化均在一组特征上独立完成，不用 D-dev 统计量。
7. G-CLS / G-Mean 头宽度 1024；G-Concat 2048；MIL 每个 patch 1024。图像 bag 只包含该图像有效空间 patches，bag 级标签只来自该图像的合规二元属性状态。

### E2-A 必需 smoke（需另行授权后才可做）

在无 protected data 的授权 source-side 样本上分别验证：输入 shape、RGB/归一化与 padding 约定、模型版本/权重 SHA、输出 `CLS=(1024,)`、patch 网格与有限值、register 排除、有效 patch mask、重复运行确定性、448 动态网格/位置编码路径、FP32 输出，以及单例原始特征分数与 batch 无关。记录 CPU/GPU、torch/torchvision、源码 tree SHA、权重 SHA、预处理 fingerprint。未通过则停止，不改 crop、网格、归一化或阈值去追求好结果。

## 4. 特征缓存和资源估算

全部按 102 图像、1024 维 FP32 估算，未包含模型权重、临时工作区、目录/manifest 开销或 attention activation。CLS 与 patch 单独存储时：

| Backbone | CLS | Spatial patch | 两者合计 |
|---|---:|---:|---:|
| DINOv2，32×32=1,024 patches | 417,792 bytes（0.398 MiB） | 427,819,008 bytes（408 MiB） | 408.40 MiB |
| DINOv3，28×28=784 patches | 417,792 bytes（0.398 MiB） | 328,204,288 bytes（312.94 MiB） | 313.34 MiB |
| 两个 backbone | — | — | 约 721.74 MiB FP32 |

建议另存每图 patch-valid mask、源文件 SHA、输入尺寸/变换 fingerprint、数据版本和安全的 opaque ID；标签和 evaluator-only source metadata 不进特征 payload。所有分片和 index 保存完整 SHA-256，标记 local/ignored，绝不覆盖 P1 cache。

P1 cache 的输入 fingerprint 是 `820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63`，P1 feature index 是 `ee0232b29f64f9fd4f10198ba3ef20da89fa85b2083a389b774f6a9723bbb99d`；其 392 crop 与未保留 patch features 不满足 E2 的 448 全视场要求。E2 将需重新访问那 102 个已授权图像像素并提取两个新缓存，但这项访问和 GPU 时间仍需独立明确授权。12 GPU 小时是预算上限，不是已证明可行的耗时。
