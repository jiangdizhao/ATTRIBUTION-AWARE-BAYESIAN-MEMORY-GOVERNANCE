# P3-G0-R1：验证器与合成测试

执行环境：`D:\anaconda3\envs\torch\python.exe`；工作区起始 HEAD `a512d5ebce74374fb38cfa475ca54d71be736f34`。验证器版本 `p3-g0-r1-identity-validator-v1.0`。本轮没有输入真实候选清单；没有搜索或读取原始 JSON、图像、mask 或特征。

## 功能及证据边界

`scripts/research_d6/p3_g0/validate_identity_manifest.py` 只在调用者显式指定单个 `*.identity-only.csv` 或 `*.identity-only.json`、声明获授权、提供非敏感来源代号及独立获得的预期 SHA 后读取该文件。无输入时返回 `MISSING_AUTHORIZED_MANIFEST`，退出码 2；不搜索目录。文件名、顶层 JSON envelope 和严格字段白名单可阻断普通原始 JSON/图片或混入额外字段的输入。它不能证明一个被误命名的文件确实安全、保管方声明真实、排除范围覆盖全部封存数据或伪名具有密码学匿名性；授权人应先核对来源、签名/凭据及范围。

工具检验字段和枚举、全表图像 ID / SHA、跨产品组 ID、组内状态一致性、重复视角与最少两视角，按要求的 D-cal 或 D-eval 角色分别统计 PASS/UNKNOWN/INELIGIBLE/BLOCKED 组及产品。未知排除、未知近重复、来源摘要缺失均不计 confirmed。六产品各十个组是操作性容量门槛，不等价于物理独立或可以运行模型。只输出聚合数量、产品名、原因代码、清单/代码 SHA 和非敏感来源代号；不输出图像/组 ID、标签或原始路径。工具不生成样本选择或结果文件。

用法（仅在取得独立授权的真实清单和可信 SHA 之后）：

```powershell
D:\anaconda3\envs\torch\python.exe scripts\research_d6\p3_g0\validate_identity_manifest.py --manifest <经授权的.identity-only.csv> --role D_EVAL --authorized-identity-only --source-name <非敏感授权记录代号> --expected-sha256 <保管方独立提供的64位摘要>
```

同一个未分区清单分别运行 D_CAL、D_EVAL 容量检查，只能证明两个角色各自的潜在容量，不能把同一产品或组同时放入两侧。最终 2+4 产品的互斥分配仍需后续冻结并验证。本轮没有调用真实清单。

## 实际测试

```powershell
D:\anaconda3\envs\torch\python.exe -m unittest tests.research_d6.test_p3_g0_r1_manifest -v
D:\anaconda3\envs\torch\python.exe scripts\research_d6\p3_g0\validate_identity_manifest.py
```

第一条：**24 个合成单元测试通过、0 失败、0 测试执行阻断**。合成 fixture 内覆盖 PASS、INELIGIBLE、UNKNOWN、BLOCKED 和缺少清单；覆盖重复 image ID/SHA、缺 SHA、单/重复视角、跨产品组 ID、已知保护冲突、UNKNOWN、角色不足、少于 6 产品或 10 组、未授权读取、格式外字段、CSV/JSON、固定结果与文件摘要不符。图片和普通 Real-IAD 产品 JSON 文件名在读取前拒绝，测试用 mock 确认这些分支不调用文件内容读取。第二条实际输出 `MISSING_AUTHORIZED_MANIFEST`，退出码 2，为预期阻断而非测试失败。

此次测试只使用虚构名称、虚构组及图像 ID、临时合成 CSV/JSON；没有使用真实候选或人工属性标签。源代码 SHA-256：`070fbfd71e6407bbe569cc333b13f7bf84b27ac7ccaf265a0f73a8c2d89bf5b3`；测试文件 SHA-256：`9cad4f0bc3c8d63a24cc9d11bdc89913dcdc690335e88d4e75a3d9e47b82e095`。
