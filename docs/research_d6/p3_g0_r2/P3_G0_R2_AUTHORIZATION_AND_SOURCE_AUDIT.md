# P3-G0-R2：授权与来源审计

2026-11-20，Australia/Sydney；起始 HEAD `a61777203abb597c056d1ade7bf7de72a976492a`，分支 `stage3-eg-famar-execution`。本轮最终状态 **`R2_BLOCKED_SOURCE_ACCESS`**。

## 授权事实和范围

**VERIFIED：** 研究负责人在本次任务中明确授权 12 个既有 P1 元数据曝光产品、原 `test` split 的 identity-only 工作和去敏审计。不能再称为缺少研究负责人同意。声明不独立认证第三方许可、保管方排除证明或全面保护集无重叠，也不自动授权读取含未批准字段的完整混合 JSON。

本轮实际读取 P1 产品级曝光台账、历史分组结构与已登记清单的产品/身份范围、G0/R1 文档和 R1 验证器；不读取原始产品 JSON、图像、mask、特征、protected truth 或另一论文的私有结果。工作区相关已跟踪输入无未解释修改。旧 P1、P2 结果、HOLD 与 Git 状态未改变。

## 逐来源处置

P1 历史台账登记原始来源根为 `G:\dataset\realiad\jsons\realiad_jsons\`，以下每个文件均为此前在允许范围内登记的来源；**并非本轮打开或重新计算文件摘要**。P1 保存的 schema 表明记录同时有 `category/anomaly_class/image_path/mask_path`；无法仅读取身份字段而不读取完整混合 JSON 的字节。本轮未获得该混合字段完整读取权限，故以下 12/12 均为 `MIXED_JSON_READ_NOT_AUTHORIZED`；缺少逐件 identity-only 来源记为 `SOURCE_IDENTITY_UNAVAILABLE`。

| 产品 | 已登记相对 JSON 文件 | 历史 test 组 / 视角数 | 本轮状态 |
|---|---|---:|---|
| regulator | `regulator.json` | 771 / 3,855 | BLOCKED |
| rolled_strip_base | `rolled_strip_base.json` | 761 / 3,805 | BLOCKED |
| switch | `switch.json` | 764 / 3,820 | BLOCKED |
| tape | `tape.json` | 770 / 3,850 | BLOCKED |
| terminalblock | `terminalblock.json` | 766 / 3,830 | BLOCKED |
| toy | `toy.json` | 755 / 3,775 | BLOCKED |
| toy_brick | `toy_brick.json` | 754 / 3,770 | BLOCKED |
| usb | `usb.json` | 764 / 3,820 | BLOCKED |
| vcpill | `vcpill.json` | 753 / 3,765 | BLOCKED |
| wooden_beads | `wooden_beads.json` | 783 / 3,915 | BLOCKED |
| woodstick | `woodstick.json` | 750 / 3,750 | BLOCKED |
| zipper | `zipper.json` | 750 / 3,750 | BLOCKED |

**RECALCULATED：** 台账中 12/12 产品 test 汇总为 9,141 个操作性登记组、45,705 条视角元数据记录。它们不是本轮逐件处理量，也不是 9,141 个确认合格或独立工件。P1 的 18 类为 `NEW_STUDY_DEV_EXPOSED`，即已有产品级元数据曝光。

## 已检查的合法来源身份

| 来源 | SHA-256 | 本轮可用信息 / 许可限制 |
|---|---|---|
| `research_pilot/LP_W1_D5R_SOURCE_METADATA/P1_Source_Metadata_Exposure_Ledger.csv` | `d16dfb81f414b5197e80e955b50e636cc2eb69f5b39a3fdb5bad1fa3486f8c90` | 12 产品已登记 JSON 相对路径、历史来源摘要及分区聚合；非逐件 manifest |
| `research_pilot/LP_W1_D5R_SOURCE_METADATA/Source_Metadata_Coverage.csv` | `6f8c98367a6d8eb0c0792864c7cd66074a82d32b8181a2f26242d92ac120861a` | 已授权的产品/split 数量汇总；不含可导出逐件 ID |
| `research_pilot/LP_W1_D5R_IMAGE_PILOT/candidate_pool_freeze.csv` | `258975e3d31c3014e02b8b6bdbf87f6835fe61f2671e2c88bebd0b2e7ed9ba96` | 旧 66 组池中 12 目标产品行数为 0；含来源分层，不用于新产品选择 |
| `outputs/research_d6/p2/e2a1/registered_input_manifest.csv` | `820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63` | 旧 102 输入中目标 12 产品行数为 0 |
| `research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/exposure_ledger.csv` | `a274bbbdae3a1e4c1822c2968458d9c3878a1f466e82177632be1c5b449d3306` | 历史 120 视角曝光，含隐藏来源字段；本轮仅查看表头与文件摘要，不用作新候选 |

这些清单文件本身的 SHA 可复核；台账中每个原始 JSON 的历史 SHA 是历史记录，**当前文件内容和现时来源身份未验证**。没有搜索未登记的数据根、没有打开 12 个源 JSON 或任何受保护资产。

## 具体阻断与最低补救

需由已获授权的数据保管方在其控制环境中导出符合 R1 字段规范的去敏逐视角身份清单，附授权范围、已有合法 source SHA（没有则留空）、非敏感保护冲突/近重复状态及独立提供的清单 SHA；保管方不披露属性、缺陷类型、mask 或受保护身份。另一可行路径是经具有相应数据权限的授权方明确批准读取所列 12 个混合 JSON 的完整字节并规定字段隔离；研究负责人本次批准单独不足以推定第三方或保护数据权限。无论何种方式，未获核查的冲突和近重复保持 UNKNOWN，不能自动解除旧 HOLD。
