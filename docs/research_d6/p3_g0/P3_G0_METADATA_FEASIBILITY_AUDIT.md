# LP-D6-P3-G0 v0.2：仅元数据可行性审计

审计开始时间：2026-11-20 08:27，Australia/Sydney。分支 `stage3-eg-famar-execution`，HEAD `55be96accbd25dffed916d8c6add18f731006c6f`。工作区的相关已跟踪输入无未解释改动；原有未跟踪文件未修改。`LP_D6_P3_G0_Correction_Applicability_Protocol_v0_2.md` 未在本地检索到，状态 `PROTOCOL_V02_NOT_LOCAL`；本报告依据本次指令和仓库 v1.1，未声称读取了该文件。

**决策：`BLOCKED_PERMISSION_METADATA`。** P1 的已授权聚合元数据表明，P2 六个产品之外的 12 个产品名义容量充足；但聚合表不提供可被本轮合法使用的逐组、逐视角 identity-only 候选清单，也未给出逐件曝光及排除关系。当前合格产品、合格组、可选视角的实际数量均为 **UNKNOWN**，不能将 12 个名义产品写成 12 个已通过资格审查的产品。未生成候选身份名单。

## 可核查的容量

以下为对 `Source_Metadata_Coverage.csv` 的产品/原分区数值求和，**RECALCULATED**；每组的五视角完整性来自 P1 已存统计，非本次重新读取 JSON。所有 12 类均在 P1 的 18 类授权元数据曝光内，标记 `NEW_STUDY_DEV_EXPOSED`；这里没有发现足以证明这些类别在全部历史工作中未曝光的证据。P1 记录这些类别当时没有读取像素，不能据此断言其他研究从未读图。组键为操作性分组，物理工件独立性 **INFERRED**。

| P2 六类之外的产品 | train 组 | test 组 | 每组记录视角数 | 组身份与合规资格 |
|---|---:|---:|---:|---|
| regulator | 249 | 771 | 5 | UNKNOWN，需逐组 identity-only 清单 |
| rolled_strip_base | 249 | 761 | 5 | 同上 |
| switch | 246 | 764 | 5 | 同上 |
| tape | 250 | 770 | 5 | 同上 |
| terminalblock | 244 | 766 | 5 | 同上 |
| toy | 250 | 755 | 5 | 同上 |
| toy_brick | 248 | 754 | 5 | 同上 |
| usb | 246 | 764 | 5 | 同上 |
| vcpill | 252 | 753 | 5 | 同上 |
| wooden_beads | 218 | 783 | 5 | 同上 |
| woodstick | 251 | 750 | 5 | 同上 |
| zipper | 251 | 750 | 5 | 同上 |

12/12 名义产品各有至少 10 个完整五视角 test 组；这是 **原始分区聚合上限**，不是完成授权和去重后的 eligible 数。此前 18 产品合计 18,103 个组、90,515 条视角记录，其中 train 4,420 组 / 22,100 视角，test 13,683 组 / 68,415 视角。P1 聚合记录无缺失或重复视角组；其物理工件 ID 和批次独立性未得到采集凭证。P1 文档指出基础四位序号可能跨状态/type 复用，不能只用该序号归并物件；也不能把包含状态/type 的不同 full key 当作已证实不同工件。

目标为 **6 产品、60 组、120 基础视角**。名义数量缺口为 0；经合法逐件排除后可确认容量缺口为 **UNKNOWN**。当前可形成合格身份名单的产品为 **0 个已核验**，这是证据缺口，不是断言数据里不存在合格产品。最低补救：由数据权限负责人提供覆盖上述已授权 18 类中候选范围的 identity-only manifest，至少含产品、不可逆或受控 full group key、view、原 split、公开来源身份/已有 SHA、权限与曝光标志，以及可用的已授权排除记录；不含属性标签、defect type、mask 列或受保护身份。再按该清单做去重和权限复核。

## 来源、字段及访问记录

本次只读取研究文档、汇总 CSV 和现存授权清单元数据；没有读取任何特征张量、原始图片、mask 或 Real-IAD 产品 JSON。以下 SHA-256 是文件身份哈希，不是新计算的图像哈希。

| 来源 | SHA-256 | 范围与可用性 |
|---|---|---|
| `docs/Last_Paper_Four_Month_Research_Execution_Plan_2026-11_to_2027-02_v1_1.md` | `1a3b83fa46dcc315963ad2a17deb019a6d01cc596193f5de067db13723fd0844` | 反馈权限、分组、预算和停止纪律 |
| `docs/research_d6/p0/D6_P0_DATA_AND_EXPOSURE_AUDIT.md` | `2a47947ab6e8194a2dcf18fcd741e09abcfb8817c31174bb435706c717c0433f` | 54 图、18 类元数据及 120 视角历史曝光 |
| `docs/research_d6/p2/g0/D6_P2_G0_DATA_AND_FEEDBACK_AUDIT.md` | `8ac8a528738e3e48f92f7c375e0a3628b0122f2b0c25d29a865175d800b58f89` | X 反馈 7P/5A，替代 P1 Gate0 旧 6/6 文字 |
| `docs/research_d6/p1_gate0/D6_P1_DATA_SPLIT_AND_TAPE_DRAFT.md` | `66fbeef246eb3af71960a2248e00bd160113c813712c5b3ac6c2939b318be2c4` | W/X 同组隔离；其中 X 6/6 数字 **SUPERSEDED** |
| `docs/research_d6/p2/e2a1/D6_P2_E2A1_PROTOCOL_AND_DATA_AUDIT.md` | `32c1d2eb4c3737d4e7ba05a82ff806baba258b9e62699e4b29a818d33fe9513b` | 102 已缓存输入及处理身份 |
| `research_pilot/LP_W1_D5R_SOURCE_METADATA/Source_Metadata_Coverage.csv` | `6f8c98367a6d8eb0c0792864c7cd66074a82d32b8181a2f26242d92ac120861a` | 已授权 18 产品聚合；含来源代码字段，仅读产品、分区及组/view 聚合列，不按代码筛样；不含逐对象清单 |
| `research_pilot/LP_W1_D5R_SOURCE_METADATA/P1_Source_Metadata_Exposure_Ledger.csv` | `d16dfb81f414b5197e80e955b50e636cc2eb69f5b39a3fdb5bad1fa3486f8c90` | 18 条产品级 JSON 元数据曝光收据；不是逐件授权名单 |
| `outputs/research_d6/p2/e2a1/registered_input_manifest.csv` | `820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63` | 102 旧输入身份：54 D-fit、48 D-dev；含本机路径字段，不公开行级内容 |
| `outputs/research_d6/p2/e2a1/dinov2/feature_manifest.csv` | `3588bab021a9cd8043d663c1d6ce46f366f89f89802fb1ac928dfae3360efddf` | 102 条历史 feature ID 与输入 ID 集合相等；未加载特征 |
| `research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/exposure_ledger.csv` | `a274bbbdae3a1e4c1822c2968458d9c3878a1f466e82177632be1c5b449d3306` | 已有 120 视角曝光台账；检查字段/文件身份，含隐藏来源类型，不作为新候选输入 |
| `research_pilot/LP_W1_D5R_IMAGE_PILOT/candidate_pool_freeze.csv` | `258975e3d31c3014e02b8b6bdbf87f6835fe61f2671e2c88bebd0b2e7ed9ba96` | 旧 66 组仅覆盖 P2 六类，含来源分层，不用于挑新产品 |

旧 66 组清单含来源分层及路径，本轮仅核验表头与文件哈希，没有按 defect type 选择候选。

此外直接阅读了 E2-B-R1 完整结果、D2 机制报告和纠正记忆回放报告，以及两份回放脚本；它们分别支持 H/M/M+ 的历史描述，不是新的模型运行。扫描限于本仓库指定文档、已知 pilot 元数据和现存缓存 manifest；未递归扫描数据根，也未读取另一 FSAD 论文的产物。

## 去重、曝光和跨项目边界

- 当前注册输入清单为 102 行、102 个不同 source SHA、0 个缺失 SHA；其中 54 D-fit 和 48 D-dev；102 个 ID 与 DINOv2 feature manifest 完全一致。六个 D-dev 产品各 8 视角。**RECALCULATED。**
- P0/P1 已记录 48 D-dev 视角是 P2 120 视角的子集，全部 120 个及可识别近重复排除于 D-conf；P0 已记录 D-fit 与 P2 来源路径/SHA 无匹配、D4B 内部 4 对 dHash 距离 ≤4。本次没有重新读图计算 dHash。
- 对 12 个名义新产品，P1 没有提供可合法用于本轮逐组连接的 identity-only 名单与逐视角 SHA，故与 D-fit 54、D-dev 48、P2 全 120 视角的逐文件去重为 **UNKNOWN**；跨源近重复和历史 sealed/HOLD/Probe 文件级重叠同为 **UNKNOWN**。不能把来源类别不同当作文件级零重叠证明。
- 研究负责人已说明 FSAD 使用过 Real-IAD，记为 **USER_REPORTED**；本次允许范围内未取得该研究针对这 12 类的产品级或文件级去敏使用清单。跨项目个体交集 **UNKNOWN**，也不能称之为 Last Paper 的全新独立外域。新于本次 D6 六产品开发 cohort 只表示产品角色不同。
- 没有检查任何受保护集合的样本身份或真值；保护资产访问 **NO**。未授权的原始 JSON、图像、mask、其他类别及旧 HOLD 状态继续关闭。

## 特征及资源

E2-A1 的现有 102 个缓存 ID 全属于历史输入；新候选尚无身份清单，无法做逐项缓存匹配，故新候选特征可复用数 **UNKNOWN**。若合法名单不在该缓存中，G1 必须另获授权读取指定源图像，使用冻结 DINOv2 及现有 canonical RGB letterbox、计算和记录输入/输出 SHA。

历史 E2-A1 记录：RTX 3090 Ti，DINOv2 102 张 worker 31.75 秒（含加载），模型特征生成时间合计 8.35 秒；全双 backbone 守护任务 119.39 秒、输出约 1,006,956,552 字节。这些是旧机器/负载的观测而非 G1 承诺。机械线性外推 120 张的 DINOv2 特征生成约 9.8 秒仅为理想算术量，整体墙钟、复核和存储需重新测量；历史双 backbone 总输出按比例约 1.18 GB，只是保守规划参照，不能当作 DINOv2 单独增量测量。未来须另批准 GPU、磁盘、源路径白名单与摘要校验。
