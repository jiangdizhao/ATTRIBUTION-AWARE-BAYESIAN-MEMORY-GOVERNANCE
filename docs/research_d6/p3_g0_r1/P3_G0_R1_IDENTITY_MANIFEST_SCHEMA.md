# P3-G0-R1：Identity-only 清单规范 v1.0

状态：准备草案；2026-11-20；起始 HEAD `a512d5ebce74374fb38cfa475ca54d71be736f34`。本规范不授权导出、读取原始数据、生成候选样本或启动 G1。数据保管方须先审定权限、范围及历史排除记录。

## 文件和共同约束

一个显式指定的 UTF-8 CSV（文件名以 `.identity-only.csv` 结尾）或 JSON（`.identity-only.json`）供验证器读取；JSON 唯一顶层结构为 `{"schema":"lp.p3.identity-only.v1","rows":[...]}`。每行严格包含下表全部字段，字段值均为字符串；不得添加 defect type、属性真值、mask、路径、源完整键或其他字段。除明确允许空字符串的列外不允许缺失；未知枚举使用 `UNKNOWN`。导出前由保管方移除任何标签、来源编码和受保护样本身份。

不透明图像 ID 应为保管方用安全随机方法生成的 `IMG_` 加 32 个小写十六进制字符；组 ID 使用 `GRP_` 加 32 个字符。稳定跨视角映射由保管方保存；若采用密钥伪名，密钥和源键映射不能交给研究者。不可把对可枚举原键的普通 SHA-256 作为匿名化方案。格式校验不能证明 ID 的随机性、不可逆性或源键未被泄漏，保管方须单独保证。64 位十六进制 `source_sha256` 只复用既有合法摘要，可能使受限身份可链接，须在许可中明确；本任务不计算图像哈希。

| 字段 | 缺失 / 格式或枚举 | 约束、来源及访问 / 模型用途 |
|---|---|---|
| `opaque_image_id` | 必填；`IMG_` + 32 hex | 全表唯一；保管方产生；研究者可用于连接和输出身份，不是属性输入 |
| `product` | 必填；字母开头的 1–64 位字母数字/下划线 | 产品级角色；来自批准清单；只作分区/评价，除经契约授权的上下文外不直接输入模型 |
| `opaque_group_id` | 必填；`GRP_` + 32 hex | 全表跨产品唯一；同组视角相同；保管方映射；仅分区/分组 |
| `view` | 必填；`C1`–`C99` | 同组不可重复、合格组至少两种；来源 metadata；仅分组/取视角 |
| `source_split` | 必填；`train` / `test` / `UNKNOWN` | 同组一致；保管方声明原分区；UNKNOWN 待核；仅角色审计 |
| `source_sha256` | 可为空；否则 64 位小写 hex | 非空摘要全表唯一；既有来源、保管方许可；为空为 UNKNOWN；仅身份核验 |
| `exposure_status` | 必填；`METADATA_EXPOSED_ONLY` / `NO_PRIOR_PIXEL_EXPOSURE_WITHIN_SCOPE` / `PIXEL_EXPOSED_DEV` / `NOT_KNOWN_EXPOSED` / `UNKNOWN` | 保管方按已核范围声明；已开发像素曝光不进入新开发池；后两项待核；仅权限审计 |
| `allowed_research_role` | 必填；`D_CAL` / `D_EVAL` / `D_CAL_OR_D_EVAL` / `NONE` / `UNKNOWN` | 同组一致；保管方授权；OR 仅表示两角色可选之一，后续按组/产品冻结为单一角色；仅权限 |
| `permission_authority` | 必填；1–128 字符字母数字及 `_ . : -`，字母或数字开头 | 授权方代号；保管方声明；仅审计，不作模型输入 |
| `permission_record_version` | 必填；同上 token | 同组一致；与权限档案关联；仅审计 |
| `permission_status` | 必填；`ACTIVE` / `REVOKED` / `EXPIRED` / `UNKNOWN` | 保管方声明当前状态；非 ACTIVE 不可确认资格；仅审计 |
| `exclusion_status` | 必填；`CLEAR` / `CONFLICT` / `UNKNOWN` | 可逐视角报告；任何视角 CONFLICT 则整组排除，任何 UNKNOWN 则组不计确认；保管方不披露封存身份；仅审计 |
| `exclusion_check_scope` | 必填；`DFIT_DDEV_P2_AND_PROTECTED` / `PARTIAL` / `UNKNOWN` | CLEAR 要求完整指定范围与授权记录，否则 UNKNOWN；仅审计 |
| `exclusion_authority` | 可为空；否则 token | 清除声明的保管方；空为 UNKNOWN；仅审计 |
| `exclusion_record_version` | 可为空；否则 token | 清除声明的受控记录版本；空为 UNKNOWN；仅审计 |
| `near_duplicate_status` | 必填；`CLEAR_WITHIN_SCOPE` / `CONFLICT` / `UNKNOWN` | 仅复用已合法获得的近重复信息；UNKNOWN 不合格；审计字段 |
| `group_conflict_status` | 必填；`CLEAR_WITHIN_SCOPE` / `CONFLICT` / `UNKNOWN` | 保管方核查组碰撞/泄漏；UNKNOWN 不合格；审计字段 |
| `object_identity_confidence` | 必填；`VERIFIED` / `INFERRED` / `UNKNOWN` | 保管方给证据等级；即使通过操作组资格，也不证明物理独立；审计字段 |
| `source_manifest_version` | 必填；token | 来源导出版本，同组一致；保管方声明；审计字段 |

`DFIT_DDEV_P2_AND_PROTECTED` 的记录应明确检查 D-fit、D-dev、P2 已曝光视角和可依法检查的单独保护样本，以及相应近重复/组冲突判断的范围。它是保管方的范围声明，不保证检查到未提供的其他历史集合；未有明确且有效的授权及记录不可声明 CLEAR。跨来源近重复、跨项目身份交集仍可为 UNKNOWN。`METADATA_EXPOSED_ONLY` 与 `NO_PRIOR_PIXEL_EXPOSURE_WITHIN_SCOPE` 均为有范围的声明，不等于所有历史研究中绝未曝光。

## 资格和状态

验证器只检查声明及清单内部一致性；不认证保管方签名、权限、排除范围完整性或工件独立性。完整授权与 CLEAR 的每个组在同一角色内至少两种 view、SHA 已有且无重复、无已知冲突，才能计为操作性 **PASS**。`INFERRED` 或 `UNKNOWN` 物理独立等级可与操作性 PASS 并存，报告必须保留等级；不得写成已证实独立物件。

缺少 SHA、权限或排除声明未知为 **UNKNOWN**；明确冲突、失效权限、P2 六产品、已曝光开发像素或角色不允许为 **INELIGIBLE**；字段结构/ID、重复视角、重复 image ID、跨产品组键或组内部关键状态不一致为 **BLOCKED**。任一组 BLOCKED 时整体 BLOCKED；没有 BLOCKED 时，六个产品各 10 个 PASS 组可得操作性 PASS，不能将 UNKNOWN 组计入该数。少于六个确认产品但含 UNKNOWN 组足以达到六个时为 UNKNOWN，否则 INELIGIBLE。名义数量是行级组数，不等于资格数量。通过验证不冻结 D-cal/D-eval 角色，不授权 G1 或 D-conf。
