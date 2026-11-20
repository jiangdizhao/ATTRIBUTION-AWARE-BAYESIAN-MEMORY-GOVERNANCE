# P3-G0-R2b：分片验证与资格审计

日期 2026-11-21；状态 **`R2B_IDENTITY_EXPORTED_ELIGIBILITY_PENDING`**。

## 实际 R1/R2 验证

12 个 identity-only 产品分片分别以现有 R1 验证器对 D_CAL、D_EVAL 两种角色验证，清单摘要再次核对；每个分片的字段/结构检查通过，组状态全部 UNKNOWN。R1 的**单个分片整体状态为 INELIGIBLE**，因为每个分片只包含一个产品，不能满足其六产品总体容量门槛；这不表示该产品的全部组被认定有已知冲突。不得将分片级整体状态当作六产品分区结果。

额外流式聚合并复核 45,705 行：全体 9,141 个不透明组均有五个不同 view；跨分片图像 ID 冲突 0、组 ID 跨产品冲突 0；45,705 个 source SHA 全缺失，因此真实图像 SHA 重复关系 UNKNOWN，不将零个非空 SHA 写成“没有重复”。所有 manifest 哈希由本任务计算并在验证时重用，仅证明内部一致性；没有独立保管方认证。

R2 `partition_eligibility.assess` 使用全部 identity-only 行与 `relation_status=UNKNOWN`：输出 `status=UNKNOWN`，D_CAL、D_EVAL 各有 **9,141 个 UNKNOWN 组、0 个确认满足 10 组的产品、0 个确认满足 18 组的产品**，未生成正式 2+4 角色或代表组。两角色检查使用同一组集合，不能相加成 18,282 个独立组。实际名义产品 12、名义组 9,141，但缺少组级合法 source SHA、后续使用授权和独立排除/近重复证明；名义容量不能成为 PASS。

按 P3 固定盐、NFC、U+001F 及 SHA-256 对不透明产品和组创建 **`CONDITIONAL_METADATA_POOL`** 私有身份排序，覆盖 12 产品、9,141 组；只冻结无标签顺序，正式 D-cal/D-eval 角色数 0、`STRESS_SELECTED` 数 0。条件池的 HMAC 不透明 ID 与源键映射均为私有数据，不对外披露。新特征、H 推理和 stress bin 筛选仍需独立授权。

## 测试、代码和复现

| 测试 | 实际结果 |
|---|---:|
| 既有 R1 合成测试 | 24/24 PASS |
| 既有 R2 合成测试 | 17/17 PASS |
| 新 R2b 合成测试 | 24/24 PASS |

R2b 的合成负控改变非身份 defect label、mask 路径及其他字段后，导出身份和后续资格结果不变；也覆盖训练分区不导出、HMAC 域/密钥和顺序稳定、重复/视角冲突、UNKNOWN 不升级、白名单外产品、20 MB 限制与禁止图片/mask 文件读取。测试不证明原始组键内的来源类型编码不产生统计混杂。

| 工件 | SHA-256 |
|---|---|
| 真实提取时执行的 `extract_identity.py` 快照 | `ca8b28767a2a37df6dd920c3f049c7553961d8d480b037e6a70de9cdd6d2cb38` |
| 当前 `scripts/research_d6/p3_g0_r2b/extract_identity.py` | `941d65c0d94c76d79295c42065c1c32b29d9a107067651d7da865ec9ae1d3798` |
| `scripts/research_d6/p3_g0_r2b/validate_shards.py` | `4b354773aa329d1a362db6e1cdd0eee9ea9e1bb2ceaff626307f8c31dab92fd2` |
| `scripts/research_d6/p3_g0_r2b/conditional_order.py` | `02eccbd3738ce507f08ba831672ea4ad43e87fded35e45f856dfa3a265eff44c` |
| `tests/research_d6/test_p3_g0_r2b_extract.py` | `ab463d8421efc681209a807e53b8d249c6c9c319dcb49f60c7ea98464e8b4cb0` |

真实执行后的代码添加了 `--output` 和 `--reuse-from` 参数以允许以同一密钥在新目录复现；实际执行版本的逐字节快照已按 receipt SHA 核验并保存于私有目录。重新执行时不得覆盖现有输出；密钥与源映射访问仅限私有 ACL 下的获准人员。具体命令（需仍有对应授权）：

```powershell
D:\anaconda3\envs\torch\python.exe scripts\research_d6\p3_g0_r2b\extract_identity.py --output outputs\research_d6\p3_g0_r2b\reproduce_01 --reuse-from outputs\research_d6\p3_g0_r2b\run_20261121_01
```

后续 `validate_shards.py` 与 `conditional_order.py` 当前绑定原始运行目录；它们已对原始分片运行，不能直接以该命令验证复现目录。未经扩展不可宣称复现目录已验证。
