# P3-G0-R2：资格及产品分区审计

2026-11-20；HEAD `a61777203abb597c056d1ade7bf7de72a976492a`；实际状态 **`R2_BLOCKED_SOURCE_ACCESS`**。

## 实际容量

**RECALCULATED：** P1 产品级台账记录 12 个名义候选产品、9,141 个 test 登记组、45,705 条视角记录。历史 P1 汇总显示每组五视角，12/12 产品名义上超过 10 组和 18 组。它并不说明每个候选的文件 SHA、保护排除、近重复、授权或物理对象身份已解决。

**BLOCKED：** 本轮没有取得逐项身份，因此按 R1 的确认 PASS 产品/组和可确认的 10 组/18 组容量均不能计算；不是实际数量为零。D_CAL 和 D_EVAL 的潜在产品容量、相互隔离、每组两个可用视角、代表层和 stress 候选池数量均为 UNKNOWN。没有冻结 2+4 产品或任何组，也没有选出 `STRESS_SELECTED`。历史汇总不允许充当合法 identity-only manifest。

## 新增纯函数分区检查

`scripts/research_d6/p3_g0_r2/partition_eligibility.py` 不读取文件，也不自动发现或导出真实身份。它提供：

- 通过 R1 原验证规则逐组区分角色资格；只允许 `test`、至少两个不同视角、已知 SHA、明确权限与无已知冲突的操作性组计入相应角色容量；全局重复 SHA 的组不计入确认。
- 单独统计每角色至少 10 组和 18 组的产品数。产品按注册盐 `LP-D6-P3-G0-v0.2-20261120`、NFC、U+001F 和 SHA-256 稳定排序；前两产品为 D-cal、其余四个为 D-eval。若固定角色不满足资格，停止，不看标签重新挑选。
- 检查 2+4 产品集合互斥与组跨角色，要求由保管方另行声明已在候选总体内解决可识别跨组同物件/近重复关系；声明 UNKNOWN 时不输出冻结计划。`CLEAR_FRAME` 只是函数输入声明，不是独立保管方认证。
- 已确认组按身份规则选每产品 6 个 `REPRESENTATIVE_SELECTED`，其余为 `STRESS_POOL_ONLY`；不会输出 `STRESS_SELECTED`。18 组只是未来 6+12 池容量参考，不保证各无标签 stress bin 充足。内部私有计划包含不透明 ID，公共返回只含聚合数；不写文件、不读取原图。

该函数接收内存中的已授权清单行；调用方必须先明确身份读取许可、逐文件摘要和 R1 文件级检查，并将实际清单/代码/权限记录及密钥映射封存在受控目录。函数布尔参数不是来源认证。本轮未调用真实清单，未宣称它完成了实际候选分区。

## 合成测试及版本

执行 `D:\anaconda3\envs\torch\python.exe -m unittest tests.research_d6.test_p3_g0_r2_partition -q`：**17/17 通过**。覆盖合法 2+4、产品/组跨角色、缺组/视角、缺 SHA、保护范围与近重复未知或冲突、train 排除、容量不足、R1 文件大小上限、行序稳定、密钥伪名稳定、公共摘要不泄露身份字段、缺授权及不打开数据文件。

重新执行 R1 合成测试：**24/24 通过**。两套均为虚构数据，测试成功不代表真实资格通过。

| 程序 | SHA-256 |
|---|---|
| `scripts/research_d6/p3_g0_r2/partition_eligibility.py` | `9854aa0bae0a38b462bcfb456ee4a71eb006ff106376fac3693a23a794c62c65` |
| `tests/research_d6/test_p3_g0_r2_partition.py` | `293bc36758f006768105ff1b3dad1eaf253418a57aab2f27a1ed09054be7c7ac` |
| 既有 `scripts/research_d6/p3_g0/validate_identity_manifest.py` | `070fbfd71e6407bbe569cc333b13f7bf84b27ac7ccaf265a0f73a8c2d89bf5b3` |

没有加载 DINO、模型或图像，没有启动 G1 或标注。物理对象独立性仍是 INFERRED/UNKNOWN；受保护数据及跨项目文件级交集仍 UNKNOWN。
