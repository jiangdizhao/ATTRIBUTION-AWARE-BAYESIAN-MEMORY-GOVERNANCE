# P3-G0-R2：身份提取与验证结果

2026-11-20；HEAD `a61777203abb597c056d1ade7bf7de72a976492a`。状态 **`R2_BLOCKED_SOURCE_ACCESS`**。

| 指标 | 本轮实际结果 | 证据状态 |
|---|---:|---|
| 历史登记名义产品 | 12 | VERIFIED：P1 台账 |
| 历史 test 登记组 / 视角记录 | 9,141 / 45,705 | RECALCULATED：产品级台账求和；非逐项新读取 |
| 本轮可合法访问的目标逐组/视角身份 | 0 / 0 | BLOCKED：没有已获准的 identity-only 逐项来源 |
| 被合法逐项处理的目标产品 | 0 / 12 | BLOCKED：12 个混合 JSON 未打开 |
| 导出的 manifest、身份映射与 SHA | 0 | BLOCKED；没有真实身份输入，不能生成伪匿名映射或伪造输出 |
| 已核验 PASS / UNKNOWN / INELIGIBLE / BLOCKED 组数 | 0 / UNKNOWN / UNKNOWN / UNKNOWN | 未执行逐组验证；不能把未读取的组计为已验证结果 |
| 实际可复用 source SHA 数 | UNKNOWN | 旧 JSON 来源摘要不是其内部图像文件 SHA；未读取图像计算摘要 |
| 保护冲突、近重复、跨项目交集 | UNKNOWN | 无合法候选逐项身份和完整非敏感排除凭证 |
| 物理对象独立性 | INFERRED / UNKNOWN | 历史 sample-key 操作性分组，不等同于有采集凭证的独立工件 |

没有已获准的目标逐项清单，故没有运行 R1 验证器的 D_CAL/D_EVAL 真实输入，也没有发生同进程 SHA 自认证。R1 现有验证器源码 SHA-256 `070fbfd71e6407bbe569cc333b13f7bf84b27ac7ccaf265a0f73a8c2d89bf5b3`。未来只能对显式获准的 `.identity-only.csv` 或 `.identity-only.json` 执行；新导出摘要若仅由同一进程计算并输入，最多证明内部一致，不称独立保管方认证。大于 `20,000,000` 字节时 R1 阻断；不能改上限或任意截断后冒充总体代表样本。

Git 忽略规则已核对：`outputs/` 忽略。由于不存在合法逐件身份输入，本轮没有创建私有身份目录、清单或映射。公开报告不包含逐件候选身份、属性、source defect type、图像路径或保管映射。
