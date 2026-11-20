# P3-G0-R2b：来源与授权审计

日期：2026-11-21，Australia/Sydney。结果 **`R2B_IDENTITY_EXPORTED_ELIGIBILITY_PENDING`**。起始和结束分支 `stage3-eg-famar-execution`，HEAD `d0f75a45f10225a53874d659937c139e4c5a9951`；远端同分支通过 `git ls-remote` 查询为同一 HEAD。该提交的唯一父提交为 `a61777203abb597c056d1ade7bf7de72a976492a`，且仅新增 R2 指定的四个报告、一个检查器和一个测试文件。原有未跟踪材料保持不变。

## 授权与实际读取

研究负责人在 2026-11-21 明确批准对 P1 台账精确登记的 12 个产品 JSON 读取完整字节，授权记录代号 `RESEARCH_LEAD_CHAT_APPROVAL_20261121_R2B`；仅提取原 `test` 身份。**实际读取原始混合 JSON：YES，12/12。** 提取进程解析的字节包含来源标签和 mask 路径；这些字段没有用于抽样、学习监督或评价真值，也未写入 identity-only 清单。full sample key 本身可能编码来源类型，因此字段隔离与伪名化不证明统计去混杂。

实际图像或 mask 文件、D-conf、sealed/HOLD、Probe、Mint/PCB、其他产品 JSON、未登记的数据根及另一论文私有结果的访问均为 **NO**。未训练、推理、提取特征或标注。P1 台账 SHA-256 为 `d16dfb81f414b5197e80e955b50e636cc2eb69f5b39a3fdb5bad1fa3486f8c90`；12 个文件的当前 JSON SHA 与台账历史摘要均一致。JSON 的 SHA 是来源元数据文件摘要，**不是**图像 source SHA。

Real-IAD [官方数据集页面](https://huggingface.co/datasets/Real-IAD/Real-IAD)标注 CC BY-NC-SA 4.0、研究用途，并对文件访问设有条件；未发现当前研究目的与所示许可明确冲突。本次未登录或下载数据；研究负责人授权不替代第三方许可或独立保管方认证。

## 已核对来源与产量

从 P1 台账取得 12 个精确路径：`G:\dataset\realiad\jsons\realiad_jsons\<product>.json`，只按该白名单打开。各来源 test 结果、当前 JSON SHA 如下；每行 5 视角 × 组数均匹配历史台账：

| 产品 | test 组 | test 视角 | JSON SHA-256 |
|---|---:|---:|---|
| regulator | 771 | 3,855 | `a487d9516790709440c0545c52b260a8e0beb89bee6dc3bf783f6eee8c369d32` |
| rolled_strip_base | 761 | 3,805 | `c5eaa563cfe5faa01b31434c808b640851d1a988b0054e7819bb0065e250ae23` |
| switch | 764 | 3,820 | `b863b819fe7734a9c83acc26b25d419c8d8897a9ae93d5b8fd9d0e424ecf42d8` |
| tape | 770 | 3,850 | `00b763947dd5de805bf5647d4df4a514a39a2439f67c96236bfcc99b41b36feb` |
| terminalblock | 766 | 3,830 | `a0b242878f3ef61f978a710e665cdf74b2a521f92596b43c59ab5a3849bae760` |
| toy | 755 | 3,775 | `8f4060d38d0cdc34f7c34c7f280719a99f85fcc0926d6912f8323efdd481db10` |
| toy_brick | 754 | 3,770 | `6d06fdf7bd1e4d647c18ae3bc252d28cc6875b6091520efbdac17525f9018087` |
| usb | 764 | 3,820 | `475d4bc63dadefa65021dd1cba1fb3cd1839aec433fb1fd0e619b6dbfa2bbdbd` |
| vcpill | 753 | 3,765 | `95b65a765058f9d708895ec7b241aa27e943b4ffd51b8390ea5864494ecb2594` |
| wooden_beads | 783 | 3,915 | `c8943660294ef651614d038bede408a4eb55ac42e28d33dcc29bacacf8e49f25` |
| woodstick | 750 | 3,750 | `37d45076545df49ae2bdb65ceefc0b43ba51bbdb9adc19330f5ca531cbf24af8` |
| zipper | 750 | 3,750 | `0e7c1e6da4e5720fd29219d26383477b9c4b71c70d76f1c93fd0fbbb51e31631` |

所有来源结构检查通过；0 个来源隔离。JSON 原分区的 `train` 字节因完整解析而进入提取进程，但不参与候选导出。私有输出仅在 Git 忽略的 R2b 目录中；无新 Git commit/push。
