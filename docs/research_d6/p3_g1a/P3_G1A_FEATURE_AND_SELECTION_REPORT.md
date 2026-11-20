# P3-G1A 冻结特征与无标签分层报告

状态：`P3_G1A_READY_FOR_ANNOTATION_APPROVAL`

Git：`stage3-eg-famar-execution` / `e2c0bd0113327173bf5d131f7e0cd8a5a826c8ea`；授权记录：`RESEARCH_LEAD_CHAT_APPROVAL_20261121_P3_G1A`。

## 执行摘要

白名单 216 张；原图 SHA 成功 216 张、失败 0 张；DINOv2 特征成功 216 张。
H12 重建差异最大值 0.0（容差 1e-8）；历史纠正锚点核验 5/5。
最终开发角色 60 组 / 120 视角；D-cal 20 组，D-eval 40 组。

## 产品级聚合

| 产品 | SHA 成功 | 特征成功 | Representative 组 | Stress bin | 最终组 |
|---|---:|---:|---:|---|---:|
| switch | 36 | 36 | 6 | PASS | 10 |
| terminalblock | 36 | 36 | 6 | PASS | 10 |
| toy_brick | 36 | 36 | 6 | PASS | 10 |
| usb | 36 | 36 | 6 | PASS | 10 |
| woodstick | 36 | 36 | 6 | PASS | 10 |
| zipper | 36 | 36 | 6 | PASS | 10 |

## 预注册 Stress bin 覆盖

| 产品 | 低置信低距 | 低置信高距 | 高置信低距 | 高置信高距 | 状态 |
|---|---:|---:|---:|---:|---|
| switch | 2 | 4 | 4 | 2 | PASS |
| terminalblock | 3 | 3 | 3 | 3 | PASS |
| toy_brick | 3 | 3 | 3 | 3 | PASS |
| usb | 2 | 4 | 4 | 2 | PASS |
| woodstick | 4 | 2 | 2 | 4 | PASS |
| zipper | 4 | 2 | 2 | 4 | PASS |

## 权限与解释边界

本任务读取的候选只登记为新论文 D-dev 开发数据；旧 protected 集合的逐文件交集仍为 UNKNOWN。它们及可识别近重复不得进入未来 D-conf。未读取新人工属性标签、GT 缺陷 mask 或旧 protected 存储；未训练或更新模型。Stress 分层仅使用冻结特征、H12 分数、旧的五个合法纠正锚点和固定身份排序。

本报告不评价属性准确率、修复/损害或模型泛化；Stress 组不代表自然发生率。

私有执行目录：`outputs/research_d6/p3_g1a/run_20261121_01`（已由 Git 忽略并限制 Windows ACL）。

## 运行及完整性

私有输出约 1.45 GB（低于 3 GiB 上限）；GPU 峰值已分配 1,275,088,896 bytes；完整特征续跑用时 58.9s。首轮 29.7s 在特征适配器导入失败后停止，未生成特征；续跑复用已校验的 canonical 缓存，没有再次打开原图。
输入 manifest、模型、代码与逐样本 SHA 收据均保存在私有输出中。

## 执行恢复记录

首轮适配器将共享 token 接口从错误目录导入，导致 216 个特征调用均失败；原图 SHA 和 canonical 预处理已完成并保存。修正导入路径后，第二阶段逐项核验 216 个 canonical bundle 的文件哈希、数组哈希及 source SHA 绑定，再生成全部特征。首轮错误日志、失败清单和收据均保留，没有覆盖。完整错误细节见私有执行目录中的首轮日志与 `run_failure.json`。
