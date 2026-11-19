# P3-G0：分区、抽样和标注契约草案

日期 2026-11-20；HEAD `55be96accbd25dffed916d8c6add18f731006c6f`；状态 `BLOCKED_PERMISSION_METADATA`。本文件只是规则草案，不构成数据读取、标注或特征提取许可。P3-G0 v0.2 独立协议文件未在本地找到。

## 资格与冻结次序

先取得经授权的 identity-only manifest 及非敏感排除记录。以 manifest 内容 SHA-256 为快照，记录来源版本、权限和曝光状态；不得读取属性标签、缺陷类型、mask、H 的结果或隐藏来源字段。资格要求产品不属于 P2 的六个开发评价产品；每产品至少 10 个满足权限、独立角色及两不同视角要求的登记组；全组及可识别近重复在同一角色；原 source split 不更名。对于保护集重叠无法验证的条目保留 UNKNOWN，并由研究者在后续权限决定中明确是否可作开发用途；不得自动归为排除已通过。

逐对象不具可信 source SHA 时保留缺失并暂停该对象的身份核验；若不存在可读且已批准的逐件排除清单，不声称文件级无重叠。P1 的 12 个名义候选产品不是已核验的 12 个合格产品。当前没有名单，以下算法没有在实际候选上执行。

## 产品、组和视角的确定性方案

固定盐为 UTF-8 字符串 `LP-D6-P3-G0-v0.2-20261120`。数据保管方先提供不含来源类别的稳定不透明组 ID，负责将完整 source key 映射为唯一组并保留受控映射；研究抽样程序不读取编码了来源类型的 full key。所有抽样字段先按 Unicode NFC 标准化，禁止字段中包含 U+001F；组键为 `product + U+001F + opaque_group_id`，保证产品边界。SHA-256 输入使用 UTF-8：`salt + U+001F + domain + U+001F + canonical_key`；升序按 32 字节摘要比较，摘要相同时按 NFC 后 UTF-8 字节序的不透明键比较。domain 分别为 `PRODUCT`、`REP_GROUP`、`STRESS_TIE`、`VIEW`、`REPEAT_GROUP`，避免交叉排序依赖。

`PRODUCT` 的 canonical key 是 product；`REP_GROUP`、`STRESS_TIE`、`REPEAT_GROUP` 使用上述组键；`VIEW` 使用组键再连接 U+001F 和 view。展示顺序的独立 presentation nonce 必须从不含身份/标签的预先固定序列生成，重复视角使用不同 nonce，不能从人工标签生成。

1. 只对完成资格审核且每类至少 10 组的产品按 `PRODUCT` 排序，取前 6；第 1–2 个产品整类为 D-cal，第 3–6 个整类为 D-eval。少于 6 类则停止，不用标签或模型结果补齐。
2. 在每个产品已授权的总体候选组中，按 `REP_GROUP` 排序，取前 6 组代表层；代表组即使困难也不能按标签或模型结果换掉。若样本组或批次凭证显示跨 full key 同属一个物件，合并成单一不可拆组再排序；若身份不能排除碰撞，记录 UNKNOWN，不称物理独立。
3. 其余合法组作为后续 stress 候选池，最少每产品 4 组；建议至少 12 组以容纳跨 bin 的选择。该池在 G0 未提特征，故 `STRESS_SELECTION_PENDING_FEATURE_AUTHORIZATION`。未来另获许可后在标签不可见的阶段，针对冻结 H12 的 `|logit|` 及混合距离，以各产品合法候选池的无标签中位数定义二值低/高区间，形成四个交叉 bin；每 bin 按 `STRESS_TIE` 顺序取 1 组。用相同已授权特征确定阈值，无须属性真值。若任一 bin 空、候选不足或候选池不足 12 组，停止该产品并报告，不能事后合并 bin；研究者可另行批准一次预注册修订。Stress 池的预测前筛选不能读取未来标签或 source defect type。
4. 每个被选组的合格视角按 `VIEW` 哈希排序选前两种不同 view，分组后选视角；缺两种则该组不合格并按已冻结规则重审，而不是依据视觉容易程度替换。视角关联同组必须同 D-cal/D-eval。记录 full key、视角、摘要、排序和排除理由于受控私有清单。
5. 每产品 6 representative + 4 stress 组；目标 D-cal 2×10=20 组、D-eval 4×10=40 组。两层分开报告；representative 按抽样设计报告，stress 为富集诊断，不直接估计总体发生率。若未来需要合并，必须先获得各产品/各层抽样框大小与包含概率，预注册逆包含概率加权；不能以未经核验的比例权重合并。
6. 盲重复按 `REPEAT_GROUP` 对 60 组排序取前 12 组；每组两个视角都重复，重复顺序另以该域和 presentation nonce 散列打乱，不能向标注者显示重复关联。重复仅衡量一致性，不增加独立组数。

## 四态与标注字段

保留冻结 `surface_abrasion_visible` v1.1：`present`、`absent_assessable`、`unobservable_or_uncertain`、`not_applicable`。未标注、U 和 N/A 均不能编码为 absent。每视角单独判断，不能用另一视角多数投票补真值，也不能以 defect type 代替属性。标注界面不暴露原产品、split、来源标签或重复身份。

建议 CSV：`annotation_event_id, opaque_image_id, product, sample_key, view, role, stratum, attribute_state, confidence, annotator_id, blind_repeat_flag, elapsed_seconds, source_provenance, exposure_status, annotation_version`。产品/组/来源字段由程序写受控 manifest，盲界面隐藏；原始逐项状态留私有输出。记录位置点仅作为额外可选字段且需单列成本，不能当作属性区域 mask。

## 成本与资源

60 组×2 视角 = 120 次基础标注；12 组×2 视角 = 24 次盲重复，共 **144 次呈现**。假设每次分别 1/3/5 分钟，纯呈现时间约 **2.4 / 7.2 / 12.0 小时**。这是情景假设，不是实测；准备、图像切换、复核和争议裁决分别 UNKNOWN。原计划 ≤40 人时是整体包络，已使用与剩余人时 UNKNOWN，因此目前无法证明总成本通过。

现有 102 项缓存属于旧 D-fit/D-dev，不可据此保证新候选已缓存。按照合法名单匹配后，新样本视角最多 120 张可能需要另行授权提取；若重复不需重复提特征。历史 E2-A1 的 102 图 DINOv2 31.75 秒和全双 backbone 输出约 1.007 GB 仅为资源参考；具体 GPU、磁盘、输入路径、图像 SHA 与后处理校验必须在 G1 前另获批准。
