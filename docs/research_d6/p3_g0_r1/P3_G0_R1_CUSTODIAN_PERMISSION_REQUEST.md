# 给数据保管方的权限申请草案

状态：未发送、未批准；2026-11-20；Last Paper / LP-D6-P3-G0-R1。请研究负责人先审核。本草案不凭 Git 仓库访问权推断任何原始数据权限。

拟请求受权保管方为 Last Paper 开发性研究导出 **identity-only metadata**，范围为先前 P1 已获准元数据范围中、P2 六个产品以外的以下 12 类：`regulator, rolled_strip_base, switch, tape, terminalblock, toy, toy_brick, usb, vcpill, wooden_beads, woodstick, zipper`。优先原 `test` 分区；如需提供 `train` 分区，须明确单列授权与来源。请求的是候选资格审计所需身份，不是将这些产品称为全球首次使用或独立外域。

请在单独审查授权后，按 [身份清单规范](P3_G0_R1_IDENTITY_MANIFEST_SCHEMA.md) 提供不透明图像 ID、不透明组 ID、产品、视角、原 split、现有可合法复用的 source SHA 或缺失状态、开发曝光状态、研究角色、授权/排除记录及组身份置信度。禁止导出属性标签、defect type、异常 mask、源图像、源原始路径、原始组键或可反推 source type 的字段。原始键与随机 ID 的映射由保管方保留；如果使用密钥伪名，密钥不交给研究者。

请保管方在其合法边界内执行 D-fit、D-dev、P2 已曝光对象与 separately protected 清单的交集核查，只返回每个不透明组的 `CLEAR / CONFLICT / UNKNOWN` 及明确检查范围、负责方和记录版本；不返回 protected 数据身份。没有合法检查权限或清单时请返回 UNKNOWN，而不是根据其他数据源不同就填 CLEAR。可识别的近重复和组冲突请分别返回已核范围及状态；不能确定真实工件独立性时报告 INFERRED/UNKNOWN。请说明既有 source SHA 能否用于跨清单连接；本申请不包含打开图像或计算新摘要。

请另提供 FSAD 与 Last Paper 已有使用记录的**非敏感产品级或逐不透明身份交集摘要**；如无权共享或无法映射，请标 UNKNOWN，不披露受保护样本。P1 曾对 18 类元数据曝光；FSAD 使用 Real-IAD 是研究者报告，不能自动推断每个文件是否重叠。

拟要求的授权档案至少含：责任方、批准/生效日期、到期或撤销状态、版本、具体产品与分区、允许字段、允许研究角色、排除核查范围、已知限制、导出清单 SHA-256、保存位置与保管期限。为避免自证，研究者须通过独立渠道核实档案及清单摘要。若确认可复用 source SHA，明确它的再分发和连接权限；未确认时不使用。

此项申请只覆盖身份清单导出与验证。源图像读取、标注、DINO 推理、GPU/磁盘使用、D-cal 标签选择及 D-eval 评价均需另外分别批准；D-conf/sealed/HOLD/Probe/Mint/PCB 不因本申请而解除。
