# R2c 验证记录

本地 R2b 输入 SHA-256：提取收据 `3aa6d1e7a8b10c426e462adb0f5085281acf62d4cc480dd654328f07fc5f49d6`；独立结构核验 `08348e5378e2ba3fde51b8b8cf5597e4c36fcaabbad1b4f110ca762a34430ecd`；条件排序 `38abc314eac0b4d79405e947a87f9cbe5b1fbe619efe7c5961cbdf8d345fe50d`；受控来源映射 `6b0754dd466a6a29dff43004a4853489eb80871fa84e3ec1ae06324b9d83df78`。十二份分片均与两份 R2b 收据所载摘要一致。上述摘要是现有本地工件的固定身份和内部一致性证据，并非独立签名的保管证明。

已检查完整组/视角、身份唯一性、全量条件排序、所有原候选 UNKNOWN、空 source SHA 和无正式分区声明。选择结果：6 产品、108 组、216 个不重复视角，按既有确定性算法产生。私有选择收据 SHA-256：`cb5e3eb303444a67e22f47a6cc146cc72ece58a3bb0bb612bee389006728d9e6`。全量后备排序由原排序及其上述 SHA 固定；补充后备规则和曝光比较分别记录在独立私有文件，且有各自摘要，不属于原选择收据的覆盖范围。

新脚本 SHA-256：`5f58cbfd8a5c352f2bf96e1ac16706dc66369da7c88ac28fce302526da1e24a1`；新合成测试 SHA-256：`503ea37f6935dad19f346a5934bbae759acd39cd62aab88ac27aeac785bf2246`。R2c 合成测试 19/19 通过；既有 R1、R2、R2b 合成测试分别 24/24、17/17、24/24 通过。测试使用虚构数据；未运行 R2b 历史提取器。

输出目录受 `.gitignore` 的 `outputs/` 规则保护；Windows ACL 已核验限制为当前用户、SYSTEM 和 Administrators。以上不是权限或内容去重证明。本次没有打开图像、mask、特征、人工属性标签或 protected 数据，没有计算图像 SHA，没有使用 GPU、训练或标注。
