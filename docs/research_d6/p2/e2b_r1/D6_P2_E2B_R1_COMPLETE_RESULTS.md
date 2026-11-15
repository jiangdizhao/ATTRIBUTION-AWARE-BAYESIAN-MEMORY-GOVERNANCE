# D6-P2-E2B-R1 — 主矩阵结果

## 汇总口径

表中 BA 为固定阈值下、在可定义产品上计算的产品宏平均 Balanced Accuracy；AP 与 ROC-AUC 也先按产品计算，再对具备正、负例的产品宏平均。行内均为对应协议的 learner 轨迹间算术平均。X 有 14 个注册 learner，但只有 13 个到达 checkpoint 12；W 的 14 个 learner 均完成。B0 每个协议均为 14 个固定控制。

## 未来评价总体表现

| 协议 | 状态 | checkpoint | 轨迹数 | BA | 配对 AP | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| X | B0 | 0/12 | 14 | 0.5758 | 0.8495 | 0.7623 |
| X | learner | 0 | 14 | 0.5758 | 0.8495 | 0.7623 |
| X | learner | 12 | 13 | 0.6897 | 0.8885 | 0.8094 |
| W | B0 | 0/12 | 14 | 0.5446 | 0.8408 | 0.7455 |
| W | learner | 0 | 14 | 0.5446 | 0.8408 | 0.7455 |
| W | learner | 12 | 14 | 0.6607 | 0.8899 | 0.8125 |

checkpoint 0 的 learner 与匹配 B0 完全相同。checkpoint 12 相对匹配 B0 的逐轨迹差值：

| 协议 | 指标 | 平均差 | 改善 / 持平 / 下降 |
|---|---|---:|---:|
| X（13 对） | BA | +0.1017 | 6 / 6 / 1 |
| X（13 对） | AP | +0.0323 | 3 / 6 / 4 |
| X（13 对） | ROC-AUC | +0.0312 | 3 / 6 / 4 |
| W（14 对） | BA | +0.1161 | 14 / 0 / 0 |
| W（14 对） | AP | +0.0491 | 5 / 9 / 0 |
| W（14 对） | ROC-AUC | +0.0670 | 5 / 9 / 0 |

这些是小样本、固定协议的开发性结果，不是独立确认集估计。协议 X 未来集有 24 个视角，其中 21 个二元可评价（Present 12、Absent 9、Uncertain 3）；三个产品均有正、负例，BA/AP/ROC-AUC 在 3/3 个产品可定义。协议 W 有 24 个未来视角（Present 14、Absent 7、Uncertain 3），但 porcelain_doll 与 sim_card_set 的未来视角全部为 Present，故 BA 和非退化 AP 只在 4/6 个产品定义。

### 更新引起的分类变化

按 checkpoint 0 到 12 汇总 learner 对未来视角的预测变化：

| 协议 | learner 轨迹数 | 轨迹-视角预测数 | 预测翻转 | 二元样本翻转 | 纠正 | 伤害 |
|---|---:|---:|---:|---:|---:|---:|
| X | 13 | 312 | 58 | 52 | 42 | 10 |
| W | 14 | 336 | 93 | 79 | 66 | 13 |

以上按“轨迹 × 视角”计数，同一未来视角可在多个方法中重复出现，不是唯一图像数。Uncertain 视角可以出现分类变化，但不进入纠正/伤害二元统计。更新能纠正错误，也会引入新的错误。

## 表征与损失对照

下表为 checkpoint-12 的产品宏平均 BA / 配对 AP。MIL 两个 seed 取均值；全局 readout 每个 backbone/协议只有一条轨迹。

| 协议 | Backbone | G-CLS | G-Mean | G-Concat | M-BCE | M-Rank-Softplus |
|---|---|---:|---:|---:|---:|---:|
| X | DINOv2 | —（CP9: 0.417 / 0.763） | 0.706 / 0.872 | 0.500 / 0.807 | 0.694 / 0.891 | 0.694 / 0.891 |
| X | DINOv3 | 0.444 / 0.826 | 0.567 / 0.983 | 0.528 / 0.826 | 0.861 / 0.919 | 0.861 / 0.919 |
| W | DINOv2 | 0.688 / 1.000 | 0.688 / 0.875 | 0.688 / 1.000 | 0.563 / 0.833 | 0.563 / 0.833 |
| W | DINOv3 | 0.688 / 0.875 | 0.813 / 1.000 | 0.688 / 0.875 | 0.688 / 0.875 | 0.688 / 0.875 |

X 的 DINOv2/G-CLS 在 checkpoint 12 没有有效值；括号中的数值仅为失败轨迹最后有效 checkpoint 9，不计入 checkpoint-12 汇总。X 中 M-BCE 相比 G-Mean 的 BA 差异随 backbone 改变（DINOv2 −0.011、DINOv3 +0.294），AP 则分别 +0.019 与 −0.064。W 中两个 backbone 的 M-BCE BA 均低于 G-Mean（各 −0.125）。这不支持“MIL 一致优于简单全局均值”的结论。

在相同 backbone、协议和 seed 下，M-Rank-Softplus 与 M-BCE 在 checkpoint 12 的 24/24 未来预测类别一致；二者的 BA、AP、ROC-AUC 聚合数值也一致。原始分数不完全相同，最大绝对差约 `7.49e-05`。本矩阵没有显示 ranking loss 相对 BCE 的可测指标增益。

## 属性与来源类别诊断

属性是视角级人工首标；来源类别为 evaluator-only。下列数据不能当作完整数据集发生率。

| 数据角色 | 产品 / 来源分组 | Present | Absent | Uncertain |
|---|---|---:|---:|---:|
| D-fit | 全部 54 张 | 27 | 25 | 2 |
| D-fit | capsule | 9 | 8 | 1 |
| D-fit | metal_nut | 10 | 7 | 1 |
| D-fit | wood | 8 | 10 | 0 |
| D-fit | good/正常来源 | 2 | 21 | 1 |
| D-fit | defect 来源 | 25 | 4 | 1 |
| D-fit | good（正常来源） | 2 | 21 | 1 |
| D-fit | scratch 来源类型 | 18 | 0 | 0 |
| D-fit | 其他 defect type 合计 | 7 | 4 | 1 |
| D-dev | 全部 48 张 | 25 | 19 | 4 |
| D-dev | porcelain_doll | 7 | 1 | 0 |
| D-dev | sim_card_set | 6 | 2 | 0 |
| D-dev | toothbrush | 3 | 2 | 3 |
| D-dev | transistor1 | 3 | 5 | 0 |
| D-dev | u_block | 3 | 5 | 0 |
| D-dev | usb_adaptor | 3 | 4 | 1 |
| D-dev | OK/正常来源 | 4 | 15 | 1 |
| D-dev | defect 来源 | 21 | 4 | 3 |
| D-dev | CH 来源类型 | 5 | 1 | 2 |
| D-dev | HS 来源类型 | 5 | 0 | 0 |
| D-dev | OK 来源类型 | 4 | 15 | 1 |
| D-dev | AK/BX/QS/ZW 其他类型合计 | 11 | 3 | 1 |

D-fit 中全部 18 个 `scratch` 来源候选都被标为 Present，显示强来源混杂；但非 scratch 缺陷中有 7 个 Present，good 来源中有 2 个 Present。D-dev 的 OK 来源有 4 个 Present；CH 有 1 个 Absent、2 个 Uncertain；其他非 CH/HS defect type 也有人工 Present。因此当前样本中存在解耦例子，但抽样富集、样本量小且并非随机自然发生率，不能据此断言完整数据集没有混杂。

D-fit 置信度：High 47（Present 23、Absent 23、Uncertain 1），Medium 7（4、2、1）。D-dev：High 35（14、17、4），Medium 12（10、2、0），另 1 个 Present 的置信度为空。属性和 defect/source type 从未提供给 learner。

为保留 `Product × Source type × Attribute` 联合信息，下表每格为 `Present/Absent/Uncertain`，未列的组合计数为 0；它是 evaluator-only 汇总：

| 角色 / 产品 | 来源类型及 Present/Absent/Uncertain |
|---|---|
| D-fit / capsule | crack 0/1/0；faulty_imprint 1/0/0；good 1/6/1；poke 1/0/0；scratch 6/0/0；squeeze 0/1/0 |
| D-fit / metal_nut | bent 2/0/0；color 1/0/0；flip 0/0/1；good 1/7/0；scratch 6/0/0 |
| D-fit / wood | color 0/1/0；good 0/8/0；hole 2/0/0；liquid 0/1/0；scratch 6/0/0 |
| D-dev / porcelain_doll | CH 1/0/0；HS 1/0/0；OK 3/1/0；ZW 2/0/0 |
| D-dev / sim_card_set | CH 2/0/0；HS 2/0/0；OK 0/2/0；ZW 2/0/0 |
| D-dev / toothbrush | CH 1/1/2；OK 0/1/1；ZW 2/0/0 |
| D-dev / transistor1 | BX 2/2/0；OK 0/2/0；ZW 1/1/0 |
| D-dev / u_block | CH 1/0/0；HS 1/0/0；OK 0/5/0；QS 1/0/0 |
| D-dev / usb_adaptor | AK 1/0/1；HS 1/0/0；OK 1/4/0 |

已核验 102 个注册输入的 source SHA 无重复。D4B 的 18 个 D-fit 图像内部有 4 对已有 dHash 距离 ≤4（最小 3）；这四对留在 D-fit 内，不属于 W/X stream。W/X 的 D-dev 反馈与未来角色按 Real-IAD sample-key 分开。跨来源 perceptual hash 未重新计算，故不能声称全面排除近重复。24 个 Real-IAD sample-key 是操作性分组，不能等同于 24 个已证实独立物理工件。所有 D4-Lite、D4B 与 D5R 图像已曝光；均须排除于未来 D-conf。
