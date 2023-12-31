# Normalized Gaussian Wasserstein Distance

# 1 摘要

小目标检测是一个非常具有挑战性的问题，因为小目标只包含几个像素大小。作者证明，由于缺乏外观信息，最先进的检测器也不能在小目标上得到令人满意的结果。作者提出，基于IoU的指标，如IoU本身及其扩展，对小目标的位置偏差非常敏感，在Anchor-Based的检测器中使用时，严重降低了检测性能。

为了解决这一问题，论文提出了一种新的基于Wasserstein距离的小目标检测评估方法。具体来说，首先将BBox建模为二维高斯分布，然后提出一种新的度量标准，称为Normalized Wasserstein Distance(NWD)，通过它们对应的高斯分布计算它们之间的相似性。

NWD度量可以很容易地嵌入到任何Anchor-Based的检测器的Assignment、非最大抑制和损失函数中，以取代常用的IoU度量。作者在一个用于小目标检测(AI-TOD)的新数据集上评估了度量，其中平均目标大小比现有的物体检测数据集小得多。大量的实验表明，当使用NWD度量时，本文方法的性能比标准fine-tuning baseline高出6.7 AP，比SOTA模型高出6.0 AP。

# 2 问题解析

## 2.1 IoU

![image-20230808091151615](./.assets/image-20230808091151615.png)

如上图，IoU对不同尺度的物体的**敏感性差异很大**。具体来说，对于 6×6 像素的小目标，轻微的位置偏差会导致明显的loU下降(从0.53下降到 0.06 )，导致标签分配不准确。然而，对于 36×36 像素的正常目标，loU略有变化(从0.90到 0.65 )，位置偏差相同。 此外，下图给出了4条不同目标尺度的loU-Deviation曲线，随着目标尺度的减小，曲线下降速度更快。 值得注意的是，loU的敏感性来自于BBox位置只能离散变化的特殊性。

<img src="./.assets/image-20230808091410259.png" alt="image-20230808091410259" style="zoom: 50%;" />

这种现象意味着IoU度量对离散位置偏差的目标尺度是变化的，最终导致标签分配存在以下2个缺陷（其中，IoU阈值 $(\theta_{p}, \ \theta_n)$ 用于Anchor-Based检测器中Pos/Neg训练样本的分配，(0.7,0.3) 用于Region Proposal Network (RPN)）：

- 首先，由于IoU对小目标的敏感性，使得微小的位置偏差翻转Anchor标记，导致Pos/Neg样本特征相似，网络收敛困难；
- 其次，利用IoU度量，作者发现AI-TOD数据集中分配给每个Ground-Truth (GT)的平均正样本数小于1，因为GT与任何Anchor之间的IoU低于最小正阈值。

因此，训练小目标检测的监督信息不足。尽管ATSS等动态分配策略可以根据物体的统计特性自适应地获得分配Pos/Neg标签的IoU阈值，但IoU的敏感性使得小目标检测难以找到一个良好的阈值并提供高质量的Pos/Neg样本。

鉴于IoU不是一个很好的度量小目标的度量标准，论文提出了一种新的度量标准，用Wasserstein距离来度量BBox的相似性来代替标准IoU。具体来说：

- 首先，将包围盒建模为二维高斯分布；
- 然后，使用提出的Normalized Wasserstein Distance (NWD)来度量导出的高斯分布的相似性。

## 2.2 Wasserstein distance的主要优点

Wasserstein distance的主要优点如下：

1. 无论小目标之间有没有重叠都可以度量分布相似性;
2. NWD**对不同尺度的目标不敏感**，更适合测量**小目标**之间的相似性。

NWD可应用于One-Stage和Multi-Stage Anchor-Based检测器。此外，NWD不仅可以替代标签分配中的IoU，还可以替代非最大抑制中的IoU(NMS)和回归损失函数。在一个新的TOD数据集AI-TOD上的大量实验表明，本文提出的NWD可以持续地提高所有检测器的检测性能。

**本论文的贡献总结如下**：

1. 分析了IoU对小目标定位偏差的敏感性，提出了NWD作为衡量2个BBox之间相似性的更好的度量；
2. 将NWD应用于Anchor-Based检测器的标签分配、NMS和损失函数，并设计了一个小目标检测器；
3. 提出的NWD可以显著提高目前流行的Anchor-Based检测器的TOD性能，在AI-TOD数据集上Faster R-CNN实现了从11.1%到17.6%的性能提升。
