# 可切换的空洞卷积

## SAC（Switchable Atrous Convolution）

<img src="./.assets/image-20230727171016003.png" alt="image-20230727171016003" style="zoom: 50%;" />

如上图所示是SAC的整体结构，作者将ResNet中所有的 $3 \times 3$ 的卷积层都转换为SAC，以实现在不同空洞率的情况下卷积操作的软切换。图中红色的锁表示权重是相同的，只是训练过程可能有差异。两个一前一后的全局上下文模块用于将图像级别的信息引入特征中。

用 $y = Conv(x, w, r)$ 来表示卷积操作，其中 $w$ 表示权重，$r$ 表示空洞率，$x$ 是输入，$y$ 是输出。那么可以根据下式将卷积层转换为SAC
$$
\textbf{Conv}(x, w, 1)\xrightarrow[\text{to  SAC}]{\text{Convert}} \textbf{S}(x) \cdot \textbf{Conv}(x, w, 1) + (1 - \textbf{S}(x)) \cdot \textbf{Conv}(x, w + \Delta w, r)
$$
其中 $\Delta w$ 是一个可学习的权重，转换函数 $\textbf{S}()$ 由 $5 \times 5$ 的平均池化层和 $1 \times 1$ 的卷积层组成，其与输入和位置相关。

作者提出一种锁机制，也就是将一个权重设为 $w$，将另一个权重设为 $ w + \Delta w$，这样做的原因为：目标检测通常使用预训练好的模型来对权重进行初始化，但是对于一个由标准卷积层转换而来的SAC来说，较大空洞率的权重通常是缺失的。由于可以在相同的权重下以不同的空洞率粗略地检测出不同大小的目标，因此还是可以用预训练模型的权重来初始化缺失的权重的。

本文使用 $ w + \Delta w$ 作为缺失的权重，其中 $w$ 来自于预训练模型， $\Delta w$ 被初始化为0。当固定 $\Delta w = 0$ 时，AP会降低约0.1%。但是没有锁机制的话， $\Delta w$ 会使AP降低很多。

