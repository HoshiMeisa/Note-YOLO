# YOLOv5中的配置文件参数

YOLOv5通过yaml配置文件来搭建先在common.py文件中构建相同名称的类，再从配置文件中读取到对应的操作后就会对该操作的类实例化。整个网络结构由Model类构成，Model中调用了parse_model()函数，由该函数对配置文件进行解析后调用对应的类进行网络构建，构建后由Model实现后面的处理。
parse_model()函数存在于yolo.py文件中。

```python
def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    add = d['backbone'] + d['head']

```

以`YOLOv5s.yaml`其中一行为例:

```python
[-1, 1, Conv, [64, 6, 2, 2]]
[from, number, module, args]
```

定义好以上参数后，parse_model()函数对yaml文件中的内容按行取出，依次放到f, n, m, args中。

- f：-1代表从上一层接受特征
- n：1代表只有一个这样的操作
- m：Conv这层要执行Conv操作
- args：[64, 6, 2, 2]代表这层输出为64维，用 $3 \times 3$ 的卷积核，步长为1
  

其中需要解析yolo.py文件中的 `m = eval(m) if isinstance(m, str) else m` .



<img src="./.assets/image-20230701095131770.png" alt="image-20230701095131770" style="zoom: 50%;" />

# SPP

一般来说CNN可以简略的看做卷积网络层加上全连接网络部分。
- 卷积核：完全能够适用任意大小的输入，并且能够产生任意大小的输出

- 全连接层：参数是神经元对于所有输入的连接权重。如果输入尺寸不固定的话，全连接层参数的个数都不能固定

CNN对于每一个区域候选都需要首先将图片放缩到固定尺度。

## 存在的问题 

速度瓶颈：重复为每个Region Proposal提取特征是极其耗费时间的。

性能瓶颈：对于所有的Region Proposal缩放到固定的尺寸会导致我们不期望看到的几何形变，而且

由于速度瓶颈的存在，不可能采用多尺度或者是大量的数据增强去训练模型。

## 常见解决方法

全连接层需要固定的输入，在全连接层前加入一个网络层，对任意的输入产生固定的输出；对于最后一层卷积层的输出进行Pooling，但这个Pooling窗口的尺寸及步长设置为相对值（输出尺寸的一个比例值）。

**后果：任意输入–固定的输出**

## 解决方案SPPNet

在以上想法上加入SPM的思路。

对于一副图像，分成若干尺度的块。比如将一副图像分成1份，4份，8份等，然后对于每一块提取特征然后融合在一起，以此兼容多个尺度的特征。

<img src="./.assets/image-20230701094122692.png" alt="image-20230701094122692" style="zoom:50%;" />

eg. 把特征图依次输出为4 x 4 x 256，2 x 2 x 256， 1 x 1 x 256最后按256通道拼接后就可以输入到FC中。

# YOLOv5中的SPP

YOLO中的SPP只是借鉴了SPP的思想。如图所示，YOLO采用统一的步长但不同尺寸的卷积核实现SPP，统一步长则代表输出的特征图尺寸一样，只是对区域的敏感性不一样，再通过concate按通道拼接后用1x1卷积，实现特征融合。

<img src="./.assets/image-20230701094242479.png" alt="image-20230701094242479" style="zoom:50%;" />

<img src="./.assets/image-20230701094315536.png" alt="image-20230701094315536" style="zoom:50%;" />

输入特征先通过一个Conv，然后分别进行不同Kernel的Pooling，3个Pooling和输入拼接，再通过一个Conv。

```python
class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)): ## ch_in, ch_out
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

```

**SPPF**

```python
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher 
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

```

