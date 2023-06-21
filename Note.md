

# 2.1任务说明

定位和检测：

- 定位：找到检测图像中带有一个给定标签的**单个目标**
- 检测：找到图像中带有给定标签的**所有目标**

得出类别标签和置信度得分



# 2.2 数据集

- PASCAL VOC

PASCAL VOC 2012有11530张图像，27450个标注，20个类别。

 

- MS COCO

20万个图像，50万个标注，80个类别。



# 2.3 性能指标与检测精度

**检测精度**：

- Precision, Recall, F1 score
- IoU (Intersection over Union)
- P-R curve (Precision-Recall curse)
- AP (Average Precision)
- mAP (mean Average Precision)

**检测速度**：

- 前传还是
- FPS
- 浮点运算量（FLOPS）



## 混淆矩阵

![image-20230614213829249](./.assets/image-20230614213829249.png)

- 精度Precision（查准率）是评估预测的准不准（看预测列）
- 召回率Recall（查全率）是评估找的全不全（看预测行）



**直观理解**:

![](./.assets/Precisionrecall.svg)





## **IoU**

![image-20230614214121867](./.assets/image-20230614214121867.png)

IoU越大，意味着预测越准确。可以设置一个阈值，比如是0.5：

- 如果 IoU > 0.5, 就将其设置为Ture Positive（TP） 
- 如果 IoU < 0.5, 就将其设置为False Postitive（FP）

- 当图片中有一个ground truth，但是网络没有检测到的话，就把它标记为False Negative（FN）

- 任何一个没有检测为物体的部分，是True Negative（TN），但是在目标检测中没有作用，所以忽略它



## AP与mAP

AP衡量的是学习出来的模型在每个类别上的好坏。

mAP衡量出来的是模型在所有类别上的好坏，mAP就是所有类别上AP的均值。



![image-20230614214834897](./.assets/image-20230614214834897.png)



AP在不同数据集的规则：

- 对于PASCAL VOC，如果IoU大于0.5，就视为正样本（TP）。但是，如果检测到同一目标的多个检测，则视第一个目标为正样本（TP），其余目标位负样本（FP）

在COCO数据集中：

AP@.50 meas the AP with IoU=0.50.

AP@.75 meas the AP with IoU=0.75.



![image-20230614215400607](./.assets/image-20230614215400607.png)



![image-20230614215501565](./.assets/image-20230614215501565.png)



$$AP_S$$代表大物体的精度 

$$AP_M$$代表中等大小的物体的精度

$$AP_{S}$$代表小物体的精度 

![image-20230614215640933](./.assets/image-20230614215640933.png)



![image-20230614215904109](./.assets/image-20230614215904109.png)



![image-20230614215917173](./.assets/image-20230614215917173.png)



![image-20230614215953514](./.assets/image-20230614215953514.png)



![image-20230614220034516](./.assets/image-20230614220034516.png)



![image-20230614220044994](./.assets/image-20230614220044994.png)



# 3 Linux下使用YOLOv5训练

```shell
python detect.py  --source ./data/images/bus.jpg --weights weights/yolov5s.pt --conf 0.4
```





# 4.1 YOLO系列发展史

![image-20230618200927352](./.assets/image-20230618200927352.png)



![image-20230618201207647](./.assets/image-20230618201207647.png)



