# 深度学习经典检测方法

- One-Stage（单阶段）：YOLO系列

一个CNN做回归任务，直接给出预测结果

![image-20230621125544895](./.assets/image-20230621125544895.png)



- Two-Stage（两阶段）：Faster-RCNN、Mask-RCNN系列

加入一个RPN（区域建议网络），多了预选框（Proposal）

![image-20230621125941008](./.assets/image-20230621125941008.png)



# 优缺点

单阶段：

速度快，适合做实时目标检测

但是效果不太好



两阶段：

速度慢，Mask-RCNN论文给出的速度是5FPS

![image-20230621130958048](./.assets/image-20230621130958048.png)



指标分析：

![image-20230621131048208](./.assets/image-20230621131048208.png)



![image-20230621131601689](./.assets/image-20230621131601689.png)



