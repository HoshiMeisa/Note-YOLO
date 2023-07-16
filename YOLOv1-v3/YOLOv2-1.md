# 所做的改进

<img src="../.assets/image-20230622104759118.png" alt="image-20230622104759118" style="zoom:50%;" />



<img src="../.assets/image-20230622104851031.png" alt="image-20230622104851031" style="zoom:50%;" />



<img src="../.assets/image-20230622105315058.png" alt="image-20230622105315058" style="zoom:50%;" />



<img src="../.assets/image-20230622105600203.png" alt="image-20230622105600203" style="zoom:50%;" />



YOLOv1有两种先验框，但是物体的形状远不止两种。

YOLOv2选取了三个scale，产生了更多先验框。

YOLOv2对COCO数据集中的检测框进行聚类，使用K-Means算法对相似形状的框进行聚类。找到更合适的比列。按照这种方法提取出的比例都是实际值，更具有可信度。

在YOLOv2中是使用的k=5进行聚类。

在聚类中使用的距离并不是欧式距离，而是根据IoU计算得出的距离。此时距离不会与物体的大小有关。

<img src="../.assets/image-20230622110629849.png" alt="image-20230622110629849" style="zoom:50%;" />

<img src="../.assets/image-20230622141829260.png" alt="image-20230622141829260" style="zoom:50%;" />



<img src="../.assets/image-20230622142936378.png" alt="image-20230622142936378" style="zoom:50%;" />



<img src="../.assets/image-20230623092504708.png" alt="image-20230623092504708" style="zoom:50%;" />



$\sigma$ 即Sigmoid函数

<img src="../.assets/image-20230623093707582.png" alt="image-20230623093707582" style="zoom:50%;" />



## 感受野

<img src="../.assets/image-20230622145441775.png" alt="image-20230622145441775" style="zoom:50%;" />



<img src="../.assets/image-20230622145851070.png" alt="image-20230622145851070" style="zoom:50%;" />



<img src="../.assets/image-20230622150050988.png" alt="image-20230622150050988" style="zoom:50%;" />



<img src="../.assets/image-20230622150932121.png" alt="image-20230622150932121" style="zoom:50%;" />



## 多尺度检测

<img src="../.assets/image-20230623100215008.png" alt="image-20230623100215008" style="zoom:50%;" />









