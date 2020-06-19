# TIR-Tracking-PixelNet
*Graduation Design - Fine-grained Feature Based TIR Tracking*

**PixelNet embedded** SiamRPN++ tracker using ResNet-50 as backbone

主要使用框架：[PySOT](https://github.com/STVIR/pysot)中的SiamRPN++追踪器

课题中增加的代码有：
 - 探究RGB图像与热红外图像特性对比时使用的[代码](https://github.com/HanlynnKe/TIR-Tracking/tree/master/paper-steps)
 - 骨干网络中嵌入PixelNet时使用的[与pixelnet和resnet相关的代码](https://github.com/HanlynnKe/TIR-Tracking/tree/master/pysot/pysot/models/backbone)
 - 添加对热红外图像训练集TIR的支持（修改[config代码](https://github.com/HanlynnKe/TIR-Tracking/blob/master/pysot/pysot/core/config.py)）
 - 添加热红外图像训练集TIR及其[预处理代码](https://github.com/HanlynnKe/TIR-Tracking/tree/master/pysot/training_dataset/tir)
 - 添加对热红外图像测试集PTB-TIR的支持（修改[init代码](https://github.com/HanlynnKe/TIR-Tracking/blob/master/pysot/toolkit/datasets/__init__.py)，增加[预处理代码](https://github.com/HanlynnKe/TIR-Tracking/blob/master/pysot/toolkit/datasets/ptbtir.py)）
 - 添加热红外图像测试集PTB-TIR及其[预处理代码](https://github.com/HanlynnKe/TIR-Tracking/tree/master/pysot/testing_dataset/PTBTIR)

在离校前得到的结果：

 - SiamRPN++追踪器在RGB图像与TIR图像数据集中的表现
 
   Dataset | Accuracy | Robustness | Loss |  EAO
   --------|----------|------------|------|-------
   VOT2019 |  0.594   |    0.467   |  93  | 0.287
   PTB-TIR |  0.404   |    0.194   |  51  | 0.308
   
 - PixelNet两种嵌入方式的比较
 
      Model   | Accuracy | Robustness | Loss |  EAO  |  FPS
   -----------|----------|------------|------|-------|-------
   PixelNet-a |  0.413   |    0.362   |  95  | 0.247 |  53.6
   PixelNet-b |  0.408   |    0.331   |  87  | 0.260 |  62.4
   
 - PixelNet（PixelNet-b）与原模型的比较
 
     Model  | Accuracy | Robustness | Loss |  EAO  |  FPS
   ---------|----------|------------|------|-------|-------
   PixelNet |  0.408   |    0.331   |  87  | 0.260 |  62.4
   Original |  0.428   |    0.350   |  92  | 0.264 |  69.3
   
 - PixelNet中AttentionBlock的个数的比较
   - 10个epoch时
   
        AttnBlk No. | Accuracy | Robustness | Loss |  EAO  |  FPS
        ------------|----------|------------|------|-------|-------
        PixelNet-1b |  0.378   |    0.312   |  82  | 0.245 |  72.5
        PixelNet-2b |  0.363   |    0.354   |  93  | 0.225 |  70.3
        PixelNet-3b |  0.369   |    0.312   |  82  | 0.240 |  68.4
        PixelNet-4b |  0.357   |    0.343   |  90  | 0.220 |  66.4
     
   - 19个epoch时
   
        AttnBlk No. | Accuracy | Robustness | Loss |  EAO  |  FPS
        ------------|----------|------------|------|-------|-------
        PixelNet-2b |  0.408   |    0.331   |  87  | 0.260 |  62.4
        PixelNet-3b |  0.418   |    0.339   |  89  | 0.259 |  72.1
        PixelNet-4b |  0.399   |    0.350   |  92  | 0.240 |  64.4
 
