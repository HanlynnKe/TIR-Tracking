# TIR-Tracking-PixelNet
*Graduation Design - Fine-grained Feature Based TIR Tracking*

**PixelNet embedded** SiamRPN++ tracker using ResNet-50 as backbone

主要使用框架：[PySOT](https://github.com/STVIR/pysot)中的SiamRPN++追踪器
课题中增加的代码有：
 - [探究RGB图像与热红外图像特性对比时使用的代码](https://github.com/HanlynnKe/TIR-Tracking/tree/master/paper-steps)
 - [backbone中嵌入PixelNet时使用的与pixelnet和resnet相关的代码](https://github.com/HanlynnKe/TIR-Tracking/tree/master/pysot/pysot/models/backbone)
 - 添加对热红外图像训练集TIR的支持（修改[config代码](https://github.com/HanlynnKe/TIR-Tracking/blob/master/pysot/pysot/core/config.py)）
 - 添加热红外图像训练集TIR及其[预处理代码](https://github.com/HanlynnKe/TIR-Tracking/tree/master/pysot/training_dataset/tir)
 - 添加对热红外图像测试集PTB-TIR的支持（修改[init代码](https://github.com/HanlynnKe/TIR-Tracking/blob/master/pysot/toolkit/datasets/__init__.py)，增加[数据集处理代码](https://github.com/HanlynnKe/TIR-Tracking/blob/master/pysot/toolkit/datasets/ptbtir.py)）
 - 添加热红外图像测试集PTB-TIR及其[预处理代码](https://github.com/HanlynnKe/TIR-Tracking/tree/master/pysot/testing_dataset/PTBTIR)
