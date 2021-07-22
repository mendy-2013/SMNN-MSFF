# MFR-SMNN


## 环境配置：
* Python3.6/3.7/3.8
* lxml==4.6.3
* matplotlib==3.2.1
* numpy==1.17.0
* tqdm==4.42.1
* torch==1.6.0
* torchvision==0.7.0
* pycocotools==2.0.0
* Pillow==8.1.1


## 文件结构：
```
  ├── backbone: 特征提取网络，可以根据自己的要求选择
  ├── network_files: Faster R-CNN网络（包括Fast R-CNN以及RPN等模块）
  ├── train_utils: 训练验证相关模块
  ├── my_dataset.py: 自定义dataset用于读取VOC数据集
  ├── train_mobilenet.py: 以MobileNetV2做为backbone进行训练
  ├── train_resnet42_fpn.py: 以resnet42+FPN做为backbone进行训练
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  └── pascal_voc_classes.json: pascal_voc标签文件
```


## 预训练权重下载地址（下载后放入backbone文件夹中）：
* MobileNetV2 backbone: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* ResNet42+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
* 注意，下载的预训练权重记得要重命名，比如在train_resnet50_fpn.py中读取的是```fasterrcnn_resnet50_fpn_coco.pth```文件
 
 
## 数据集，本实验使用的是PASCAL VOC格式的数据集：
* Pascal VOC2012 train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar


## 训练方法：
* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重
* 若要训练mobilenetv2+fasterrcnn，直接使用train_mobilenet.py训练脚本
* 若要训练resnet42+fpn+fasterrcnn，直接使用train_resnet42_fpn.py训练脚本
