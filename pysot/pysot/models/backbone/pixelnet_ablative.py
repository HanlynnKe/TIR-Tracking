from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models


# Attention Block模块由GC Block删改而来，有关GC Block的源码请参照官方代码仓库：
# https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = in_channels

        self.conv = nn.Conv2d(self.in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.transformer = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.in_channels, kernel_size=1)
        )
        nn.init.constant_(self.transformer[-1].weight, 0)
        nn.init.constant_(self.transformer[-1].bias, 0)

    def feature_extractor(self, x):
        batch, channel, height, width = x.size()
        
        route_1 = x
        # [N, C, H * W]
        route_1 = route_1.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        route_1 = route_1.unsqueeze(1)
        
        # [N, 1, H, W]
        route_2 = self.conv(x)
        # [N, 1, H * W]
        route_2 = route_2.view(batch, 1, height * width)
        # [N, 1, H * W]
        route_2 = self.softmax(route_2)
        # [N, 1, H * W, 1]
        route_2 = route_2.unsqueeze(-1)
        
        # [N, 1, C, 1]
        feature = torch.matmul(route_1, route_2)
        # [N, C, 1, 1]
        feature = feature.view(batch, channel, 1, 1)
        
        return feature

    def forward(self, x):
        # [N, C, 1, 1]
        feature = self.feature_extractor(x)
        fine_grained = self.transformer(feature)
        out = x + fine_grained
        
        return out


# 由预训练的ResNet50组成浅卷积网络，只使用三层卷积层
class ShallowResNet(nn.Module):
    def __init__(self):
        super(ShallowResNet, self).__init__()
        resnet50 = models.resnet50(pretrained=False)

        # 这一部分直接使用原框架定义即可
#         self.Layer1 = nn.Sequential(
#             resnet50.conv1,
#             resnet50.bn1,
#             resnet50.relu,
#             resnet50.maxpool,
#         )

        self.Layer2 = nn.Sequential(
            resnet50.layer1[0].conv1,
            resnet50.layer1[0].bn1,
        )
        self.Layer3 = nn.Sequential(
            resnet50.layer1[0].conv3,
            resnet50.layer1[0].bn3,
        )

#         for param in self.Layer1.parameters():
#             param.requires_grad = False
#         for param in self.Layer2.parameters():
#             param.requires_grad = False
#         for param in self.Layer3.parameters():
#             param.requires_grad = False
        
    def forward(self, x):
#         x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)

        return x

    
class PixelNet1B(nn.Module):
    # ResNet50 embedded
    def __init__(self):
        super(PixelNet1B, self).__init__()
        
        self.shallow = ShallowResNet()
        
        self.atblock = AttentionBlock(in_channels=256)
        
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
        
    def forward(self, t):
        t = self.shallow(t)
        
        t_1 = self.atblock(t)
        t_ = self.W(t_)
        t = t_ + t
        
        return t
    

class PixelNet2B(nn.Module):
    # ResNet50 embedded
    def __init__(self):
        super(PixelNet2B, self).__init__()
        
        self.shallow = ShallowResNet()
        
        self.atblock_1 = AttentionBlock(in_channels=256)
        
        self.atblock_2 = AttentionBlock(in_channels=256)
        
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
        
    def forward(self, t):
        t = self.shallow(t)
        
        t_1 = self.atblock_1(t)
        t_2 = self.atblock_2(t)
        t_ = torch.cat((t_1, t_2), 1)
        t_ = self.W(t_)
        t = t_ + t
        
        return t
    
    
class PixelNet3B(nn.Module):
    # ResNet50 embedded
    def __init__(self):
        super(PixelNet3B, self).__init__()
        
        self.shallow = ShallowResNet()
        
        self.atblock_1 = AttentionBlock(in_channels=256)
        
        self.atblock_2 = AttentionBlock(in_channels=256)
        
        self.atblock_3 = AttentionBlock(in_channels=256)
        
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
        
    def forward(self, t):
        t = self.shallow(t)
        
        t_1 = self.atblock_1(t)
        t_2 = self.atblock_2(t)
        t_3 = self.atblock_3(t)
        t_ = torch.cat((torch.cat((t_1, t_2), 1), t_3), 1)
        t_ = self.W(t_)
        t = t_ + t
        
        return t
    
    
class PixelNet4B(nn.Module):
    # ResNet50 embedded
    def __init__(self):
        super(PixelNet4B, self).__init__()
        
        self.shallow = ShallowResNet()
        
        self.atblock_1 = AttentionBlock(in_channels=256)
        
        self.atblock_2 = AttentionBlock(in_channels=256)
        
        self.atblock_3 = AttentionBlock(in_channels=256)
        
        self.atblock_4 = AttentionBlock(in_channels=256)
        
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
        
    def forward(self, t):
        t = self.shallow(t)
        
        t_1 = self.atblock_1(t)
        t_2 = self.atblock_2(t)
        t_3 = self.atblock_3(t)
        t_4 = self.atblock_4(t)
        t1_ = torch.cat((t_1, t_2), 1)
        t2_ = torch.cat((t_3, t_4), 1)
        t_ = torch.cat((t1_, t2_), 1)
        t_ = self.W(t_)
        t = t_ + t
        
        return t