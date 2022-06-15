import math

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

import torchvision

from collections import OrderedDict


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        #-----------------------------------#
        #   假设输入进来的图片是600,600,3
        #-----------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #-------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        #-------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 这里需要修改
def resnet50(pretrained = False):
    # 这里是初始化一个ResNet的网络模型
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    # params = model.state_dict()  # 获得模型的原始状态以及参数。
    # for k, v in params.items():
    #     print(k)  # 只打印key值，不打印具体参数。

    print("-----------------------------------")
    # backbone在body中是有的
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    if pretrained:
        # state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", model_dir="./model_data")
        # for k, v in state_dict.items():
        #     print(k)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # 只选取backbone部分的关键字
            if k[9:13] == 'body':
                name = k[14:]  # remove `backbone.body.`，从第14个key值字符取到最后一个字符
                new_state_dict[name] = v
        # fc 随机初始化
        fc_weight = nn.init.normal_(model.fc.weight)
        fc_bias = nn.init.normal_(model.fc.bias)
        new_state_dict['fc.weight'] = fc_weight
        new_state_dict['fc.bias'] = fc_bias
        model.load_state_dict(new_state_dict)
    #----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    #----------------------------------------------------------------------------#
    features    = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # features = list([model.backbone.body.conv1, model.backbone.body.bn1, model.backbone.body.relu, model.backbone.body.maxpool, model.backbone.body.layer1, model.backbone.body.layer2, model.backbone.body.layer3])
    #----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    #----------------------------------------------------------------------------#
    classifier  = list([model.layer4, model.avgpool])
    # classifier = list([model.backbone.body.layer4, model.backbone.body.maxpool])

    features    = nn.Sequential(*features)
    classifier  = nn.Sequential(*classifier)
    return features, classifier
