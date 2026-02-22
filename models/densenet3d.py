import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# 假设 resnet3d.py 和 densenet3d.py 在同一目录下
# 如果报错找不到模块，可以将 CBAM3D 相关的类复制到这个文件中
from .resnet3d import CBAM3D

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, use_cbam=False):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate
        self.use_cbam = use_cbam

        # CBAM 模块处理的是新生成的特征 (通道数为 growth_rate)
        if self.use_cbam:
            self.cbam = CBAM3D(growth_rate)

    def forward(self, x):
        new_features = self.norm1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)

        new_features = self.norm2(new_features)
        new_features = self.relu2(new_features)
        new_features = self.conv2(new_features)

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        # --- CBAM 插入点 ---
        if self.use_cbam:
            new_features = self.cbam(new_features)
        # ------------------

        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, use_cbam=False):
        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
                use_cbam=use_cbam # 传递参数
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm3d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet3D(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=5, in_channels=2, use_cbam=False):
        super(DenseNet3D, self).__init__()

        # Initial Conv
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                use_cbam=use_cbam # 传递参数
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.norm5 = nn.BatchNorm3d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.norm5(features)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def densenet121_3d(num_classes=5, in_channels=2, use_cbam=False):
    return DenseNet3D(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                      num_classes=num_classes, in_channels=in_channels, use_cbam=use_cbam)