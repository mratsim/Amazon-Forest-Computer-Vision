from torch import nn
import torch.nn.functional as F
from torchvision import models
import src.net_wide_resnet as wrn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.dense1 = nn.Linear(246016, 256)
        self.dense1_bn = nn.BatchNorm1d(256)
        self.dense2 = nn.Linear(256, 64)
        self.dense2_bn = nn.BatchNorm1d(64)
        self.dense3 = nn.Linear(64, 17)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1_bn(self.dense1(x)))
        x = F.relu(self.dense2_bn(self.dense2(x)))
        x = self.dense3(x)
        return F.sigmoid(x)
    
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        # Everything except the last linear layer
        original_model = models.__dict__['resnet18'](pretrained=True)

        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(
        nn.Linear(512, num_classes)
        )

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False


    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return F.sigmoid(y)

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        # Everything except the last linear layer
        original_model = models.__dict__['resnet50'](pretrained=True)

        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(
        nn.Linear(2048, num_classes)
        )

        # # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False


    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return F.sigmoid(y)
    
class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        original_model = wrn.WideResNet(depth, num_classes, widen_factor, dropRate)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(
        nn.Linear(1048576, num_classes)
        )
        
    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return F.sigmoid(y)