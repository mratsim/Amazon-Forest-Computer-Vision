from torch import nn, ones
from torchvision import models
from torch.nn.init import kaiming_normal
from torch import np
import torch
import torch.nn.functional as F


## Custom baseline
class Net(nn.Module):    
    def __init__(self, input_size=(3,224,224), nb_classes=17):
        
        super(Net, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3,3))
        )
        
        ## Compute linear layer size
        self.flat_feats = self._get_flat_feats(input_size, self.features)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_feats, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(64, nb_classes)
        )
     
        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d or BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.apply(_weights_init)       
    
    def _get_flat_feats(self, in_size, feats):
        f = feats(Variable(ones(1,*in_size)))
        return int(np.prod(f.size()[1:]))
    

            
    def forward(self, x):
        feats = self.features(x)
        flat_feats = feats.view(-1, self.flat_feats)
        out = self.classifier(flat_feats)
        return out

    
## ResNet fine-tuning
class ResNet50(nn.Module):
    ## We use ResNet weights from PyCaffe.
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        
        # Loading ResNet arch from PyTorch and weights from Pycaffe
        original_model = models.resnet50(pretrained=False)
        original_model.load_state_dict(torch.load('./zoo/resnet50.pth'))
        
        # Everything except the last linear layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
        # Get number of features of last layer
        num_feats = original_model.fc.in_features
        
        # Plug our classifier
        self.classifier = nn.Sequential(
        nn.Linear(num_feats, num_classes)
        )
        
        # Init of last layer
        for m in self.classifier:
            kaiming_normal(m.weight)

        # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

class ResNet101(nn.Module):
    ## We use ResNet weights from PyCaffe.
    def __init__(self, num_classes):
        super(ResNet101, self).__init__()
        
        # Loading ResNet arch from PyTorch and weights from Pycaffe
        original_model = models.resnet101(pretrained=False)
        original_model.load_state_dict(torch.load('./zoo/resnet101.pth'))
        
        # Everything except the last linear layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
        # Get number of features of last layer
        num_feats = original_model.fc.in_features
        
        # Plug our classifier
        self.classifier = nn.Sequential(
        nn.Linear(num_feats, num_classes)
        )
        
        # Init of last layer
        for m in self.classifier:
            kaiming_normal(m.weight)

        # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

class ResNet152(nn.Module):
    ## We use ResNet weights from PyCaffe.
    def __init__(self, num_classes):
        super(ResNet152, self).__init__()
        
        # Loading ResNet arch from PyTorch and weights from Pycaffe
        original_model = models.resnet152(pretrained=False)
        original_model.load_state_dict(torch.load('./zoo/resnet152.pth'))
        
        # Everything except the last linear layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
        # Get number of features of last layer
        num_feats = original_model.fc.in_features
        
        # Plug our classifier
        self.classifier = nn.Sequential(
        nn.Linear(num_feats, num_classes)
        )
        
        # Init of last layer
        for m in self.classifier:
            kaiming_normal(m.weight)

        # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y
    
## VGG fine-tuning
class VGG16(nn.Module):
        def __init__(self, nb_classes=17):
            super(VGG16, self).__init__()
            original_model = models.vgg16(pretrained=False)
            self.features = original_model.features
            self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(25088, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, num_classes),
                )

            # Freeze Convolutional weights
            for p in self.features.parameters():
                p.requires_grad = False

        def forward(self, x):
            f = self.features(x)
            f = f.view(f.size(0), -1)
            y = self.classifier(f)
            return y

class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        
        original_model = models.densenet121(pretrained=True)
        
        # Everything except the last linear layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
        # Get number of features of last layer
        num_feats = original_model.classifier.in_features
        
        # Plug our classifier
        self.classifier = nn.Sequential(
        nn.Linear(num_feats, num_classes)
        )

        # Init of last layer
        for m in self.classifier:
            kaiming_normal(m.weight)
            
        # Freeze weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        out = F.relu(f, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(f.size(0), -1)
        out = self.classifier(out)
        return out