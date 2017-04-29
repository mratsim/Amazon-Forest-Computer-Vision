from torch import nn, ones
from torch.autograd import Variable
from torchvision import models
from torch.nn.init import kaiming_normal
from torch import np


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

    
## ResNets fine-tuning
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
        return y

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        # Everything except the last linear layer
        original_model = models.__dict__['resnet50'](pretrained=True)

        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(
        nn.Linear(2048, num_classes)
        )

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y