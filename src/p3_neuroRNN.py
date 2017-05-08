from torch import nn, ones
from torch.autograd import Variable
from torchvision import models
from torch.nn.init import kaiming_normal
from torch import np
import torch
import torch.nn.functional as F


class GRU_ResNet50(nn.Module):
    ## We use ResNet weights from PyCaffe.
    def __init__(self, num_classes, hidden_size, num_layers):
        super(GRU_ResNet50, self).__init__()
        
        # Loading ResNet arch from PyTorch and weights from Pycaffe
        original_model = models.resnet50(pretrained=False)
        original_model.load_state_dict(torch.load('./zoo/resnet50.pth'))
        
        # Everything except the last linear layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
        # Get number of features of last layer
        num_feats = original_model.fc.in_features
        
        self.bn = nn.BatchNorm1d(num_feats, momentum=0.01)
        
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=num_feats,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first = True)
        
        # Plug our classifier
        self.classifier = nn.Sequential(
        nn.Linear(hidden_size, num_classes)
        )
        
        # Init of last layer
        for m in self.classifier:
            kaiming_normal(m.weight)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
        # How to init RNN?

        # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

    def forward(self, x, hidden=None):
        f = self.features(x)
        f = self.bn(f.view(f.size(0), -1))
        f = f.unsqueeze(1)
        x, hidden = self.rnn(f, hidden)
        x = x.view(-1, self.hidden_size)
        y = self.classifier(x)
        return y
    
class LSTM_ResNet50(nn.Module):
    ## We use ResNet weights from PyCaffe.
    def __init__(self, num_classes, hidden_size, num_layers):
        super(LSTM_ResNet50, self).__init__()
        
        # Loading ResNet arch from PyTorch and weights from Pycaffe
        original_model = models.resnet50(pretrained=False)
        original_model.load_state_dict(torch.load('./zoo/resnet50.pth'))
        
        # Everything except the last linear layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
        # Get number of features of last layer
        num_feats = original_model.fc.in_features
        
        self.bn = nn.BatchNorm1d(num_feats, momentum=0.01)
        
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=num_feats,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first = True)
        
        # Plug our classifier
        self.classifier = nn.Sequential(
        nn.Linear(hidden_size, num_classes)
        )
        
        # Init of last layer
        for m in self.classifier:
            kaiming_normal(m.weight)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
                
        # How to init RNN?

        # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

    def forward(self, x, hidden=None):
        f = self.features(x)
        f = self.bn(f.view(f.size(0), -1))
        f = f.unsqueeze(1)
        x, hidden = self.rnn(f, hidden)
        x = x.view(-1, self.hidden_size)
        y = self.classifier(x)
        return y

    
class Skip_LSTM_RN50(nn.Module):
    ## We use ResNet weights from PyCaffe.
    def __init__(self, num_classes, hidden_size, num_layers):
        super(Skip_LSTM_RN50, self).__init__()
        
        # Loading ResNet arch from PyTorch and weights from Pycaffe
        original_model = models.resnet50(pretrained=False)
        original_model.load_state_dict(torch.load('./zoo/resnet50.pth'))
        
        # Everything except the last linear layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
        # Get number of features of last layer
        num_feats = original_model.fc.in_features
        
        self.bn = nn.BatchNorm1d(num_feats, momentum=0.01)
        
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=num_feats,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first = True)
        
        # Plug our classifier
        self.classifier = nn.Sequential(
        nn.Linear(hidden_size + num_feats, num_classes)
        )
        
        # Init of last layer
        for m in self.classifier:
            kaiming_normal(m.weight)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
                
        # How to init RNN?

        # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

    def forward(self, x, hidden=None):
        f = self.features(x)
        f = self.bn(f.view(f.size(0), -1))
        x, hidden = self.rnn(f.unsqueeze(1), hidden)
        x = x.view(-1, self.hidden_size)
        c = torch.cat((x,f),1) # Skip connection to avoid the LSTM eating the whole gradients
        y = self.classifier(c)
        return y