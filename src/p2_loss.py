import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
import torch
from torch.autograd import Variable

# If needed to code the categorical cross entropy from scratch: https://github.com/twitter/torch-autograd/blob/master/src/loss/init.lua

class ConvolutedLoss(_WeightedLoss):
    """ Treat the weather as MultiClassification (only one label possible)
        Treat the rest as Multilabel
        ==> Multi-Task learning
    """
    def __init__(self, weight=None, size_average=True):
        super(ConvolutedLoss, self).__init__(size_average)
        if weight is None:
            self.register_buffer('weight_weather', None)
            self.register_buffer('weight_other', None)
        else:
            self.register_buffer('weight_weather', weight[:4]) # Weather conditions are the first 4
            self.register_buffer('weight_other', weight[4:])
    
    def forward(self, input, target):
        # Cross-Entropy wants categorical not one-hot
        # Reverse one hot
        weather_targets = Variable(torch.arange(0,4).expand(target.size(0),4).masked_select(target[:,:4].data.byte().cpu()).long().cuda(), requires_grad = False)
        
        loss_weather = F.cross_entropy(input[:,:4],
                                       weather_targets,
                                       self.weight_weather,
                                       self.size_average)
        loss_other = F.binary_cross_entropy(F.sigmoid(input[:,4:]),
                                            target[:,4:],
                                            self.weight_other,
                                            self.size_average)
        
        return (loss_weather * 4/17) + (loss_other * 13/17)
        