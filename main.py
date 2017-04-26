from src.dataload import KaggleAmazonDataset
from src.neuralnet import Net
from src.training import train, validate
from src.model_selection import train_valid_split
# from src.model_selection import train_valid_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import fbeta_score
import numpy as np
import torch
import random

MODEL = Net().cuda()
OPTIMIZER = optim.SGD(MODEL.parameters(), lr=0.01, momentum=0.5)

# LOSS_FUNC = nn.MultiLabelSoftMarginLoss().cuda()
LOSS_FUNC = F.binary_cross_entropy
LABEL_THRESHOLD = 0.2

def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred > LABEL_THRESHOLD, beta=2, average='samples')

if __name__ == "__main__":
    # Setting random seeds for reproducibility. (Caveat, some CuDNN algorithms are non-deterministic)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)
    
    # Loading the dataset
    X_train = KaggleAmazonDataset('./data/train.csv','./data/train-jpg/','.jpg',
                                transforms.ToTensor()
                                )
    
    # Creating a validation split
    train_idx, valid_idx = train_valid_split(X_train, 15000)
    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # Both dataloader loads from teh same dataset but with different indices
    train_loader = DataLoader(X_train,
                          batch_size=64,
                          sampler=train_sampler,
                          num_workers=1, # 1 when CUDA
                          pin_memory=True)
    valid_loader = DataLoader(X_train,
                          batch_size=64,
                          sampler=valid_sampler,
                          num_workers=1, # 1 when CUDA
                          pin_memory=True)
    
    # Start training
    for epoch in range(1, 6):
        train(epoch, train_loader, MODEL, LOSS_FUNC, OPTIMIZER)
        validate(epoch, valid_loader, MODEL, LOSS_FUNC, f2_score)