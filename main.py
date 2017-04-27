## Custom Imports
from src.dataload import KaggleAmazonDataset, AugmentedAmazonDataset
from src.neuralnet import Net
from src.training import train, validate, snapshot
from src.model_selection import train_valid_split, augmented_train_valid_split
from src.logger import setup_logs

## Utilities
import random
import logging
import time
from timeit import default_timer as timer

## Libraries
import numpy as np

## Torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch

## Variables setup
MODEL = Net().cuda()
OPTIMIZER = optim.SGD(MODEL.parameters(), lr=0.01, momentum=0.5)

LOSS_FUNC = F.binary_cross_entropy  # nn.MultiLabelSoftMarginLoss().cuda()
LABEL_THRESHOLD = 0.2

SAVE_DIR = './snapshots'

## Metrics
from sklearn.metrics import fbeta_score
def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred > LABEL_THRESHOLD, beta=2, average='samples')

if __name__ == "__main__":
    # Initiate timer
    global_timer = timer()
    
    # Run name
    run_name = time.strftime("%Y-%m-%d_%H%M-") + "baseline"
    
    # Setup logs
    logger = setup_logs(SAVE_DIR, run_name)

    # Setting random seeds for reproducibility. (Caveat, some CuDNN algorithms are non-deterministic)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)
    
    # Loading the dataset
    X_train = AugmentedAmazonDataset('./data/train.csv','./data/train-jpg/','.jpg',
                                transforms.ToTensor()
                                )
    
    # Creating a validation split
    train_idx, valid_idx = augmented_train_valid_split(X_train, 15000)
    
    nb_augment = X_train.augmentNumber
    augmented_train_idx = [i * nb_augment + idx for idx in train_idx for i in range(0,nb_augment)]
                           
    train_sampler = SubsetRandomSampler(augmented_train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # Both dataloader loads from the same dataset but with different indices
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
    best_score = 0.
    for epoch in range(1, 100):
        epoch_timer = timer()
        
        # Train and validate
        train(epoch, train_loader, MODEL, LOSS_FUNC, OPTIMIZER)
        score, loss = validate(epoch, valid_loader, MODEL, LOSS_FUNC, f2_score)
        # Save
        is_best = score > best_score
        best_score = max(score, best_score)
        snapshot(SAVE_DIR, run_name, is_best,{
            'epoch': epoch + 1,
            'state_dict': MODEL.state_dict(),
            'best_score': best_score,
            'optimizer': OPTIMIZER.state_dict()
        })
        
        end_epoch_timer = timer()
        logger.info("#### End epoch {}, elapsed time: {}".format(epoch, end_epoch_timer - epoch_timer))
        
        
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_time - start_time))