## Custom Imports
from src.dataload import KaggleAmazonDataset, AugmentedAmazonDataset
from src.neuralnet import Net, ResNet18, ResNet50, WideResNet
from src.training import train, snapshot
from src.validation import validate
from src.model_selection import train_valid_split, augmented_train_valid_split
from src.logger import setup_logs
from src.prediction import predict, output

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
MODEL = ResNet50(17).cuda() #Net().cuda()
# MODEL = WideResNet(16, 17, 4, 0.3).cuda()

# OPTIMIZER = optim.Adam(MODEL.parameters(), lr=0.1, weight_decay=1e-4) # From scratch
# OPTIMIZER = optim.SGD(MODEL.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)  # From scratch
OPTIMIZER = optim.SGD(MODEL.classifier.parameters(), lr=1e-2, momentum=0.9) # Finetuning

CRITERION = F.binary_cross_entropy

SAVE_DIR = './snapshots'

if __name__ == "__main__":
    # Initiate timer
    global_timer = timer()
    
    # Run name
    run_name = time.strftime("%Y-%m-%d_%H%M-") + "wideresnet-28-10"
    
    # Setup logs
    logger = setup_logs(SAVE_DIR, run_name)

    # Setting random seeds for reproducibility. (Caveat, some CuDNN algorithms are non-deterministic)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)
    
    # Loading the dataset
    
    ## Normalization for full training
    # ds_transform = transforms.Compose([
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(mean=[0.30249774, 0.34421161, 0.31507745], std=[0.13718569, 0.14363895, 0.16695958])
    #                 ])
    
    ## Normalization from ImageNet
    ds_transform = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                     ])
    
    ############################################################
    # Non-Augmented
    # X_train = KaggleAmazonDataset('./data/train.csv','./data/train-jpg/','.jpg',
    #                              ds_transform
    #                             )
    # Creating a validation split
    # train_idx, valid_idx = train_valid_split(X_train, 0.2)
    
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)
    ############################################################
    
    ############################################################
    # Augmented part
    X_train = AugmentedAmazonDataset('./data/train.csv','./data/train-jpg/','.jpg',
                                ds_transform
                                )
    
    # Creating a validation split
    train_idx, valid_idx = augmented_train_valid_split(X_train, 0.2)
    
    nb_augment = X_train.augmentNumber
    augmented_train_idx = [i * nb_augment + idx for idx in train_idx for i in range(0,nb_augment)]
                           
    train_sampler = SubsetRandomSampler(augmented_train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    ###########################################################
    
    # Both dataloader loads from the same dataset but with different indices
    train_loader = DataLoader(X_train,
                          batch_size=8,
                          sampler=train_sampler,
                          num_workers=4,
                          pin_memory=True)
    
    valid_loader = DataLoader(X_train,
                          batch_size=64,
                          sampler=valid_sampler,
                          num_workers=4,
                          pin_memory=True)
    
    ## Start training
    best_score = 0.
    for epoch in range(1, 15):
        epoch_timer = timer()
        
        # Train and validate
        train(epoch, train_loader, MODEL, CRITERION, OPTIMIZER)
        score, loss, threshold = validate(epoch, valid_loader, MODEL, CRITERION)
        # Save
        is_best = score > best_score
        best_score = max(score, best_score)
        snapshot(SAVE_DIR, run_name, is_best,{
            'epoch': epoch + 1,
            'state_dict': MODEL.state_dict(),
            'best_score': best_score,
            'optimizer': OPTIMIZER.state_dict(),
            'threshold': threshold,
            'val_loss': loss
        })
        
        end_epoch_timer = timer()
        logger.info("#### End epoch {}, elapsed time: {}".format(epoch, end_epoch_timer - epoch_timer))
        
        
    ## Prediction
    X_test = KaggleAmazonDataset('./data/sample_submission.csv','./data/test-jpg/','.jpg',
                                  ds_transform
                                 )
    test_loader = DataLoader(X_test,
                              batch_size=64,
                              num_workers=4,
                              pin_memory=True)
    
    predictions = predict(test_loader, MODEL) # TODO load model from the best on disk
    
    output(predictions,
           threshold,
           X_test,
           X_test.getLabelEncoder(),
           './out',
           run_name,
           score) # TODO early_stopping and use best_score
    
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_time - start_time))