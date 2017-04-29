## Custom Imports
from src.dataload import KaggleAmazonDataset
from src.neuro import Net, ResNet18, ResNet50
from src.training import train, snapshot
from src.validation import validate
from src.model_selection import train_valid_split, augmented_train_valid_split
from src.logger import setup_logs
from src.prediction import predict, output
from src.data_augmentation import ColorJitter
# from src.zoo.wideresnet import WideResNet

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


############################################################################
#######  CONTROL CENTER ############# STAR COMMAND #########################
## Variables setup
model = ResNet50(17).cuda()
# model = Net().cuda()
# model = WideResNet(16, 17, 4, 0.3)

epochs = 7
batch_size = 16

# Run name
run_name = time.strftime("%Y-%m-%d_%H%M-") + "resnet50-mcc-thresholding"

## Normalization on dataset mean/std
# normalize = transforms.Normalize(mean=[0.30249774, 0.34421161, 0.31507745],
#                                  std=[0.13718569, 0.14363895, 0.16695958])
    
## Normalization on ImageNet mean/std for finetuning
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


# optimizer = optim.Adam(model.parameters(), lr=0.1) # From scratch # Don't use Weight Decay with PReLU
# optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)  # From scratch
optimizer = optim.SGD(model.classifier.parameters(), lr=1e-2, momentum=0.9) # Finetuning

criterion = torch.nn.MultiLabelSoftMarginLoss()

save_dir = './snapshots'

#######  CONTROL CENTER ############# STAR COMMAND #########################
############################################################################

if __name__ == "__main__":
    # Initiate timer
    global_timer = timer()
    
    # Setup logs
    logger = setup_logs(save_dir, run_name)

    # Setting random seeds for reproducibility. (Caveat, some CuDNN algorithms are non-deterministic)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)
    
    ##############################################################
    ## Loading the dataset
    
    ## Augmentation + Normalization for full training
    ds_transform_augmented = transforms.Compose([
                     transforms.RandomSizedCrop(224),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     ColorJitter(),
                     normalize
                     ])
    
    ## Normalization only for validation and test
    ds_transform_raw = transforms.Compose([
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     normalize
                     ])
    
    ####     #########     ########     ###########     #####
    
    X_train = KaggleAmazonDataset('./data/train.csv','./data/train-jpg/','.jpg',
                                 ds_transform_augmented
                                 )
    X_val = KaggleAmazonDataset('./data/train.csv','./data/train-jpg/','.jpg',
                                 ds_transform_raw
                                 )
    # Creating a validation split
    train_idx, valid_idx = train_valid_split(X_train, 0.2)
    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    ######    ##########    ##########    ########    #########
    
    # Both dataloader loads from the same dataset but with different indices
    train_loader = DataLoader(X_train,
                          batch_size=batch_size,
                          sampler=train_sampler,
                          num_workers=4,
                          pin_memory=True)
    
    valid_loader = DataLoader(X_val,
                          batch_size=batch_size,
                          sampler=valid_sampler,
                          num_workers=4,
                          pin_memory=True)
    
    ###########################################################
    ## Start training
    best_score = 0.
    for epoch in range(epochs):
        epoch_timer = timer()
        
        # Train and validate
        train(epoch, train_loader, model, criterion, optimizer)
        score, loss, threshold = validate(epoch, valid_loader, model, criterion)
        # Save
        is_best = score > best_score
        best_score = max(score, best_score)
        snapshot(save_dir, run_name, is_best,{
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
            'threshold': threshold,
            'val_loss': loss
        })
        
        end_epoch_timer = timer()
        logger.info("#### End epoch {}, elapsed time: {}".format(epoch, end_epoch_timer - epoch_timer))
        
    ###########################################################
    ## Prediction
    X_test = KaggleAmazonDataset('./data/sample_submission.csv','./data/test-jpg/','.jpg',
                                  ds_transform_raw
                                 )
    test_loader = DataLoader(X_test,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True)
    
    predictions = predict(test_loader, model) # TODO load model from the best on disk
    
    output(predictions,
           threshold,
           X_test,
           X_train.getLabelEncoder(),
           './out',
           run_name,
           score) # TODO early_stopping and use best_score
    
    ##########################################################
    
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))