## Custom Imports
from src.p2_dataload import KaggleAmazonDataset
from src.p_neuro import Net, ResNet50, ResNet101, ResNet152, DenseNet121
from src.p3_neuroRNN import GRU_ResNet50, LSTM_ResNet50, Skip_LSTM_RN50
from src.p_training import train, snapshot
#from src.p2_validation import validate
from src.p_validation import validate
from src.p_model_selection import train_valid_split
from src.p_logger import setup_logs
#from src.p2_prediction import predict, output
from src.p_prediction import predict, output
from src.p_data_augmentation import ColorJitter, PowerPIL
# from src.p_metrics import SmoothF2Loss
from src.p2_loss import ConvolutedLoss
from src.p_sampler import SubsetSampler, balance_weights

## Utilities
import random
import logging
import time
from timeit import default_timer as timer
import os

## Libraries
import numpy as np
import math

## Torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchsample.transforms import Affine
from torch.utils.data.sampler import WeightedRandomSampler, SubsetRandomSampler

############################################################################
#######  CONTROL CENTER ############# STAR COMMAND #########################
## Variables setup
model = ResNet50(17).cuda()
# model = ResNet152(17).cuda()

# model = GRU_ResNet50(17, 128, 2).cuda()
# model = LSTM_ResNet50(17, 128, 2).cuda()
# model = Skip_LSTM_RN50(17, 128, 2).cuda()

epochs = 16
batch_size = 64

# Run name
run_name = time.strftime("%Y-%m-%d_%H%M-") + "resnet50-L2reg-new-data"
## Normalization on ImageNet mean/std for finetuning
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

# Note, p_training has lr_decay automated
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0005) # Finetuning whole model

# criterion = ConvolutedLoss()
criterion = torch.nn.MultiLabelSoftMarginLoss(
    weight = torch.Tensor([1,  4,  2,  1,
                             1,  3,  3,  3,
                             4,  4,  1,  2,
                             1,  1,  3,  4,  1])
    ).cuda()

#classes = [
#    'clear', 'cloudy', 'haze','partly_cloudy',
#    'agriculture','artisinal_mine','bare_ground','blooming',
#    'blow_down','conventional_mine','cultivation','habitation',
#    'primary','road','selective_logging','slash_burn','water'
#    ]
## Frequency
#    [28203,  2330,  2695,  7251,
#     12338,   339,   859,   332,
#        98,   100,  4477,  3662,
#     37840,  8076,   340,   209,  7262]

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
                     PowerPIL(),
                     transforms.ToTensor(),
                     # ColorJitter(), # Use PowerPIL instead, with PillowSIMD it's much more efficient
                     normalize,
                     # Affine(
                     #    rotation_range = 15,
                     #    translation_range = (0.2,0.2),
                     #    shear_range = math.pi/6,
                     #    zoom_range=(0.7,1.4)
                     #)
    ])
    
    ## Normalization only for validation and test
    ds_transform_raw = transforms.Compose([
                     transforms.Scale(224),
                     transforms.ToTensor(),
                     normalize
                     ])
    
    ####     #########     ########     ###########     #####
    
    X_train = KaggleAmazonDataset('./data/train_v2.csv','./data/train-jpg/','.jpg',
                                 ds_transform_augmented
                                 )
    X_val = KaggleAmazonDataset('./data/train_v2.csv','./data/train-jpg/','.jpg',
                                 ds_transform_raw
                                 )
    
    # Resample the dataset
    # weights = balance_weights(X_train.getDF(), 'tags', X_train.getLabelEncoder())
    # weights = np.clip(weights,0.02,0.2) # We need to let the net view the most common classes or learning is too slow

    # Creating a validation split
    train_idx, valid_idx = train_valid_split(X_train, 0.2)
    
    # weights[valid_idx] = 0
    
    # train_sampler = WeightedRandomSampler(weights, len(train_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetSampler(valid_idx)
    
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
        score, loss, threshold = validate(epoch, valid_loader, model, criterion, X_train.getLabelEncoder())
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
    X_test = KaggleAmazonDataset('./data/sample_submission_v2.csv','./data/test-jpg/','.jpg',
                                  ds_transform_raw
                                 )
    test_loader = DataLoader(X_test,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True)
    
    # Load model from best iteration
    logger.info('===> loading best model for prediction')
    checkpoint = torch.load(os.path.join(save_dir,
                                        run_name + '-model_best.pth'
                                        )
                           )
    model.load_state_dict(checkpoint['state_dict'])
    
    # Predict
    predictions = predict(test_loader, model) # TODO load model from the best on disk
    
    output(predictions,
           checkpoint['threshold'],
           X_test,
           X_train.getLabelEncoder(),
           './out',
           run_name,
           checkpoint['best_score']) # TODO early_stopping and use best_score
    
    ##########################################################
    
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))