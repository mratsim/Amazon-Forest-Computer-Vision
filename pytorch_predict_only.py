## Custom Imports
from src.p_dataload import KaggleAmazonDataset
from src.p_neuro import Net, ResNet50, DenseNet121
from src.p_training import train, snapshot
from src.p_validation import validate
from src.p_model_selection import train_valid_split
from src.p_logger import setup_logs
from src.p_prediction import predict, output
from src.p_data_augmentation import ColorJitter

## Utilities
import random
import logging
import time
from timeit import default_timer as timer
import os

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

# Run name
run_name = "2017-05-04_1730-thresh_densenet121-predict-only"

model = DenseNet121(17).cuda()
batch_size = 32

## Normalization on dataset mean/std
# normalize = transforms.Normalize(mean=[0.30249774, 0.34421161, 0.31507745],
#                                  std=[0.13718569, 0.14363895, 0.16695958])
    
## Normalization on ImageNet mean/std for finetuning
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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
    
    ## Normalization only for validation and test
    ds_transform_raw = transforms.Compose([
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     normalize
                     ])
    

    
    X_test = KaggleAmazonDataset('./data/sample_submission_v2.csv','./data/test-jpg/','.jpg',
                                  ds_transform_raw
                                 )
    test_loader = DataLoader(X_test,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True)
    
    # Load model from best iteration
    model_path = './snapshots/2017-05-04_1730-thresh_densenet121-model_best.pth'
    logger.info('===> loading {} for prediction'.format(model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Predict
    predictions = predict(test_loader, model) # TODO load model from the best on disk
    
    # Output
    X_train = KaggleAmazonDataset('./data/train.csv','./data/train-jpg/','.jpg')
                                 
    
    output(predictions,
           checkpoint['threshold'],
           X_test,
           X_train.getLabelEncoder(),
           './out',
           '2017-05-04_1730-thresh_densenet121',
           checkpoint['best_score'])
    
    ##########################################################
    
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))