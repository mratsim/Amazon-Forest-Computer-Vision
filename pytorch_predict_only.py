## Custom Imports
from src.p_dataload import KaggleAmazonDataset
from src.p_neuro import Net, ResNet50
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
## Variables setup
model = ResNet50(17).cuda()
# model = Net().cuda()
# model = WideResNet(16, 17, 4, 0.3)

epochs = 22
batch_size = 16

# Run name
run_name = time.strftime("%Y-%m-%d_%H%M-") + "resnet50-pycaffe"

## Normalization on dataset mean/std
# normalize = transforms.Normalize(mean=[0.30249774, 0.34421161, 0.31507745],
#                                  std=[0.13718569, 0.14363895, 0.16695958])
    
## Normalization on ImageNet mean/std for finetuning
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

# Note, p_training has lr_decay automated
# optimizer = optim.Adam(model.parameters(), lr=0.1) # From scratch # Don't use Weight Decay with PReLU
# optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)  # From scratch
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9) # Finetuning whole model
# optimizer = optim.SGD(model.classifier.parameters(), lr=1e-2, momentum=0.9) # Finetuning classifier

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
    
    ## Normalization only for validation and test
    ds_transform_raw = transforms.Compose([
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     normalize
                     ])
    
    
    X_test = KaggleAmazonDataset('./data/sample_submission.csv','./data/test-jpg/','.jpg',
                                  ds_transform_raw
                                 )
    test_loader = DataLoader(X_test,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True)
    
    # Load model from best iteration
    model_path = './snapshots/2017-05-01_1923-resnet50-pycaffe-model_best.pth'
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
           '2017-05-01_1923-resnet50-pycaffe',
           checkpoint['best_score']) # TODO early_stopping and use best_score
    
    ##########################################################
    
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))