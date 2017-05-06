from torch.autograd import Variable
import numpy as np
import logging
import torch.nn.functional as F
from tqdm import tqdm
import torch

from src.p_metrics import best_f2_score

## Get the same logger from main"
logger = logging.getLogger("Planet-Amazon")

##################################################
#### Validate function
def validate(epoch,valid_loader,model,loss_func,mlb):
    ## Volatile variables do not save intermediate results and build graphs for backprop, achieving massive memory savings.
    
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    logger.info("Starting Validation")
    for batch_idx, (data, target) in enumerate(tqdm(valid_loader)):
        true_labels.append(target.cpu().numpy())
        
        data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
    
        raw_pred = model(data)
        # Even though we use softmax for training, it doesn't give good result here
        # However activated neuro for weather will giv emuch larger response for much easier thresholding
        # pred = torch.cat(
        #                    (
        #                        F.softmax(raw_pred[:4]),
        #                        F.sigmoid(raw_pred[4:])
        #                    ), 0
        #       )
        pred = F.sigmoid(raw_pred)
        predictions.append(pred.data.cpu().numpy())
        
        total_loss += loss_func(raw_pred,target).data[0]
    
    avg_loss = total_loss / len(valid_loader)
    
    predictions = np.vstack(predictions)
    true_labels = np.vstack(true_labels)
   
    score, threshold = best_f2_score(true_labels, predictions)
    logger.info("Corresponding tags\n{}".format(mlb.classes_))
    
    logger.info("===> Validation - Avg. loss: {:.4f}\tF2 Score: {:.4f}".format(avg_loss,score))
    return score, avg_loss, threshold