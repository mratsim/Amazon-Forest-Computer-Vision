from torch.autograd import Variable
import numpy as np
import logging
import torch.nn.functional as F

from src.p_metrics import best_f2_score

## Get the same logger from main"
logger = logging.getLogger("Planet-Amazon")

##################################################
#### Validate function
def validate(epoch,valid_loader,model,loss_func):
    ## Volatile variables do not save intermediate results and build graphs for backprop, achieving massive memory savings.
    
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    for batch_idx, (data, target) in enumerate(valid_loader):
        true_labels.append(target.cpu().numpy())
        
        data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
    
        pred = F.sigmoid(model(data))
        predictions.append(pred.data.cpu().numpy())
        
        total_loss += loss_func(pred,target).data[0]
    
    avg_loss = total_loss / len(valid_loader)
    
    predictions = np.vstack(predictions)
    true_labels = np.vstack(true_labels)
   
    score, threshold = best_f2_score(true_labels, predictions)
    
    logger.info("===> Validation - Avg. loss: {:.4f}\tF2 Score: {:.4f}".format(avg_loss,score))
    return score, avg_loss, threshold