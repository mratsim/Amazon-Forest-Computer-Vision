from torch.autograd import Variable
import numpy as np
import logging
from sklearn.metrics import fbeta_score
from scipy.optimize import fmin_cobyla

## Get the same logger from main"
logger = logging.getLogger("Planet-Amazon")

##################################################################################
## Metrics
## Given the labels imbalance we can't use the same threshold for each label.
## We could implement our own maximizer on all 17 classes but scipy.optimize already have
## 4 optimizations algorithms in C/Fortran that can work with constraints: L-BFGS-B, TNC, COBYLA and SLSQP.
## Of those only cobyla doesn't rely on 2nd order hessians which are error-prone with our function
## based on inequalities

# Cobyla constraints are build by comparing return value with 0.
# They must be >= 0 or be rejected

def constr_sup0(x):
    return np.min(x)
def constr_inf1(x):
    return 1 - np.max(x)

def f2_score(true_target, predictions):

    def f_neg(threshold):
        ## Scipy tries to minimize the function so we must get its inverse
        return - fbeta_score(true_target, predictions > threshold, beta=2, average='samples')

    # Initialization of best threshold search
    thr_0 = np.array([0.2 for i in range(17)])
    
    # Search
    thr_opt = fmin_cobyla(f_neg, thr_0, [constr_sup0,constr_inf1], disp=0)

    logger.info("===> Optimal threshold for each label:\n{}".format(thr_opt))
    
    score = fbeta_score(true_target, predictions > thr_opt, beta=2, average='samples')
    return score, thr_opt

##################################################
#### Validate function
def validate(epoch,valid_loader,model,loss_func):
    ## Volatile variables do not save intermediate results and build graphs for backprop, achieving massive memory savings.
    
    model.eval()
    total_loss = 0
    predictions = []
    true_target = []
    
    for batch_idx, (data, target) in enumerate(valid_loader):
        true_target.append(target.cpu().numpy())
        
        data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
    
        pred = model(data)
        predictions.append(pred.data.cpu().numpy())
        
        total_loss += loss_func(pred,target).data[0]
    
    avg_loss = total_loss / len(valid_loader)
    
    predictions = np.vstack(predictions)
    true_target = np.vstack(true_target)
   
    score, threshold = f2_score(true_target, predictions)
    
    logger.info("===> Validation - Avg. loss: {:.4f}\tScore: {:.4f}".format(avg_loss,score))
    return score, avg_loss, threshold