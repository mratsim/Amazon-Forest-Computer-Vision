import numpy as np
import logging
from sklearn.metrics import fbeta_score
from scipy.optimize import fmin_l_bfgs_b, basinhopping
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from timeit import default_timer as timer


## Get the same logger from main"
logger = logging.getLogger("Planet-Amazon")

def best_f2_score(true_labels, predictions):

    def f_neg(threshold):
        ## Scipy tries to minimize the function so we must get its inverse
        return - fbeta_score(true_labels, predictions > threshold, beta=2, average='samples')

    # Initialization of best threshold search
    thr_0 = [0.20] * 17
    constraints = [(0.,1.)] * 17
    def bounds(**kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= 1))
        tmin = bool(np.all(x >= 0)) 
        return tmax and tmin
    
    # Search using L-BFGS-B, the epsilon step must be big otherwise there is no gradient
    minimizer_kwargs = {"method": "L-BFGS-B",
                        "bounds":constraints,
                        "options":{
                            "eps": 0.05
                            }
                       }
    
    # We combine L-BFGS-B with Basinhopping for stochastic search with random steps
    logger.info("===> Searching optimal threshold for each label")
    start_time = timer()
    
    opt_output = basinhopping(f_neg, thr_0,
                                stepsize = 0.1,
                                minimizer_kwargs=minimizer_kwargs,
                                niter=10,
                                accept_test=bounds)
    
    end_time = timer()
    logger.info("===> Optimal threshold for each label:\n{}".format(opt_output.x))
    logger.info("Threshold found in: %s seconds" % (end_time - start_time))
    
    score = - opt_output.fun
    return score, opt_output.x


# We use real valued F2 score for training. Input can be anything between 0 and 1.
# Threshold is not differentiable so we don't use it during training
# We get a smooth F2 score valid for real values and not only 0/1
def torch_f2_score(y_true, y_pred):
    return torch_fbeta_score(y_true, y_pred, 2)

def torch_fbeta_score(y_true, y_pred, beta, eps=1e-9):
    beta2 = beta**2

    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))


class SmoothF2Loss(nn.Module):
    def __init__(self):
        super(MeanF2Loss, self).__init__()
    
    def forward(self, input, target):
        return 1 - torch_f2_score(target, torch.sigmoid(input))