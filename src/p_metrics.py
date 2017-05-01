import numpy as np
import logging
from sklearn.metrics import fbeta_score
from scipy.optimize import fmin_l_bfgs_b

## Get the same logger from main"
logger = logging.getLogger("Planet-Amazon")

def best_f2_score(true_labels, predictions):

    def f_neg(threshold):
        ## Scipy tries to minimize the function so we must get its inverse
        return - fbeta_score(true_labels, predictions > threshold, beta=2, average='samples')

    # Initialization of best threshold search
    thr_0 = np.array([0.2 for i in range(17)])
    constraints = [(0.,1.) for i in range(17)]
    
    # Search, the epsilon step must be big otherwise there is no gradient
    thr_opt, score_neg, dico = fmin_l_bfgs_b(f_neg, thr_0, bounds=constraints, approx_grad=True, epsilon=1e-02)

    logger.info("===> Optimal threshold for each label:\n{}".format(thr_opt))
    
    score = - score_neg
    return score, thr_opt