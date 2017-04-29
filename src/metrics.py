import numpy as np
import logging
from sklearn.metrics import fbeta_score

## Get the same logger from main"
logger = logging.getLogger("Planet-Amazon")

def search_best_threshold(p_valid, y_valid, try_all=False, verbose=False):
    p_valid, y_valid = np.array(p_valid), np.array(y_valid)

    best_threshold = 0
    best_score = -1
    totry = np.arange(0,1,0.005) if try_all is False else np.unique(p_valid)
    for t in totry:
        score = fbeta_score(y_valid, p_valid > t, beta=2, average='samples')
        if score > best_score:
            best_score = score
            best_threshold = t
    logger.info("===> Optimal threshold for each label:\n{}".format(best_threshold))
    return best_score, best_threshold

def best_f2_score(true_labels, predictions):
    return search_best_threshold(predictions, true_labels)