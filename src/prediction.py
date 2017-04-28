from torch.autograd import Variable
import numpy as np
import logging
import os
import pandas as pd

## Get the same logger from main"
logger = logging.getLogger("Planet-Amazon")

##################################################
#### Prediction function
def predict(test_loader, model):    
    model.eval()
    predictions = []
    
    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.cuda(async=True)
        data = Variable(data, volatile=True)
    
        pred = model(data)
        predictions.append(pred.data.cpu().numpy())
    
    predictions = np.vstack(predictions)
    
    logger.info("===> Raw predictions done. Here is a snippet")
    print(predictions)
    return predictions

def output(predictions, threshold, X_test, mlb, dir_path, run_name, accuracy):
    
    raw_pred_path = os.path.join(dir_path, run_name + '-raw-pred-'+str(accuracy)+'.csv')
    np.savetxt(raw_pred_path,predictions,delimiter=";")
    logger.info("Raw predictions saved to {}".format(raw_pred_path))
    
    predictions = predictions > threshold
    
    result = pd.DataFrame({
        'image_name': X_test.X,
        'tags': mlb.inverse_transform(predictions)
    })
    result['tags'] = result['tags'].apply(lambda tags: " ".join(tags))
    
    result_path = os.path.join(dir_path, run_name + '-final-pred-'+str(accuracy)+'.csv')
    result.to_csv(result_path, index=False)