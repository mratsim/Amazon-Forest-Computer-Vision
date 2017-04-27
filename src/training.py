from torch.autograd import Variable
import numpy as np
import time
import shutil
import torch
import os
import logging

## Get the same logger from main"
logger = logging.getLogger("Planet-Amazon")

def train(epoch,train_loader,model,loss_func, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            logger.info('Train Epoch: {:03d} [{:05d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader), loss.data[0]))

def validate(epoch,valid_loader,model,loss_func, score_func):
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
   
    score = score_func(true_target, predictions)
    logger.info("===> Validation - Avg. loss: {:.4f}\tScore: {:.4f}".format(avg_loss,score))
    return score, avg_loss
    
### Models are too big (500MB per file)
# def snapshot(dir_path, is_best, state):
#     snapshot_file = os.path.join(dir_path,
#                     time.strftime("%Y-%m-%d_%H%M-") +
#                     'snapshot-epoch_{:04d}_best_score_{:.4f}.pth'.format(state['epoch']-1, state['best_score']))
#     torch.save(state, snapshot_file)
#     print("Snapshot saved to {}".format(snapshot_file))
#     if is_best:
#         shutil.copyfile(snapshot_file, os.path.join(dir_path,'model_best.pth'))

def snapshot(dir_path, run_name, is_best, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    if is_best:
        torch.save(state, snapshot_file)
        logger.info("Snapshot saved to {}".format(snapshot_file))