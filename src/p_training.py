from torch.autograd import Variable
import torch
import os
import logging

## Get the same logger from main"
logger = logging.getLogger("Planet-Amazon")


def lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        logger.info('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def train(epoch,train_loader,model,loss_func, optimizer):
    model.train()
    optimizer = lr_scheduler(optimizer, epoch)
    
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data), Variable(target, requires_grad=False)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.info('Train Epoch: {:03d} [{:05d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader), loss.data[0]))

def snapshot(dir_path, run_name, is_best, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    if is_best:
        torch.save(state, snapshot_file)
        logger.info("Snapshot saved to {}".format(snapshot_file))