from torch.autograd import Variable
import numpy as np

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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
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
    print("===> Avg. loss: {:.4f}\tScore: {:.4f}".format(avg_loss,score))
