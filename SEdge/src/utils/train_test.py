import torch
import numpy as np

from tqdm import tqdm
import time

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def regularize_b(weights): 
    out_channels = weights.shape[0]
    self_var = 4/12 
    cs = (weights@weights.T - self_var*torch.eye(out_channels, device=weights.device))*(1-torch.eye(out_channels, device=weights.device)) 
    return torch.mean(cs**2).sqrt()


def train_one_epoch(model, criterion, optimizer, trainloader, regularize=False, scheduler=None, sub_epoch_documentation=10, augments_use=None):
    running_loss = 0.0
    running_reg_loss = 0.0
    correct = 0
    model.train()
    sub_epoch_loss = []
    sub_epoch_reg_loss = []
    sub_epoch_acc = []
    sub_epoch_epoch = []
    sub_epoch_lr = []
    alpha = 0.1
    last_loss = None
    device = next(model.parameters()).device  # get device the model is located on    

    time_start = time.time()

    with tqdm(enumerate(trainloader),total=len(trainloader), unit=' steps',mininterval=1.0, ncols=150, desc=f"IGNORE: loss={0:>6.4f} running loss={0:>6.4f}") as pbar:
        for i, (inputs, labels) in pbar:
            inputs = inputs.to(device, non_blocking=True)  # move data to same device as the model
            labels = labels.to(device, non_blocking=True)
            # zero the parameter gradients
            optimizer.zero_grad(True)
            # forward + backward + optimize
            
            if augments_use is not None:
                inputs = augments_use(inputs)

            outputs = model(inputs)
            loss_weight = criterion(outputs, labels)
            pred = get_likely_index(outputs)
            correct += number_of_correct(pred, labels)
            weight_reg_loss = 0
            if regularize:
                w = model.decoder.weight
                orth = (w@w.T)
                orth = orth*(1-torch.eye(w.shape[0], device=w.device))
                weight_reg_loss = 0.1 * torch.mean(orth**2).sqrt()
                loss = loss_weight + weight_reg_loss
            else:
                loss = loss_weight

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # print statistics
            running_loss += loss_weight.item()
            if last_loss is None:
                last_loss = loss_weight.item()
            last_loss = alpha*loss_weight.item() + (1-alpha)*last_loss

            if regularize:
                running_reg_loss += weight_reg_loss.item()
                pbar.set_description(f"IGNORE: loss={loss.item():>6.4f} running loss={running_loss / (i + 1):>6.4f} reg_loss={running_reg_loss / (i + 1):>6.4f}")
            else:
                pbar.set_description(f"IGNORE: loss={loss.item():>6.4f} running loss={running_loss / (i + 1):>6.4f}")
            # pbar.update(1)
            
            if sub_epoch_documentation and i % sub_epoch_documentation == 0:
                sub_epoch_loss.append(last_loss)
                if regularize:
                    sub_epoch_reg_loss.append(weight_reg_loss.item())
                sub_epoch_acc.append(correct / (inputs.shape[0]*(i + 1)))
                sub_epoch_epoch.append(i/len(trainloader))
                if scheduler is not None:
                    sub_epoch_lr.append(scheduler.get_lr())

            if i % 1000 == 0:
                pbar.write(f"step={i} loss={loss.item():>6.4f} running loss={running_loss / (i + 1):>6.4f} , time={time.time()-time_start:>6.4f} expected time={(time.time()-time_start)/(i+1)*len(trainloader):>6.4f}")


    train_loss = running_loss / len(trainloader)
    train_acc = correct / len(trainloader.dataset)

    out_dict =  {'epoch': sub_epoch_epoch, 'train_loss': sub_epoch_loss, 'train_acc': sub_epoch_acc}
    if scheduler is not None:
        lr_tmp = np.array(sub_epoch_lr)
        for i in range(lr_tmp.shape[1]):
            out_dict[f'lr_{i}'] = lr_tmp[:,i].tolist()
        out_dict['lr'] = lr_tmp[:,0].tolist()
    if regularize:
        out_dict['sub_epoch_reg_loss'] = sub_epoch_reg_loss
    return train_loss, train_acc, out_dict


def evaluate(model, criterion, valloader, transform=None, return_confusion=False):
    losses = []
    model.eval()
    correct = 0
    confusion_matrix = np.zeros((35,35)).astype(np.float32)
    #model.reset_hidden_state(batch_size)
    if getattr(model, 'parameters' , None) is not None:
        try:
            device = next(model.parameters()).device  # get device the model is located on
        except :
            device = 'cpu'
    else:
        device = 'cpu'
    with torch.no_grad():
        with tqdm(enumerate(valloader),total=len(valloader), desc=f"IGNORE: loss={0:>6.4f}") as pbar:
            for step, (inputs, labels) in pbar:
                inputs = inputs.to(device)  # move data to same device as the model
                if transform!=None:
                    inputs = transform(inputs.squeeze(-1)).unsqueeze(-1)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                pred = get_likely_index(outputs)
                correct += number_of_correct(pred,labels)
                pbar.set_postfix_str(f"IGNORE: step={step} acc={correct/(step+1)/inputs.shape[0] :>6.4f}")
                if return_confusion:
                    for i in range(pred.shape[0]):
                        confusion_matrix[labels[i].item(), pred[i].item()] += 1
                # pbar.set_postfix_str(f"IGNORE: acc={correct/len(valloader.dataset) :>6.4f}")    
    if return_confusion:
        # normalize confusion matrix
        confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1, keepdims=True)
        confusion_matrix = np.nan_to_num(confusion_matrix)
        confusion_matrix = confusion_matrix * 100
        return np.sum(losses)/len(valloader), correct / len(valloader.dataset), confusion_matrix
    return np.sum(losses)/len(valloader), correct / len(valloader.dataset)
