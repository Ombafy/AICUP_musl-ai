import sys
import torch
import torchvision as tv
from skimage.transform import radon
from numpy import unravel_index
import matplotlib.pyplot as plt
import numpy as np

def accuracy(preds, targets, y_true, y_pred):
    # print(preds.shape)
    # print(targets.shape)
    pred_indices = torch.argmax(preds, 1)
    correct = (pred_indices == targets).sum()
    y_pred.append(pred_indices.item())
    y_true.append(targets.item())
    return correct.item()

def test(config, test_dataset, test_data_loader, net, criterionMSE, epoch,  device):
    avg_mse = 0
    avg_acc = 0
    y_pred = []
    y_true = []
    counter = 1
    for i, batch in enumerate(test_data_loader, 1):
        x, t = batch[0].to(device), batch[1].to(device)
        # mean = x.mean([1, 2])
        # std = x.std([1, 2])
        # normalize = tv.transforms.Normalize(mean, std)
        # # x = normalize(x)
        if config.cuda:
            x = x.to(device)
            t = t.to(device)
        x = x.unsqueeze(0)
        out = net(x.float())
        counter += 1
        avg_acc += accuracy(out, t, y_true, y_pred)

    avg_acc = avg_acc / len(test_dataset)
    print("===> Avg. Test ACC " + format(avg_acc) + " for epoch " + format(epoch))
    
    log_test = {}
    log_test['epoch'] = epoch
    # log_test['loss'] = avg_mse
    log_test['acc'] = avg_acc
    return log_test, y_true, y_pred
