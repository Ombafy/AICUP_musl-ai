import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from data import Training_Dataset, Val_Dataset, AddGaussianNoise
from attrdict import AttrMap
from model import *
# from model_radon import network
import sys
from eval import test
from torch.autograd import Variable
from torch import optim
# from weights_init import weights
from utils import save_image, checkpoint, save_confusion_matrix
from log_record import Record
import pickle
import matplotlib.pyplot as plt
import torchvision as tv
import os
# from piq import SSIMLoss

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# def weights(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0, 0.2)
#         m.weight.data = (m.weight.data + torch.transpose(m.weight.data, 2, 3))/2
#         # m.weight.data.fill_(0)
#     elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
#         m.weight.data.fill_(0.01)
#         m.bias.data.fill_(0)

def weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0, 0.2)
        # m.weight.data = (m.weight.data + torch.transpose(m.weight.data, 2, 3))/2
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm1d') != -1 or classname.find('InstanceNorm1d') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

def weights_2(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0, 0.2)
        m.weight.data = (m.weight.data + torch.transpose(m.weight.data, 2, 3))/2
        # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
    elif classname.find('diagonalize'):
        r, c = m.weights.data.shape
        for i in range(r):
            for j in range(c):
                m.weights.data[i][j] = -1 * abs(i - j) / 10
        m.weights.data = torch.exp(m.weights.data)

def accuracy(preds, targets):
    pred_indices = torch.argmax(preds, 1)
    correct = (pred_indices == targets).sum()
    return correct.item()

class training():
    def train(self, config):
        print('===> Loading datasets')
        train_dataset = Training_Dataset(config)
        test_dataset = Val_Dataset(config)
        training_data_loader = DataLoader(dataset=train_dataset, num_workers=config.threads,
                                          batch_size=config.batchsize, shuffle=True)
        test_data_loader = DataLoader(dataset=test_dataset, num_workers=config.threads,
                                      batch_size=config.test_batchsize, shuffle=True)
        print("===> Dataset loaded")
        device = torch.device("cuda" if (torch.cuda.is_available() and config.cuda == True) else "cpu")
        print(device)
        # Loading Network
        # net = network(1, 1).to(device)
        net = cnn_network(1, 1).to(device)
        # net.load_state_dict(torch.load(config.net_pretrained))
        # initialize network weights using some initialization method
        net.apply(weights)

        opt_net = optim.Adam(net.parameters(), lr=config.lr, weight_decay = 1e-5)
        # criterionMSE = nn.MSELoss()
        criterionClass = nn.CrossEntropyLoss()

        if config.cuda:
            # criterionMSE = criterionMSE.to(device)
            criterionClass = criterionClass.to(device)

        logreport = Record(log_dir=config.out_dir)

        torch.cuda.empty_cache()
        # log_test = test(config, test_data_loader, net, criterionMSE_Eval, 1, device)
        for epoch in range(1, config.epoch + 1):
            avg_loss = 0
            avg_acc = 0
            for iteration, batch in enumerate(training_data_loader, 1):
                torch.cuda.empty_cache()
                x, t = batch[0].to(device), batch[1].to(device)
                # print(t.shape)
                # print(x.shape)
                # mean = x.mean([1, 2])
                # std = x.std([1, 2])
                # normalize = tv.transforms.Normalize(mean, std)
                # x = normalize(x)
                # print(x.shape)
                x = x.unsqueeze(1)
                opt_net.zero_grad()
                out = net(x.float())
                loss = criterionClass(out, t)
                loss.backward()
                opt_net.step()

                # if iteration == 1 or iteration % 15 == 0:
                #     print("Epoch[{}/{}]({}/{}): loss_mse: {:.4f}".format(
                #         epoch, config.epoch, iteration, len(training_data_loader), loss.item()))
                avg_loss = avg_loss + loss.item()
                avg_acc += accuracy(out, t)

            log_train = {}
            log_train['epoch'] = epoch
            log_train['loss'] = avg_loss / len(training_data_loader)
            # print(len(training_data_loader))
            log_train['acc'] = avg_acc / len(train_dataset)
            # print("===> Avg. Training LOSS " + format(avg_loss / len(training_data_loader)) + " for epoch " + format(epoch))
            print("===> Avg. Training MSE " + format(log_train['loss']) + " for epoch " + format(epoch))
            print("===> Avg. Training ACC " + format(log_train['acc']) + " for epoch " + format(epoch))

            with torch.no_grad():
                log_test, y_true, y_pred = test(config, test_dataset, test_data_loader, net, criterionClass, epoch, device)
            logreport(log_train, log_test)
            if epoch % config.snapshot == 0:
                checkpoint(config, epoch, net)
                save_confusion_matrix(config, epoch, y_true, y_pred)
            logreport.save_graph()

if __name__ == '__main__':
    with open('config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)
    obj = training()
    obj.train(config)
