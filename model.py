# from importlib.metadata import requires
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import toeplitz as toep
import scipy.io as sci
import matplotlib.pyplot as plt
import time
from torch.autograd import gradcheck
from torch import optim

class network(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(network, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.fc_1 = nn.Linear(31 * 31, 500)
        # self.fc_1 = nn.Linear(225, 225)
        # self.fc_2 = nn.Linear(225, 91)
        self.fc_1 = nn.Linear(94, 94)
        self.fc_2 = nn.Linear(94, 8)
        # self.fc_3 = nn.Linear(91, 1)
        # self.fc_2 = nn.Linear(91, 91)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.2)
        
    def forward(self, x, nx = 32, ny = 32):
        x = self.fc_1(x)
        x = self.relu(x)

        x = self.drop(x)
        x = self.fc_2(x)
        return x

class cnn_network(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(cnn_network, self).__init__()
        self.conv_1 = nn.Conv1d(in_ch, 16, kernel_size=3, stride=1, padding=1)
        self.norm_1 = nn.BatchNorm1d(16)
        # self.pool_1 = nn.MaxPool2d((2, 2))
        
        self.conv_2 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.norm_2 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(2)
        # self.pool_2 = nn.MaxPool2d((2, 2))

        self.conv_3 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.norm_3 = nn.BatchNorm1d(16)

        self.conv_4 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.norm_4 = nn.BatchNorm1d(16)

        # self.conv_5 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        # self.norm_5 = nn.BatchNorm1d(16)

        # self.conv_6 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        # self.norm_6 = nn.BatchNorm1d(16)

        # self.conv_4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=(1, 1))
        # self.norm_4 = nn.BatchNorm2d(256)

        # self.conv_5 = nn.Conv2d(256, 1, kernel_size=(3, 3), stride=1, padding=(1, 1))
        # self.norm_5 = nn.BatchNorm2d(1)

        # self.conv_6 = nn.Conv2d(256, 1, kernel_size=(3, 3), stride=1, padding=(1, 1))
        # self.norm_6 = nn.BatchNorm2d(1)

        # self.pool_4 = nn.MaxPool2d((2, 2))

        # self.conv_5 = nn.Conv2d(1, 1, kernel_size = (15, 15), stride = 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.fc_1 = nn.Linear(31 * 31, 500)
        # self.fc_1 = nn.Linear(225, 225)
        # self.fc_2 = nn.Linear(225, 91)
        self.fc_1 = nn.Linear(47, 8)
        # self.fc_2 = nn.Linear(94, 8)
        # self.fc_3 = nn.Linear(91, 1)
        # self.fc_2 = nn.Linear(91, 91)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.1)
        # self.global_pool = torch.mean(2, 3)
        
    def forward(self, x, nx = 32, ny = 32):
        # print(x.shape)
        x = self.conv_1(x)
        # print(x.shape)
        # print(x.shape)
        # x = self.norm_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        # # print(x.shape)
        # # x = self.pool_2(x)
        # # print(x.shape)
        # x = self.norm_2(x)
        x = self.pool(x)
        x = self.relu(x)

        x = self.conv_3(x)
        # # # print(x.shape)
        # # # x = self.pool_2(x)
        # # # print(x.shape)
        # x = self.norm_3(x)
        x = self.relu(x)

        x = self.conv_4(x)
        # # # # print(x.shape)
        # # # # x = self.relu(x)
        # # # x = self.pool_4(x)
        # # # # print(x.shape)
        # x = self.norm_4(x)
        # x = self.pool(x)
        x = self.relu(x)

        # x = self.conv_5(x)
        # # x = self.conv_5(x)
        # # x = self.norm_5(x)
        # x = self.pool(x)
        # x = self.relu(x)

        # x = self.conv_6(x)
        # # # x = self.norm_6(x)
        # x = self.relu(x)

        # x = self.flat(x)
        # x = self.global_pool(x)
        # print(x.shape)
        x = x.mean([1])
        # x = self.fc_1(x)
        # x = self.relu(x)
        # x = self.drop(x)
        x = self.fc_1(x)
        # x = self.relu(x)

        # x = self.drop(x)
        # x = self.fc_2(x)
        return x

class cnn_network2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(cnn_network2, self).__init__()
        self.conv_1 = nn.Conv1d(in_ch, 16, kernel_size=3, stride=1, padding=1)
        self.norm_1 = nn.BatchNorm1d(16)
        # self.pool_1 = nn.MaxPool2d((2, 2))
        
        self.conv_2 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.norm_2 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(2)
        # self.pool_2 = nn.MaxPool2d((2, 2))

        self.conv_3 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.norm_3 = nn.BatchNorm1d(16)

        self.conv_4 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.norm_4 = nn.BatchNorm1d(16)

        self.conv_5 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.norm_5 = nn.BatchNorm1d(16)

        self.conv_6 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.norm_6 = nn.BatchNorm1d(16)

        # self.conv_4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=(1, 1))
        # self.norm_4 = nn.BatchNorm2d(256)

        # self.conv_5 = nn.Conv2d(256, 1, kernel_size=(3, 3), stride=1, padding=(1, 1))
        # self.norm_5 = nn.BatchNorm2d(1)

        # self.conv_6 = nn.Conv2d(256, 1, kernel_size=(3, 3), stride=1, padding=(1, 1))
        # self.norm_6 = nn.BatchNorm2d(1)

        # self.pool_4 = nn.MaxPool2d((2, 2))

        # self.conv_5 = nn.Conv2d(1, 1, kernel_size = (15, 15), stride = 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.fc_1 = nn.Linear(31 * 31, 500)
        # self.fc_1 = nn.Linear(225, 225)
        # self.fc_2 = nn.Linear(225, 91)
        self.fc_1 = nn.Linear(23, 8)
        # self.fc_2 = nn.Linear(94, 8)
        # self.fc_3 = nn.Linear(91, 1)
        # self.fc_2 = nn.Linear(91, 91)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.1)
        # self.global_pool = torch.mean(2, 3)
        
    def forward(self, x, nx = 32, ny = 32):
        # print(x.shape)
        x = self.conv_1(x)
        # print(x.shape)
        # print(x.shape)
        # x = self.norm_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        # # print(x.shape)
        # # x = self.pool_2(x)
        # # print(x.shape)
        # x = self.norm_2(x)
        x = self.pool(x)
        x = self.relu(x)

        x = self.conv_3(x)
        # # # print(x.shape)
        # # # x = self.pool_2(x)
        # # # print(x.shape)
        # x = self.norm_3(x)
        x = self.relu(x)

        x = self.conv_4(x)
        # # # # print(x.shape)
        # # # # x = self.relu(x)
        # # # x = self.pool_4(x)
        # # # # print(x.shape)
        # x = self.norm_4(x)
        # x = self.pool(x)
        x = self.relu(x)

        x = self.conv_5(x)
        # x = self.conv_5(x)
        # x = self.norm_5(x)
        x = self.pool(x)
        x = self.relu(x)

        x = self.conv_6(x)
        # # x = self.norm_6(x)
        x = self.relu(x)

        # x = self.flat(x)
        # x = self.global_pool(x)
        # print(x.shape)
        x = x.mean([1])
        # x = self.fc_1(x)
        # x = self.relu(x)
        # x = self.drop(x)
        x = self.fc_1(x)
        # x = self.relu(x)

        # x = self.drop(x)
        # x = self.fc_2(x)
        return x
    
class lstm_network(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(lstm_network, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.fc_1 = nn.Linear(31 * 31, 500)
        # self.fc_1 = nn.Linear(225, 225)
        # self.fc_2 = nn.Linear(225, 91)
        self.lstm = nn.LSTM(1, 1, 1, batch_first=True)
        self.pool = nn.MaxPool1d(2)
        self.conv_1 = nn.Conv1d(in_ch, 16, kernel_size=3, stride=1, padding=1)        
        # self.conv_1 = nn.Conv2d(in_ch, 16, kernel_size=(3, 3), stride = (1, 1), padding = (1, 1))
        self.conv_2 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)

        self.fc_1 = nn.Linear(47, 8)
        self.fc_2 = nn.Linear(94, 8)
        # self.fc_3 = nn.Linear(23, 8)
        # self.fc_2 = nn.Linear(91, 91)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.2)
        
    def forward(self, inp, nx = 32, ny = 32):
        # x,_ = self.lstm(inp)
        # x = x[:, -1, :]
        # print(x.shape)
        # x = x.squeeze(2)
        # x = x.unsqueeze(1)
        # # print(x.shape)
        # inp = inp.squeeze(2)
        # inp = inp.unsqueeze(1)
        # x = torch.cat((inp, x), 1)
        # print(x.shape)
        # x = x.unsqueeze(3)
        x = self.conv_1(inp)
        # print(x.shape)
        # print(x.shape)
        x = self.relu(x)

        # x = x.squeeze(3)
        x = self.conv_2(x)
        x = self.pool(x)
        x = self.relu(x)

        x = self.conv_3(x)
        # print(x.shape)
        x = self.relu(x)

        x = self.conv_4(x)
        # x = self.pool(x)
        x = self.relu(x)

        x = x.mean([1])
        # print(x.shape)
        # print(y.shape)
        # x = x.unsqueeze(2)
        # x,_ = self.lstm(x)
        # x = x.squeeze(2)
        # print(x.shape)
        # x = x[:, -1, :]
        # x = torch.cat((x, y), dim = 1)

        # x = self.drop(x)
        x = self.fc_1(x)
        # x = self.relu(x)
        # x = self.drop(x)
        # x = self.fc_2(x)
        # x = self.relu(x)
        # x = self.fc_3(x)
        return x

class lstm_network2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(lstm_network2, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.fc_1 = nn.Linear(31 * 31, 500)
        # self.fc_1 = nn.Linear(225, 225)
        # self.fc_2 = nn.Linear(225, 91)
        # self.lstm = nn.LSTM(1, 1, 1, batch_first=True)
        self.pool = nn.MaxPool1d(2)
        self.conv_1 = nn.Conv1d(in_ch, 16, kernel_size=3, stride=1, padding=1)        
        # self.conv_1 = nn.Conv2d(in_ch, 16, kernel_size=(3, 3), stride = (1, 1), padding = (1, 1))
        self.conv_2 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)

        self.conv_5 = nn.Conv1d(in_ch, 8, kernel_size=3, stride=1, padding=1)        
        # self.conv_1 = nn.Conv2d(in_ch, 16, kernel_size=(3, 3), stride = (1, 1), padding = (1, 1))
        self.conv_6 = nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv_7 = nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv_8 = nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1)

        self.fc_1 = nn.Linear(23, 8)
        self.fc_2 = nn.Linear(47, 8)
        self.fc_3 = nn.Linear(23, 8)
        # self.fc_2 = nn.Linear(91, 91)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.2)
        
    def forward(self, inp, nx = 32, ny = 32):
        # x,_ = self.lstm(inp)
        # x = x[:, -1, :]
        # print(x.shape)
        # x = x.squeeze(2)
        # x = x.unsqueeze(1)
        # # print(x.shape)
        # inp = inp.squeeze(2)
        # inp = inp.unsqueeze(1)
        # x = torch.cat((inp, x), 1)
        # print(x.shape)
        # x = x.unsqueeze(3)
        x = self.conv_1(inp)
        # print(x.shape)
        # print(x.shape)
        x = self.relu(x)

        # x = x.squeeze(3)
        x = self.conv_2(x)
        x = self.pool(x)
        x = self.relu(x)

        x = self.conv_3(x)
        # print(x.shape)
        x = self.relu(x)

        x = self.conv_4(x)
        x = self.pool(x)
        x = self.relu(x)

        x = x.mean([1])

        y = self.conv_5(inp)
        # print(x.shape)
        # print(x.shape)
        y = self.pool(y)
        y = self.relu(y)

        # x = x.squeeze(3)
        y = self.conv_6(y)
        y = self.pool(y)
        y = self.relu(y)

        y = self.conv_7(y)
        # print(x.shape)
        y = self.relu(y)

        y = self.conv_8(y)
        y = self.relu(y)

        y = y.mean([1])
        # print(x.shape)
        # print(y.shape)
        # x = x.unsqueeze(2)
        # x,_ = self.lstm(x)
        # x = x.squeeze(2)
        # print(x.shape)
        # x = x[:, -1, :]
        # x = torch.cat((x, y), dim = 1)
        x = y
        # x = self.drop(x)
        x = self.fc_1(x)
        # x = self.relu(x)
        # x = self.drop(x)
        # x = self.fc_2(x)
        # x = self.relu(x)
        # x = self.fc_3(x)
        return x