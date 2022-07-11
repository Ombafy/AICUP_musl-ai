import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from data import Test_Dataset, AddGaussianNoise
from attrdict import AttrMap
from model import *
import sys
from eval import test, accuracy
from torch.autograd import Variable
from torch import optim
from utils import save_image, checkpoint, save_confusion_matrix
from log_record import Record
import pickle
import matplotlib.pyplot as plt
import torchvision as tv
import os
import pandas as pd
	
if __name__ == "__main__":
  with open('config.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
  config = AttrMap(config)
  test_dataset = Test_Dataset(config)
  test_data_loader = DataLoader(dataset=test_dataset, num_workers=config.threads,
                                batch_size=config.test_batchsize, shuffle = False)
  print("===> Dataset loaded")
  device = torch.device("cuda" if (torch.cuda.is_available() and config.cuda == True) else "cpu")
  print(device)
  # Loading Network
  # net = network(1, 1).to(device)
  net = cnn_network(1, 1).to(device)
  net.load_state_dict(torch.load(config.net_pretrained))
  # initialize network weights using some initialization method
  torch.cuda.empty_cache()
  predictions = []
  with torch.no_grad():
    for iteration, batch in enumerate(test_data_loader, 1):
      torch.cuda.empty_cache()
      x = batch[0].to(device)
      if config.cuda:
          x = x.to(device)
      x = x.unsqueeze(0)
      x = x.unsqueeze(0)
      out = net(x.float())
      pred_index = torch.argmax(out, 1).item()
      predictions.append(pred_index)
  
  df_submit = pd.DataFrame(predictions, columns=["PredictedClass"])
  df_submit.to_csv("submission.zip", sep=";", decimal=".", index=False, compression=dict(method='zip',
                      archive_name='submission.csv') ) 