import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from data import Test_Dataset, AddGaussianNoise
from attrdict import AttrMap
from model import *
import sys
from torch.autograd import Variable
from torch import optim
from utils import save_image, checkpoint, save_confusion_matrix
from log_record import Record
import pickle
import matplotlib.pyplot as plt
import torchvision as tv
import os
import pandas as pd
import numpy as np

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
  net1 = cnn_network(1, 1).to(device)
  net1.load_state_dict(torch.load(config.net_pretrained_1, map_location = device))
  
  net2 = cnn_network(1, 1).to(device)
  net2.load_state_dict(torch.load(config.net_pretrained_2, map_location = device))
  
  net3 = lstm_network2(1, 1).to(device)
  net3.load_state_dict(torch.load(config.net_pretrained_3, map_location = device))
  
  net4 = lstm_network(1, 1).to(device)
  net4.load_state_dict(torch.load(config.net_pretrained_4, map_location = device))
  
  net5 = lstm_network(1, 1).to(device)
  net5.load_state_dict(torch.load(config.net_pretrained_5, map_location = device))
  
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

      out = net1(x.float())
      pred_index1 = torch.argmax(out, 1).item()
      
      out = net2(x.float())
      pred_index2 = torch.argmax(out, 1).item()
      
      out = net3(x.float())
      pred_index3 = torch.argmax(out, 1).item()
      
      # extra
      out = net4(x.float())
      pred_index4 = torch.argmax(out, 1).item()
      
      out = net5(x.float())
      pred_index5 = torch.argmax(out, 1).item()
      # extra

      dicti = {}
      if pred_index1 in dicti:
        dicti[pred_index1] += 1
      else:
        dicti[pred_index1] = 1

      if pred_index2 in dicti:
        dicti[pred_index2] += 1
      else:
        dicti[pred_index2] = 1

      if pred_index3 in dicti:
        dicti[pred_index3] += 1
      else:
        dicti[pred_index3] = 1
# extra
      if pred_index4 in dicti:
        dicti[pred_index4] += 1
      else:
        dicti[pred_index4] = 1

      if pred_index5 in dicti:
        dicti[pred_index5] += 1
      else:
        dicti[pred_index5] = 1
# extra

      final_pred = None
      temp = []
      for keys in dicti:
        if dicti[keys] > 2:
          final_pred = keys
          break
        elif dicti[keys] == 2:
          temp.append(keys)

      if final_pred == None:
        if len(temp) != 0:
          final_pred = np.random.choice(temp)
        else:
          for_rand = [pred_index1, pred_index2, pred_index3, pred_index4, pred_index5]
          final_pred = np.random.choice(for_rand)

      # final_pred = None
      # for keys in dicti:
      #   if dicti[keys] >= 2:
      #     final_pred = keys
      #     break
      # if final_pred == None:
      #   for_rand = [pred_index1, pred_index2, pred_index3]
      #   final_pred = np.random.choice(for_rand)

      predictions.append(final_pred)
  
  df_submit = pd.DataFrame(predictions, columns=["PredictedClass"])
  df_submit.to_csv("submission.zip", sep=";", decimal=".", index=False, compression=dict(method='zip',
                      archive_name='submission.csv') ) 