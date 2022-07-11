import os
# import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import json
from skimage.transform import radon
from sklearn.metrics import confusion_matrix

def save_image(out_dir, x, num, filename, epoch):
    test_dir = out_dir
    if filename is not None:
        test_path = os.path.join(test_dir, filename + ".npy")
    else:
        test_path = os.path.join(test_dir, 'test_{0:04d}.npy'.format(num))

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    #cv2.imwrite(test_path, x_numpy)
    np.save(test_path, x)

def checkpoint(config, epoch, net):
    model_dir = os.path.join(config.out_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    network_path = os.path.join(model_dir, 'network_model_epoch_{}.pth'.format(epoch))      
    torch.save(net.state_dict(), network_path)
    print("Checkpoint saved to {}".format(model_dir))

def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def save_confusion_matrix(config, epoch, y_true, y_pred):
    classes = [i for i in range(0, 8)]
    conf = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(conf, index = classes, columns = classes)
    plt.figure(figsize = (20, 15))
    sn.heatmap(df_cm, annot=True)
    save_path = os.path.join(config.out_dir, 'cm_epoch_{}'.format(epoch))
    plt.savefig(save_path)