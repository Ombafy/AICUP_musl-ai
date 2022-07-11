import json
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

class Record():
    def __init__(self, log_dir, log_train_name='log_train_record', log_test_name='log_test_record'):
        self.log_dir = log_dir
        self.log_train_name = log_train_name
        self.log_test_name = log_test_name
        self.log_train = []
        self.log_test = []

    def __call__(self, log_train, log_test):
        self.log_train.append(log_train)
        if log_test:
            self.log_test.append(log_test)

        with open(os.path.join(self.log_dir, self.log_train_name), 'w') as f:
            json.dump(self.log_train, f, indent=4)

        if log_test:
            with open(os.path.join(self.log_dir, self.log_test_name), 'w') as f:
                json.dump(self.log_test, f, indent=4)

    def save_graph(self):
        epoch_train = []
        epoch_test = []
        train_acc = []
        test_acc = []

        for l, m in zip(self.log_train, self.log_test):
            epoch_train.append(l['epoch'])
            train_acc.append(l['acc'])

            epoch_test.append(m['epoch'])
            test_acc.append(m['acc'])

        epoch = np.asarray(epoch_train)
        train_acc = np.asarray(train_acc)
        test_acc = np.asarray(test_acc)

        o = plt.plot(epoch, train_acc, label='Train')
        o = plt.plot(epoch, test_acc, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.log_dir, 'lossgraph.png'))
        plt.close()

    def save_train_graph(self):
        epoch_train = []
        train_loss = []

        for l in self.log_train:
            epoch_train.append(l['epoch'])
            train_loss.append(l['loss'])

        epoch = np.asarray(epoch_train)
        train_loss = np.asarray(train_loss)
        
        o = plt.plot(epoch, train_loss, label='Train')
        plt.xlabel('Epoch')
        plt.ylabel('Loss/MSE')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.log_dir, 'lossgraph.png'))
        plt.close()