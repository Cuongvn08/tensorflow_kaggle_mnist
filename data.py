# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import pandas as pd


class Data():
    def __init__(self):
        self.train_data     = []
        self.train_label    = []

        self.eval_data      = []
        self.eval_label     = []

        self.test_data      = []
        self.test_label     = []

    def read_train_data(self, data_path, train_ratio=0.8):
        if train_ratio > 1 or train_ratio < 0:
            assert False, '[data] train_ratio is not valid !!!'

        # load data
        df = pd.read_csv(data_path)
        data = df.values[:,1:].reshape([-1, 28, 28, 1])
        label = df.values[:,0]

        # split indices
        num_data = data.shape[0]
        num_train = np.int(num_data * train_ratio)
        num_eval = num_data - num_train

        train_indices = random.sample(range(num_data), num_train)
        eval_indices = [i for i in range(num_data) if i not in train_indices]

        # split train data
        for i in train_indices:
            self.train_data.append(data[i])
            self.train_label.append(label[i])
        self.train_data = np.asarray(self.train_data)
        self.train_label = np.asarray(self.train_label)

        print('[data] shape of train data = {0}'.format(self.train_data.shape))
        print('[data] shape of train label = {0}'.format(self.train_label.shape))

        # split eval data
        for i in eval_indices:
            self.eval_data.append(data[i])
            self.eval_label.append(label[i])
        self.eval_data = np.asarray(self.eval_data)
        self.eval_label = np.asarray(self.eval_label)

        print('[data] shape of eval data = {0}'.format(self.eval_data.shape))
        print('[data] shape of eval label = {0}'.format(self.eval_label.shape))

    def read_test_data(self, data_path):
        df = pd.read_csv(data_path)

        self.test_data = df.values[:,:].reshape([-1, 28, 28, 1])
        self.test_label = None

        print('[data] shape of test data = {0}'.format(self.test_data.shape))
        print('[data] shape of test label = None')

    def get_train_data(self):
        return self.train_data
    
    def get_train_label(self):
        return self.train_label
    
    def get_eval_data(self):
        return self.eval_data
    
    def get_eval_label(self):
        return self.eval_label
    
    def get_test_data(self):
        return self.test_data
    
    def get_test_label(self):
        return self.test_label
