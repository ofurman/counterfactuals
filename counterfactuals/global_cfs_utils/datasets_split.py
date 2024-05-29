import numpy as np
import os
from os import path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import urllib.request
import datetime

from counterfactuals.global_cfs_utils.datasets import dataset_loader

class dataset_loader_split(dataset_loader):
    def __init__(self,  name=None,  data_path="./datasets/", dropped_features=[], n_bins=None):
        super().__init__(name=name,  data_path=data_path, dropped_features=dropped_features, n_bins=n_bins)
        
        self.split_params = {
            "compas": dict(random_state=4, test_size=0.2, shuffle=True),
            "german_credit": dict(random_state=4, test_size=0.2, shuffle=True),
            "adult_income": dict(random_state=42, test_size=0.1, shuffle=True),
            "heloc": dict(random_state=4, test_size=0.1, shuffle=True)
        }

    def get_split(self, normalise=True, shuffle=False,
                  return_mean_std=False, print_outputs=False):
        if shuffle:
            self.data = self.data.sample(frac=1)
        data = self.data.values

        X = data[:,:-1]
        y = data[:, -1]

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, stratify=y, **self.split_params[self.name]
        )

        if print_outputs:
            print("\033[1mProportion of 1s in Training Data:\033[0m {}%"\
                  .format(round(np.average(y_train)*100, 2)))
            print("\033[1mProportion of 1s in Test Data:\033[0m {}%"\
                  .format(round(np.average(y_test)*100, 2)))
        
        x_train = x_train.astype(np.int64)
        x_means, x_stds = x_train.mean(axis=0), x_train.std(axis=0)
        
        if normalise:
            x_train = (x_train - x_means)/x_stds
            x_test = (x_test - x_means)/x_stds
        
        if return_mean_std:
            return x_train, y_train, x_test, y_test, x_means, x_stds
        return x_train, y_train, x_test, y_test
