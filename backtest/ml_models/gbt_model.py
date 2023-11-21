import pandas as pd
from scipy.fft import fft
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

import random

from .base_model import BaseModel

class GBTModel(BaseModel):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.model = GradientBoostingClassifier()

    def train(self):
        self.train_test_split_time_series()

        print("y_train value counts: ", self.y_train.value_counts())
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # smote = SMOTE()
        # self.X_train_scaled, self.y_train = smote.fit_resample(self.X_train_scaled, self.y_train)

        self.model.fit(self.X_train_scaled, self.y_train)


    def predict(self):
        # self.data['PredictedLabel'] = self.model.predict(self.scaler.transform(self.X))
        # return self.data
        predicted_categories = self.model.predict(self.scaler.transform(self.X_test_scaled))
        print("CHECK predicted_labels: ", predicted_categories)
        return predicted_categories

    def evaluate(self, X, y):
        # Implement evaluation logic
        pass



