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

from base_model import BaseModel

class GBTModel(BaseModel):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.model = GradientBoostingClassifier()

    def train(self):
        # Implement or leave empty to override in derived classes
        # self.X = self.data[['Short_Moving_Avg', 'Long_Moving_Avg', 'RSI']]
        self.X = self.data[['Short_Moving_Avg', 'Long_Moving_Avg', 'RSI', 'DaysSincePeak', 'DaysSincePeakTrough']]

        self.y = self.data['Label']

        # Splitting the dataset and standardizing features
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        
        print("y_train value counts: ", self.y_train.value_counts())
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        smote = SMOTE()
        self.X_train_scaled, self.y_train = smote.fit_resample(self.X_train_scaled, self.y_train)

        self.model.fit(self.X_train_scaled, self.y_train)


    def predict(self):
        self.data['PredictedLabel'] = self.model.predict(self.scaler.transform(self.X))
        return self.data

    def evaluate(self, X, y):
        # Implement evaluation logic
        pass



