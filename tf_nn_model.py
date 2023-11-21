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

class TF_NN_Model(BaseModel):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.model = Sequential()
        self.label_encoder = LabelEncoder()

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

        # Convert labels to tensors and apply one-hot encoding
        y_train_encoded = self.label_encoder.fit_transform(self.y_train)
        y_test_encoded = self.label_encoder.transform(self.y_test)

        # Convert labels to categorical (one-hot encoding)
        y_train_categorical = to_categorical(y_train_encoded)
        y_test_categorical = to_categorical(y_test_encoded)

        # Neural Network architecture
        self.model.add(Dense(64, input_dim=self.X_train_scaled.shape[1], activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(y_train_categorical.shape[1], activation='softmax'))  # Output layer

        # Compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        self.model.fit(self.X_train_scaled, y_train_categorical, epochs=50, batch_size=10)



    def predict(self):
        print("CHECK TF_NN")
        predicted_probs = self.model.predict(self.X_test_scaled)
        # predicted_probs = self.model.predict(self.X)
        # Convert probabilities to class labels
        predicted_labels = np.argmax(predicted_probs, axis=1)

        # Assuming label_encoder was used to encode y_train
        predicted_categories = self.label_encoder.inverse_transform(predicted_labels)
        print("CHECK predicted_labels: ", predicted_categories)
        return predicted_categories

    def evaluate(self, X, y):
        # Implement evaluation logic
        pass



