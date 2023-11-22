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

# Neural Network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, num_features, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.dropout = nn.Dropout(0.5)  # 50% dropout
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PT_NN_Model(BaseModel):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.label_encoder = LabelEncoder()

    def train(self):
        self.train_test_split_time_series()

        print("y_train value counts: ", self.y_train.value_counts())
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        epochs=50
        batch_size=10
        
        # Seed value
        seed = 42

        # Python random module
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)

        # If you are using CUDA (PyTorch with GPU)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

        # Additional configurations to enforce deterministic behavior
        # Note: This can impact performance on GPUs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Convert labels to tensors and apply one-hot encoding
        y_train_encoded = self.label_encoder.fit_transform(self.y_train)
        y_test_encoded = self.label_encoder.transform(self.y_test)

        y_train_categorical = torch.tensor(pd.get_dummies(y_train_encoded).values).float()
        y_test_categorical = torch.tensor(pd.get_dummies(y_test_encoded).values).float()

        # Convert features to tensors
        X_train_tensor = torch.tensor(self.X_train_scaled).float()
        X_test_tensor = torch.tensor(self.X_test_scaled).float()

        # Create Tensor Datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_categorical)
        test_dataset = TensorDataset(X_test_tensor, y_test_categorical)

        self.model = NeuralNetwork(self.X_train_scaled.shape[1], y_train_categorical.shape[1])

        # Assuming y_train_encoded contains encoded labels
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

        # # Pass the weights to the loss function
        # criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        self.criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001) # using stochastic gradient descent
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01) # using stochastic gradient descent

        # Calculate weights for each sample
        sample_weights = class_weights_tensor[y_train_encoded]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        # Use the sampler in the DataLoader
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler)
        # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        n_epochs_stop = 5  # number of epochs to stop after no improvement

        for epoch in range(epochs):
            self.model.train()  # Set the model to training mode
            for inputs, labels in train_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, torch.max(labels, 1)[1])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_loss = 0.0
                for inputs, labels in val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, torch.max(labels, 1)[1])
                    val_loss += loss.item()
                val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
            # Validation process...
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == n_epochs_stop:
                    print("Early stopping!")
                    break


    def predict(self):
        # Convert entire dataset to PyTorch tensor
        # X_tensor = torch.tensor(self.scaler.transform(self.X)).float()
        X_tensor = torch.tensor(self.scaler.transform(self.X_test_scaled)).float()

        # Predict with the PyTorch model
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted_labels = torch.max(outputs, 1)
            print("predicted_labels: ", predicted_labels)
            predicted_categories = self.label_encoder.inverse_transform(predicted_labels.numpy())
            print("Predicted categories:", predicted_categories)

        # Convert predicted labels to DataFrame for easy integration
        predicted_labels_df = pd.DataFrame(predicted_categories, columns=['PredictedLabel'])

        # # Merge predicted labels with your existing dataset
        # self.data['PredictedLabel'] = predicted_labels_df

        # return self.data
        return predicted_labels_df

    def evaluate(self, X, y):
        # Implement evaluation logic
        pass



