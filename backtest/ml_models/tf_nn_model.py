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
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.model_selection import cross_val_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from keras import backend as K

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

import random

from .base_model import BaseModel

# Define a custom F1 score metric


def f1_score(y_true, y_pred):
    # Calculate precision and recall
    precision = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))
                      ) / K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / \
        K.sum(K.round(K.clip(y_true, 0, 1)))
    # Calculate F1 score
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


class F1ScoreThresholdCallback(Callback):
    def __init__(self, f1_threshold=0.9, loss_threshold=0.2, validation_data=()):
        super(F1ScoreThresholdCallback, self).__init__()
        self.f1_threshold = f1_threshold
        self.loss_threshold = loss_threshold
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs={}):
        # Check if F1 score in training logs exceeds the threshold
        # Assuming 'val_f1_score' is the key for F1 score in validation
        f1_score_epoch = logs.get('f1_score', 0)
        loss_epoch = logs.get('loss', 0)
        print("Epoch: {} - Validation F1 Score: {:.4f}".format(epoch+1, f1_score_epoch))
        print("Epoch: {} - Validation Loss: {:.4f}".format(epoch+1, loss_epoch))

        # Check if F1 score threshold is met or exceeded
        if f1_score_epoch > self.f1_threshold and loss_epoch < self.loss_threshold:
            print("F1 Score and Loss threshold reached, stopping training.")
            self.model.stop_training = True


class TF_NN_Model(BaseModel):
    def __init__(self, file_path, train_start, train_end, test_start, test_end):
        super().__init__(file_path, train_start, train_end, test_start, test_end)
        self.model = Sequential()
        self.label_encoder = LabelEncoder()

    def reshape_for_lstm(self, X, y, n_timesteps):
        # Reshape data to [samples, timesteps, features]
        samples, features = X.shape
        reshaped_X = np.zeros((samples - n_timesteps, n_timesteps, features))

        for i in range(n_timesteps, samples):
            reshaped_X[i - n_timesteps] = X[i - n_timesteps:i, :]

        # Adjust y to match the reshaped X
        reshaped_y = y[n_timesteps:]

        return reshaped_X, reshaped_y

    def train(self):
        self.train_test_split_time_series()

        print("y_train value counts: ", self.y_train.value_counts())
        n_timesteps = 10
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # self.X_train_scaled = self.reshape_for_lstm(self.X_train_scaled, n_timesteps)
        # self.X_test_scaled = self.reshape_for_lstm(self.X_test_scaled, n_timesteps)

        # self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        # self.X_test_scaled = self.scaler.transform(self.X_test)

        self.X_train_scaled, self.y_train = self.reshape_for_lstm(
            self.X_train_scaled, self.y_train, n_timesteps)
        self.X_test_scaled, self.y_test = self.reshape_for_lstm(
            self.X_test_scaled, self.y_test, n_timesteps)

        # Convert labels to tensors and apply one-hot encoding
        y_train_encoded = self.label_encoder.fit_transform(self.y_train)
        y_test_encoded = self.label_encoder.transform(self.y_test)

        # Convert labels to categorical (one-hot encoding)
        y_train_categorical = to_categorical(y_train_encoded)
        y_test_categorical = to_categorical(y_test_encoded)

        # # Neural Network architecture
        # self.model.add(Dense(64, input_dim=self.X_train_scaled.shape[1], activation='relu'))
        # self.model.add(Dense(32, activation='relu'))
        # self.model.add(Dense(y_train_categorical.shape[1], activation='softmax'))  # Output layer

        # self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
        # self.model.add(MaxPooling1D(pool_size=2))
        # self.model.add(Flatten())
        # self.model.add(LSTM(50, activation='relu'))
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(100, activation='relu'))
        # self.model.add(Dense(n_outputs, activation='softmax'))

        self.model.add(Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=(
            self.X_train_scaled.shape[1], self.X_train_scaled.shape[2])))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(LSTM(100))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(
            Dense(y_train_categorical.shape[1], activation='softmax'))

        # Compile the model
        # self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Now compile the model with the custom F1 score as a metric
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=[f1_score])

        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_encoded),
            y=y_train_encoded
        )
        class_weights_dict = dict(enumerate(class_weights))

        f1_score_callback = F1ScoreThresholdCallback(
            f1_threshold=0.8, loss_threshold=0.2, validation_data=(self.X_test_scaled, y_test_categorical))

        # Then pass these weights to the 'fit' method
        self.model.fit(
            self.X_train_scaled,
            y_train_categorical,
            epochs=100,
            batch_size=10,
            class_weight=class_weights_dict,
            callbacks=[f1_score_callback]
        )
        # # Train the model
        # self.model.fit(self.X_train_scaled, y_train_categorical, epochs=50, batch_size=10)

    def predict(self):
        print("CHECK TF_NN")
        predicted_probs = self.model.predict(self.X_test_scaled)
        # predicted_probs = self.model.predict(self.X)
        # Convert probabilities to class labels
        predicted_labels = np.argmax(predicted_probs, axis=1)

        # Assuming label_encoder was used to encode y_train
        self.predicted_categories = self.label_encoder.inverse_transform(
            predicted_labels)
        print("CHECK predicted_labels: ", self.predicted_categories)
        return self.predicted_categories

    def evaluate(self):
        # Implement evaluation logic
        # Assuming y_test is your true labels and predicted_labels is your model's predictions
        print("Confusion Matrix:\n", confusion_matrix(
            self.y_test, self.predicted_categories))
        print("\nClassification Report:\n", classification_report(
            self.y_test, self.predicted_categories, target_names=['Buy', 'Hold', 'Sell']))
