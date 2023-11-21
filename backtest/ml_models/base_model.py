import pandas as pd
from scipy.fft import fft
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

class BaseModel:
    def __init__(self, file_path):
        self.file_path = file_path
        # self.model_type = model_type
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

        self.model = None
        self.scaler = StandardScaler()

        self.criterion = None
        self.optimizer = None
            

    def load_preprocess_data(self):
        # Implement data loading logic
        forex_data = pd.read_csv(self.file_path)

        # Fourier Transform Analysis
        close_prices = forex_data['Close'].to_numpy()
        N = len(close_prices)
        T = 1.0  # 1 day
        close_fft = fft(close_prices)
        fft_freq = np.fft.fftfreq(N, T)
        positive_frequencies = fft_freq[:N//2]
        positive_fft_values = 2.0/N * np.abs(close_fft[0:N//2])
        amplitude_threshold = 0.1  # This can be adjusted
        significant_peaks, _ = find_peaks(positive_fft_values, height=amplitude_threshold)
        significant_frequencies = positive_frequencies[significant_peaks]
        significant_amplitudes = positive_fft_values[significant_peaks]
        days_per_cycle = 1 / significant_frequencies
        fft_features = pd.DataFrame({
            'Frequency': significant_frequencies,
            'Amplitude': significant_amplitudes,
            'DaysPerCycle': days_per_cycle
        })

        # Peak and Trough Detection for Labeling
        peaks, _ = find_peaks(close_prices)
        troughs, _ = find_peaks(-1 * close_prices)
        mid_trend = [i for i in range(len(close_prices))]
        print("check mid_trend: ", mid_trend)
        for peak in peaks:
            mid_trend.remove(peak)
        for trough in troughs:
            mid_trend.remove(trough)
        print("check mid_trend1: ", mid_trend)
        labels = pd.DataFrame(index=range(len(close_prices)), columns=['Label'])
        print("labels: ", labels)
        for label in labels:
            print("c: ", label)
        labels.loc[peaks, 'Label'] = 'Sell'
        labels.loc[troughs, 'Label'] = 'Buy'
        labels.loc[mid_trend, 'Label'] = 'Hold'

        print("fft_features: ", fft_features)
        print("forex_data: ", forex_data)
        print("labels: ", labels)
        # Merging the FFT features and labels with the Forex data
        # final_dataset = forex_data.join(fft_features).join(labels)
        final_dataset = forex_data.join(labels)
        print("check final_dataset: ", final_dataset)

        print("***")
        print("\n")

        # Calculating Moving Averages and RSI manually
        def calculate_rsi(data, window=14):
            """ Calculate the Relative Strength Index (RSI) for a given dataset and window """
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        short_window = 5
        long_window = 20
        rsi_period = 14
        forex_data['Short_Moving_Avg'] = forex_data['Close'].rolling(window=short_window).mean()
        forex_data['Long_Moving_Avg'] = forex_data['Close'].rolling(window=long_window).mean()
        forex_data['RSI'] = calculate_rsi(forex_data, window=rsi_period)
        print("forex_data1: ", forex_data)
        # Merging the new features into the final dataset
        final_dataset_with_new_features = final_dataset.join(forex_data[['Short_Moving_Avg', 'Long_Moving_Avg', 'RSI']])
        print("check final_dataset_with_new_features: ", final_dataset_with_new_features)

        # Data Preprocessing
        final_dataset_with_new_features.dropna(inplace=True)

        checkpoint_date_bottom = None  # Initialize to a sensible default or first date
        checkpoint_date_top = None  # Initialize to a sensible default or first date

        for index, row in final_dataset_with_new_features.iterrows():
            print("index: ", index, "row: ", row)
            current_price = row['Open']

            today_date = pd.to_datetime(row['Date'])

            if row['Label'] == 'Buy':
                checkpoint_date_bottom = today_date
                print("CHECK UPDATE local_trough checkpoint_date_bottom: ", today_date)
            if row['Label'] == 'Sell':
                checkpoint_date_top = today_date
                print("CHECK UPDATE local_peak checkpoint_date_top: ", today_date)

            days_since_bottom = (today_date - checkpoint_date_bottom).days if checkpoint_date_bottom else 0
            print("days_since_bottom: ", days_since_bottom)
            days_since_peak = (today_date - checkpoint_date_top).days if checkpoint_date_top else 0
            print("days_since_peak: ", days_since_peak)

            # final_dataset_with_new_features.at[index, 'DaysSincePeakTrough'] = max(days_since_bottom, days_since_peak)
            final_dataset_with_new_features.at[index, 'DaysSincePeak'] = days_since_peak
            final_dataset_with_new_features.at[index, 'DaysSincePeakTrough'] = days_since_bottom

        self.data = final_dataset_with_new_features

        return self.data

    def train(self):
        # Implement or leave empty to override in derived classes
        pass

    def predict(self):
        # Implement or leave empty to override in derived classes
        pass

    def evaluate(self, X, y):
        # Implement evaluation logic
        pass
