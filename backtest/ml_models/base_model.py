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

        self.split_idx = None

        self.model = None
        self.scaler = StandardScaler()

        self.criterion = None
        self.optimizer = None

        self.fft_features = None

        self.predicted_categories = None


    def load_preprocess_data(self):
        self.data = pd.read_csv(self.file_path)
        self.perform_fourier_transform_analysis()
        self.calculate_stochastic_oscillator()
        self.detect_peaks_and_troughs()
        self.calculate_moving_averages_and_rsi()
        self.calculate_days_since_peaks_and_troughs()
        self.preprocess_data()

    def perform_fourier_transform_analysis(self):
        # Fourier Transform Analysis
        close_prices = self.data['Close'].to_numpy()
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
        self.fft_features = pd.DataFrame({
            'Frequency': significant_frequencies,
            'Amplitude': significant_amplitudes,
            'DaysPerCycle': days_per_cycle
        })

    def calculate_stochastic_oscillator(self, k_window=14, d_window=3):
        """
        Calculate the Stochastic Oscillator.
        %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
        %D = 3-day SMA of %K

        Where:
        - Lowest Low = lowest low for the look-back period
        - Highest High = highest high for the look-back period
        - %K is multiplied by 100 to move the decimal point two places

        The result is two time series: %K and %D
        """
        # Calculate %K
        low_min  = self.data['Low'].rolling(window=k_window).min()
        high_max = self.data['High'].rolling(window=k_window).max()
        self.data['%K'] = 100 * (self.data['Close'] - low_min) / (high_max - low_min)
        
        # Calculate %D as the moving average of %K
        self.data['%D'] = self.data['%K'].rolling(window=d_window).mean()

        # Handle any NaN values that may have been created
        self.data['%K'].fillna(method='bfill', inplace=True)
        self.data['%D'].fillna(method='bfill', inplace=True)


    def detect_peaks_and_troughs(self):
        # Peak and Trough Detection for Labeling
        close_prices = self.data['Close'].to_numpy()
        peaks, _     = find_peaks(close_prices)
        troughs, _   = find_peaks(-1 * close_prices)
        mid_trend    = [i for i in range(len(close_prices))]
        for peak in peaks:
            mid_trend.remove(peak)
        for trough in troughs:
            mid_trend.remove(trough)
        labels = pd.DataFrame(index=range(len(close_prices)), columns=['Label'])
        labels.loc[peaks, 'Label'] = 'Sell'
        labels.loc[troughs, 'Label'] = 'Buy'
        labels.loc[mid_trend, 'Label'] = 'Hold'
        self.data = self.data.join(labels)

    # Calculating Moving Averages and RSI manually
    def calculate_rsi(self, window=14):
        """ Calculate the Relative Strength Index (RSI) for a given dataset and window """
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_moving_averages_and_rsi(self):
        short_window = 5
        long_window = 20
        rsi_period = 14
        self.data['Short_Moving_Avg'] = self.data['Close'].rolling(window=short_window).mean()
        self.data['Long_Moving_Avg'] = self.data['Close'].rolling(window=long_window).mean()
        self.data['RSI'] = self.calculate_rsi(window=rsi_period)

    def calculate_days_since_peaks_and_troughs(self):
        self.data['DaysSincePeak'] = 0
        self.data['DaysSinceTrough'] = 0
        self.data['PriceChangeSincePeak'] = 0
        self.data['PriceChangeSinceTrough'] = 0

        checkpoint_date_bottom = None  # Initialize to a sensible default or first date
        checkpoint_date_top = None  # Initialize to a sensible default or first date
        checkpoint_price_bottom = None
        checkpoint_price_top = None
        price_change_since_bottom = 0
        price_change_since_peak = 0

        for index, row in self.data.iterrows():
            current_price = row['Open']
            today_date = pd.to_datetime(row['Date'])

            if row['Label'] == 'Buy':
                checkpoint_date_bottom = today_date
                checkpoint_price_bottom = current_price
            if row['Label'] == 'Sell':
                checkpoint_date_top = today_date
                checkpoint_price_top = current_price

            days_since_bottom = (today_date - checkpoint_date_bottom).days if checkpoint_date_bottom else 0
            days_since_peak = (today_date - checkpoint_date_top).days if checkpoint_date_top else 0

            if checkpoint_price_bottom is not None:
                price_change_since_bottom = current_price - checkpoint_price_bottom
            if checkpoint_price_top is not None:
                price_change_since_peak = current_price - checkpoint_price_top

            # final_dataset_with_new_features.at[index, 'DaysSincePeakTrough'] = max(days_since_bottom, days_since_peak)
            self.data.at[index, 'DaysSincePeak'] = days_since_peak
            self.data.at[index, 'DaysSinceTrough'] = days_since_bottom
            self.data.at[index, 'PriceChangeSincePeak'] = price_change_since_peak
            self.data.at[index, 'PriceChangeSinceTrough'] = price_change_since_bottom

    def preprocess_data(self):
        self.data.dropna(inplace=True)

    def train_test_split_time_series(self, test_size=0.4):
        # Convert 'Date' column to datetime if it's not already
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Sort the data by date to ensure correct time sequence
        self.data.sort_values('Date', inplace=True)
        
        # Find the split point
        self.split_idx = int(len(self.data) * (1 - test_size))
        
        # Split the data without shuffling
        self.X_train = self.data[['Short_Moving_Avg', 'Long_Moving_Avg', 'RSI', 'DaysSincePeak', 'DaysSinceTrough', 'PriceChangeSincePeak', 'PriceChangeSinceTrough', '%K', '%D']].iloc[:self.split_idx]
        self.X_test = self.data[['Short_Moving_Avg', 'Long_Moving_Avg', 'RSI', 'DaysSincePeak', 'DaysSinceTrough', 'PriceChangeSincePeak', 'PriceChangeSinceTrough', '%K', '%D']].iloc[self.split_idx:]
        # self.X_train = self.data[['Short_Moving_Avg', 'Long_Moving_Avg', 'RSI', 'DaysSincePeak', 'DaysSinceTrough']].iloc[:self.split_idx]
        # self.X_test = self.data[['Short_Moving_Avg', 'Long_Moving_Avg', 'RSI', 'DaysSincePeak', 'DaysSinceTrough']].iloc[self.split_idx:]
        self.y_train = self.data['Label'].iloc[:self.split_idx]
        self.y_test = self.data['Label'].iloc[self.split_idx:]

        print("len X train: ", len(self.X_train))
        print("len X test: ", len(self.X_test))
        print("len y train: ", len(self.y_train))
        print("len y test: ", len(self.y_test))

    def retrieve_test_set(self):
        return self.data[self.split_idx:]

    def train(self):
        # Implement or leave empty to override in derived classes
        pass

    def predict(self):
        # Implement or leave empty to override in derived classes
        pass

    def evaluate(self, X, y):
        # Implement evaluation logic
        pass
