import pandas as pd
from scipy.fft import fft
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from pykalman import KalmanFilter
from hurst import compute_Hc


class BaseModel:
    def __init__(self, file_path, train_start, train_end, test_start, test_end):
        self.file_path = file_path
        # self.model_type = model_type
        self.data = None

        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

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
        self.construct_kalman_filter()
        self.detect_peaks_and_troughs()

        self.calculate_moving_averages_and_rsi()
        self.estimate_hurst_exponent()
        self.calculate_days_since_peaks_and_troughs()
        self.detect_fourier_signals()
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
        significant_peaks, _ = find_peaks(
            positive_fft_values, height=amplitude_threshold)
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
        low_min = self.data['Low'].rolling(window=k_window).min()
        high_max = self.data['High'].rolling(window=k_window).max()
        self.data['%K'] = 100 * \
            (self.data['Close'] - low_min) / (high_max - low_min)

        # Calculate %D as the moving average of %K
        self.data['%D'] = self.data['%K'].rolling(window=d_window).mean()

        # Handle any NaN values that may have been created
        self.data['%K'].fillna(method='bfill', inplace=True)
        self.data['%D'].fillna(method='bfill', inplace=True)

    def detect_fourier_signals(self):
        # Add in fourier transform
        dominant_period_lengths = sorted(set(
            (self.fft_features.loc[:10, 'DaysPerCycle'].values/2).astype(int)), reverse=True)[:5]
        print("check dominant_period_lengths: ", dominant_period_lengths)
        self.data['FourierSignalSell'] = self.data['DaysSinceTrough'].isin(
            dominant_period_lengths)
        self.data['FourierSignalBuy'] = self.data['DaysSincePeak'].isin(
            dominant_period_lengths)
        # self.data.at[index, 'DaysSincePeak'] = days_since_peak
        # self.data.at[index, 'DaysSinceTrough'] = days_since_bottom
        print("FourierSignalSell: ", self.data['FourierSignalSell'])
        print("FourierSignalBuy: ", self.data['FourierSignalBuy'])

    def detect_peaks_and_troughs(self):
        # Peak and Trough Detection for Labeling
        close_prices = self.data['Close'].to_numpy()
        peaks, _ = find_peaks(close_prices)
        troughs, _ = find_peaks(-1 * close_prices)
        mid_trend = [i for i in range(len(close_prices))]
        for peak in peaks:
            mid_trend.remove(peak)
        for trough in troughs:
            mid_trend.remove(trough)
        labels = pd.DataFrame(index=range(
            len(close_prices)), columns=['Label'])
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
        self.data['Short_Moving_Avg'] = self.data['Close'].rolling(
            window=short_window).mean()
        self.data['Long_Moving_Avg'] = self.data['Close'].rolling(
            window=long_window).mean()
        self.data['RSI'] = self.calculate_rsi(window=rsi_period)

    def construct_kalman_filter(self):
        close_prices = self.data['Close']
        # Construct a Kalman Filter
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

        # Use the observed data (close prices) to estimate the state
        state_means, _ = kf.filter(close_prices.values)

        # Convert state means to a Pandas Series for easy plotting
        kalman_estimates = pd.Series(
            state_means.flatten(), index=self.data.index)

        # Combine the original close prices and Kalman Filter estimates
        kalman_estimates = pd.DataFrame({
            'KalmanFilterEst': kalman_estimates
        })
        self.data = self.data.join(kalman_estimates)

        # print('KalmanFilterEst: ', self.data['KalmanFilterEst'])  # Display the first few rows of the dataframe

    # Function to estimate Hurst exponent over multiple sliding windows
    def estimate_hurst_exponent(self, window_size=100, step_size=1):
        """
        Calculates the Hurst exponent over sliding windows and appends the results to the DataFrame.

        :param window_size: Size of each sliding window.
        :param step_size: Step size for moving the window.
        """
        # Prepare an empty list to store Hurst exponent results and corresponding dates
        hurst_exponents = []

        for i in range(0, len(self.data) - window_size + 1, step_size):
            # Extract the window
            window = self.data['Close'].iloc[i:i + window_size]

            # Calculate the Hurst exponent
            H, _, _ = compute_Hc(window, kind='price', simplified=True)

            # Store the result along with the starting date of the window
            hurst_exponents.append(
                {'Date': self.data['Date'].iloc[i], 'HurstExponent': H})

        # Convert the results list into a DataFrame
        hurst_exponents = pd.DataFrame(hurst_exponents)

        # Combine the original DataFrame and Hurst exponent estimates
        self.data = self.data.merge(hurst_exponents, on='Date', how='left')

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

            days_since_bottom = (
                today_date - checkpoint_date_bottom).days if checkpoint_date_bottom else 0
            days_since_peak = (
                today_date - checkpoint_date_top).days if checkpoint_date_top else 0

            if checkpoint_price_bottom is not None:
                price_change_since_bottom = current_price - checkpoint_price_bottom
            if checkpoint_price_top is not None:
                price_change_since_peak = current_price - checkpoint_price_top

            # final_dataset_with_new_features.at[index, 'DaysSincePeakTrough'] = max(days_since_bottom, days_since_peak)
            self.data.at[index, 'DaysSincePeak'] = days_since_peak
            self.data.at[index, 'DaysSinceTrough'] = days_since_bottom
            # self.data.at[index, 'FourierSignalSell'] = ( (days_since_peak % 6 == 0) or (days_since_peak % 7 == 0) )
            # self.data.at[index, 'FourierSignalBuy'] = ( (days_since_bottom % 6 == 0) or (days_since_bottom % 7 == 0) )
            self.data.at[index,
                         'PriceChangeSincePeak'] = price_change_since_peak
            self.data.at[index,
                         'PriceChangeSinceTrough'] = price_change_since_bottom

    def preprocess_data(self):
        self.data.dropna(inplace=True)

    # def train_test_split_time_series(self, test_size=0.4):
    def train_test_split_time_series(self):
        # Convert 'Date' column to datetime if it's not already
        self.data['Date'] = pd.to_datetime(self.data['Date'])

        # # Filter the data for training and testing periods
        # train_data = self.data[(self.data['Date'] >= self.train_start) & (self.data['Date'] < self.train_end)]
        # test_data = self.data[(self.data['Date'] >= self.test_start) & (self.data['Date'] < self.test_end)]


        # Sort the data by date to ensure correct time sequence
        self.data.sort_values('Date', inplace=True)

        # Find the split point
        # self.split_idx = int(len(self.data) * (1 - test_size))
        self.data.to_csv('final_dataset_with_new_features.csv')
        # Split the data without shuffling

        # Filter the data for training and testing periods
        self.train_data = self.data[(self.data['Date'] >= self.train_start) & (self.data['Date'] < self.train_end)]
        self.test_data  = self.data[(self.data['Date'] >= self.test_start) & (self.data['Date'] < self.test_end)]

        self.X_train = self.train_data[
            ['Short_Moving_Avg',
             'Long_Moving_Avg',
             'RSI',
             'DaysSincePeak',
             'DaysSinceTrough',
             'FourierSignalSell',
             'FourierSignalBuy',
             '%K',
             '%D',
             'KalmanFilterEst',
             'HurstExponent']
        ]
        
        # ].iloc[:self.split_idx]
        self.X_test = self.test_data[
            ['Short_Moving_Avg',
             'Long_Moving_Avg',
             'RSI',
             'DaysSincePeak',
             'DaysSinceTrough',
             'FourierSignalSell',
             'FourierSignalBuy',
             '%K',
             '%D',
             'KalmanFilterEst',
             'HurstExponent']
        ]
        # ].iloc[self.split_idx:]

        # self.y_train = self.data['Label'].iloc[:self.split_idx]
        # self.y_test = self.data['Label'].iloc[self.split_idx:]
        self.y_train = self.train_data['Label']
        self.y_test = self.test_data['Label']

        print("len X train: ", len(self.X_train))
        print("len X test: ", len(self.X_test))
        print("len y train: ", len(self.y_train))
        print("len y test: ", len(self.y_test))

    def retrieve_test_set(self):
        # return self.data[self.split_idx:]
        # dates = self.test_data[['Date']]
        # print("dates111: ", dates)
        # return self.X_test.join( dates )
        return self.test_data

    def train(self):
        # Implement or leave empty to override in derived classes
        pass

    def predict(self):
        # Implement or leave empty to override in derived classes
        pass

    def evaluate(self, X, y):
        # Implement evaluation logic
        pass
