# src/ml/predict.py

import pandas as pd
import joblib
import os
from .feature_engineering import create_features

class SignalGenerator:
    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # --- Load the final model and its parameters ---
        model_path = os.path.join(project_root, 'models', 'best_balanced_model.pkl')
        saved_model_dict = joblib.load(model_path)
        
        self.model = saved_model_dict['model']
        self.threshold = saved_model_dict['threshold']
        
        data_path = os.path.join(project_root, 'data', 'processed', 'xauusd_m1_processed.csv')
        # Load enough data to initialize long-period indicators
        self.live_data_buffer = pd.read_csv(data_path, index_col='timestamp', parse_dates=True).tail(5000)
        
        print(f"Production model loaded. Using probability threshold: {self.threshold:.3f}")

    def update_tick_data(self, tick):
        """
        Updates the live data buffer with a new tick and resamples to 1-minute candles.
        """
        tick_time = pd.to_datetime(int(tick['timestamp']), unit='ms')
        price = float(tick['price'])
        
        # Create a new row for the tick
        new_row = pd.DataFrame([{'open': price, 'high': price, 'low': price, 'close': price, 'volume': 0, 'tick_volume': 0}], index=[tick_time])
        
        # Append and resample to maintain a clean 1-minute OHLCV dataframe
        self.live_data_buffer = pd.concat([self.live_data_buffer, new_row])
        self.live_data_buffer = self.live_data_buffer[~self.live_data_buffer.index.duplicated(keep='last')]
        self.live_data_buffer = self.live_data_buffer.resample('1min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'tick_volume': 'sum'
        }).ffill()
        
        # Keep the buffer from growing indefinitely
        self.live_data_buffer = self.live_data_buffer.tail(5000)

    def get_signal(self):
        """
        Generates a signal using the final, calibrated model and optimal threshold.
        """
        # 1. Generate the exact same features your model was trained on
        df_with_features = create_features(self.live_data_buffer.copy())
        
        # Ensure the feature DataFrame is not empty
        if df_with_features.empty:
            return "HOLD", 0.0

        latest_features = df_with_features.tail(1)
        
        # 2. Get the calibrated probability of a "Buy" event
        buy_probability = self.model.predict_proba(latest_features)[:, 1][0]
        
        # 3. Apply the optimal threshold
        signal = "HOLD"
        if buy_probability > self.threshold:
            signal = "BUY"
            
        # The confidence is the model's actual probability score
        confidence = buy_probability * 100

        return signal, confidence