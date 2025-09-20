# src/ml/backtester.py

import pandas as pd
import numpy as np
import joblib
import os
from feature_engineering import create_features

def run_backtest():
    """
    Performs a rigorous, walk-forward backtest of the final trained model,
    fully compatible with the calibrated classifier and saved threshold.
    """
    print("--- Starting Trustworthy Backtest for Final Model ---")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # --- 1. Load Model, Threshold, and Data ---
    model_path = os.path.join(project_root, 'models', 'best_balanced_model.pkl')
    saved_model_dict = joblib.load(model_path)
    
    model = saved_model_dict['model']
    prediction_threshold = saved_model_dict['threshold']
    
    data_path = os.path.join(project_root, 'data', 'processed', 'xauusd_m1_processed.csv')
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
    
    # --- 2. Define Parameters for Trade Simulation ---
    train_size_ratio = 0.8
    atr_mult_tp = 2.0
    atr_mult_sl = 2.0
    time_barrier = 30

    # --- 3. Prepare Backtest Data ---
    train_size = int(len(df) * train_size_ratio)
    backtest_df = df.iloc[train_size:].copy()
    
    print("Generating features for the backtest period...")
    features_df = create_features(backtest_df.copy()) # Pass a copy to be safe
    
    # --- THIS IS THE FIX ---
    # Align the original price data with the generated features
    # This ensures we have the 'close', 'high', 'low' columns available for simulation
    aligned_backtest_df = backtest_df.loc[features_df.index]
    
    print(f"Backtesting on {len(aligned_backtest_df)} candles of unseen data.")

    # --- 4. Simulation ---
    trades = []
    
    # Get the model's probability predictions for the entire feature set
    predicted_probabilities = model.predict_proba(features_df)[:, 1]

    for i in range(len(aligned_backtest_df) - time_barrier):
        current_proba = predicted_probabilities[i]
        
        if current_proba > prediction_threshold:
            # Get entry price and ATR from the aligned original data
            entry_price = aligned_backtest_df['close'].iloc[i]
            atr_at_entry = features_df['atr'].iloc[i] # ATR is part of the feature set

            stop_loss = entry_price - (atr_at_entry * atr_mult_sl)
            take_profit = entry_price + (atr_at_entry * atr_mult_tp)
            
            # Look into the future using the original price data
            future_prices = aligned_backtest_df.iloc[i+1 : i+1+time_barrier]
            outcome = "Loss (Time)"

            for j in range(len(future_prices)):
                if future_prices['high'].iloc[j] >= take_profit:
                    outcome = "Win"
                    break
                if future_prices['low'].iloc[j] <= stop_loss:
                    outcome = "Loss"
                    break
            
            trades.append({"outcome": outcome})

    # --- 5. Performance Report ---
    if not trades:
        print("\nNo trades were executed with the current threshold.")
        return

    wins = len([t for t in trades if t['outcome'] == "Win"])
    total_trades = len(trades)
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    print("\n--- Backtest Performance Report ---")
    print(f"Prediction Threshold Used: {prediction_threshold * 100:.2f}%")
    print(f"Total Trades Executed: {total_trades}")
    print(f"Winning Trades: {wins}")
    print(f"Losing Trades: {total_trades - wins}")
    print(f"Win Rate: {win_rate:.2f}%")

if __name__ == '__main__':
    run_backtest()