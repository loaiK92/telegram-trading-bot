# src/ml/data_formatter.py

import pandas as pd
import os

def format_mt5_data(input_file_path, output_file_path):
    """
    Reads raw MT5 data, cleans it, and saves it in a processed format.
    """
    print(f"Starting formatting for: {input_file_path}")

    column_names = [
        "date", "time", "open", "high", "low", "close",
        "tick_volume", "volume", "spread"
    ]
    
    # CORRECTED: Added 'date' and 'time' to dtypes to prevent type errors.
    data_types = {
        "date": "str", 
        "time": "str",
        "open": "float64", "high": "float64", "low": "float64",
        "close": "float64", "tick_volume": "int64", "volume": "int64",
        "spread": "int32"
    }

    df = pd.read_csv(
        input_file_path,
        sep='\t',
        header=0,
        names=column_names,
        dtype=data_types
    )
    print(f"Successfully read {len(df)} rows.")

    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('timestamp', inplace=True)

    df_processed = df[['open', 'high', 'low', 'close', 'tick_volume']]

    df_processed.to_csv(output_file_path)
    print(f"Formatting complete. Processed file saved to: {output_file_path}")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'xauusd_m1_raw.csv')
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'xauusd_m1_processed.csv')

    format_mt5_data(raw_data_path, processed_data_path)