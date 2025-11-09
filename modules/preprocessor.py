import pandas as pd
import pandas_ta as ta

def clean_data(df_raw):
    """
    Cleans the raw DataFrame (Polygon format) and prepares it for analysis.
    """
    if df_raw.empty:
        return pd.DataFrame()
        
    df = df_raw.copy()
    
    # 1. Rename columns from Polygon/simulated format
    df.rename(columns={
        't': 'timestamp',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume'
    }, inplace=True)
    
    # 2. Convert timestamp (milliseconds) to Datetime
    # errors='coerce' handles any invalid timestamps without crashing
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    
    # 3. Handle missing or invalid data
    df.dropna(subset=['timestamp', 'close'], inplace=True)
    
    # 4. Set timestamp as the index (essential for pandas-ta)
    df.set_index('timestamp', inplace=True)
    
    return df


def create_technical_features(df_clean):
    """
    Adds all technical indicators.
    (Version with MANUAL BOLLINGER BANDS calculation)
    """
    if df_clean.empty:
        return df_clean
        
    base_df = df_clean.copy()
    
    # --- MANUAL BOLLINGER BANDS CALCULATION ---
    # This uses pure pandas and cannot fail.
    
    # 1. 20-period Simple Moving Average (Middle Band)
    window_size = 20
    base_df['BBM_20_2.0'] = base_df['close'].rolling(window=window_size).mean()
    
    # 2. 20-period Standard Deviation
    std_dev = base_df['close'].rolling(window=window_size).std()
    
    # 3. Upper Band (Mean + 2*STD)
    base_df['BBU_20_2.0'] = base_df['BBM_20_2.0'] + (std_dev * 2)
    
    # 4. Lower Band (Mean - 2*STD)
    base_df['BBL_20_2.0'] = base_df['BBM_20_2.0'] - (std_dev * 2)
    
    # --- END OF MANUAL CALCULATION ---
    
    # List for other indicators (which work fine)
    indicators_list = []
    try:
        indicators_list.append(base_df.ta.ema(length=14))
        indicators_list.append(base_df.ta.ema(length=50))
        indicators_list.append(base_df.ta.sma(length=200))
        indicators_list.append(base_df.ta.rsi(length=14))
        indicators_list.append(base_df.ta.macd()) # Generates MACD, MACDh, MACDs
        indicators_list.append(base_df.ta.atr(length=14)) # Generates 'ATRr_14'
        
    except Exception as e:
        print(f"Error while calculating other pandas-ta indicators: {e}")

    # Filter out any None results (if a calculation failed)
    valid_indicators = [indf for indf in indicators_list if indf is not None and not indf.empty]
    
    # Concatenate all indicator DataFrames
    if valid_indicators:
        # Use 'base_df' which already has the manual BBands
        all_features_df = pd.concat([base_df] + valid_indicators, axis=1)
    else:
        all_features_df = base_df # If all else fails, at least return manual BBands

    # 5. Fix NaNs created by indicators
    all_features_df.ffill(inplace=True) 
    
    return all_features_df

def prepare_data_for_model(df_features):
    """
    Takes the full feature DataFrame and prepares the *last row*
    in the exact format the model expects.
    """
    if df_features.empty:
        return None

    # This MUST match the list in 'training/train_models.py'
    FEATURE_COLUMNS = [
        'open', 'high', 'low', 'close', 'volume',
        'EMA_14', 'EMA_50', 'SMA_200', 'RSI_14',
        'MACD_12_26_9', 
        'ATRr_14' # The name pandas-ta creates
    ]
    
    # Ensure all required columns exist
    available_features = [col for col in FEATURE_COLUMNS if col in df_features.columns]
    
    if len(available_features) == 0:
        print("Warning: No model features were found in the dataframe.")
        return None

    # Select only the last row and available features
    latest_data = df_features.iloc[[-1]][available_features]
    
    # Handle any remaining NaNs
    latest_data.fillna(0, inplace=True)
    
    return latest_data