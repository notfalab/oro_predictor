import streamlit as st
import pandas as pd
import numpy as np
import datetime
from polygon import RESTClient
from fredapi import Fred

# --- 1. LOAD API KEYS ---
try:
    POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except (KeyError, FileNotFoundError):
    print("WARNING: .streamlit/secrets.toml not found. Using placeholders.")
    POLYGON_API_KEY = "YOUR_POLYGON_KEY_IS_MISSING"
    FRED_API_KEY = "YOUR_FRED_KEY_IS_MISSING"

# --- 2. INITIALIZE API CLIENTS ---
try: 
    polygon_client = RESTClient(POLYGON_API_KEY)
except Exception as e: 
    print(f"Error initializing Polygon client (invalid key?): {e}")
    polygon_client = None

try: 
    fred_client = Fred(api_key=FRED_API_KEY)
except Exception as e:
    print(f"Error initializing FRED client (invalid key?): {e}")
    fred_client = None

SYMBOL = "C:XAUUSD"

# --- 3. REAL DATA FUNCTIONS (Bug-fixed) ---
def _process_polygon_aggs(aggs):
    """Internal helper to correctly process Polygon API responses."""
    if not aggs: 
        return pd.DataFrame()
    
    df = pd.DataFrame(aggs)
    
    # 1. Rename ALL Polygon fields to our internal format
    df.rename(columns={
        'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c',
        'volume': 'v', 'timestamp': 't'
    }, inplace=True)
    
    # 2. Sort by 't' (timestamp)
    df.sort_values(by='t', ascending=True, inplace=True)
    
    # 3. Return only the columns we need
    return df[['t', 'o', 'h', 'l', 'c', 'v']]

def get_initial_historical_data(symbol=SYMBOL, limit=500):
    """Fetches initial historical data."""
    print("Fetching REAL historical data from Polygon...")
    if polygon_client is None: 
        return _emergency_mock_data(periods=limit)
    try:
        today = datetime.date.today()
        aggs = polygon_client.get_aggs(
            symbol, 1, "minute",
            (today - datetime.timedelta(days=3)).isoformat(), # 3 days back for weekend-proofing
            today.isoformat(),
            limit=limit, sort="desc"
        )
        df = _process_polygon_aggs(aggs)
        if df.empty: 
            return _emergency_mock_data(periods=limit)
        return df
    except Exception as e:
        print(f"Error contacting Polygon API (get_initial): {e}")
        return _emergency_mock_data(periods=limit)

def fetch_latest_data(current_data_df):
    """Fetches the latest 1-minute candle."""
    print("Fetching REAL latest tick from Polygon...")
    if polygon_client is None: 
        return current_data_df
    try:
        today = datetime.date.today()
        aggs = polygon_client.get_aggs(
            SYMBOL, 1, "minute",
            (today - datetime.timedelta(days=1)).isoformat(),
            today.isoformat(),
            limit=1, sort="desc"
        )
        new_tick_df = _process_polygon_aggs(aggs)
        
        # If no new data or data is old, return the old dataframe
        if new_tick_df.empty or new_tick_df['t'].iloc[0] in current_data_df['t'].values:
            return current_data_df
            
        updated_df = pd.concat([current_data_df, new_tick_df], ignore_index=True)
        return updated_df.iloc[1:] # Remove oldest row
    except Exception as e:
        print(f"Error contacting Polygon API (fetch_latest): {e}")
        return current_data_df # Return old data on failure

def get_macro_and_sentiment():
    """Fetches REAL macro data from FRED."""
    print("Fetching REAL macro data from FRED...")
    if fred_client is None: 
        return _emergency_mock_macro()
    
    try:
        try: fed_rate = fred_client.get_series('FEDFUNDS').iloc[-1]
        except Exception: fed_rate = 5.25 # Fallback
        
        try: dxy_index = fred_client.get_series('TWEXBPA').iloc[-1] # Daily Dollar Index
        except Exception: dxy_index = 105.0 # Fallback
        
        try: inflation_cpi = fred_client.get_series('CPIAUCSL_PC1').iloc[-1] # Yearly CPI % change
        except Exception: inflation_cpi = 3.5 # Fallback

        real_yield = fed_rate - inflation_cpi
        
        # Fundamental Verdict (in English)
        if real_yield > 1.0: 
            fundamental_verdict = "Bearish (Positive Real Yields)"
        elif real_yield < 0: 
            fundamental_verdict = "Bullish (Negative Real Yields)"
        else: 
            fundamental_verdict = "Neutral (Low Real Yields)"
            
        return {
            'sentiment_label': "N/A", 'fed_interest_rate': fed_rate,
            'inflation_cpi': inflation_cpi, 'real_yield': real_yield,
            'usd_index': dxy_index, 'verdict': fundamental_verdict
        }
        
    except Exception as e:
        print(f"Error contacting FRED API: {e}")
        return _emergency_mock_macro()

# --- 4. EMERGENCY FALLBACK FUNCTIONS ---
def _emergency_mock_data(periods=500):
    """Fallback mock data if Polygon fails."""
    print("WARNING: Using emergency mock data.")
    base_price = 2300
    dates = pd.date_range(end=datetime.datetime.now(), periods=periods, freq='min')
    prices = base_price + np.random.randn(periods).cumsum() * 0.1
    return pd.DataFrame({'t': (dates.astype(np.int64) // 10**6), 'o': prices, 'h': prices + 0.5,
                         'l': prices - 0.5, 'c': prices + np.random.randn(periods) * 0.1,
                         'v': np.random.randint(100, 1000, periods)})

def _emergency_mock_macro():
    """Fallback mock macro data if FRED fails."""
    print("WARNING: Using emergency mock macro data.")
    return {'sentiment_label': "N/A", 'fed_interest_rate': 5.25, 'inflation_cpi': 3.5,
            'real_yield': 1.75, 'usd_index': 105.0, 'verdict': "Bearish (Simulated)"}

# --- 5. MULTI-TIMEFRAME FUNCTIONS ---
def get_weekly_data():
    """Fetches REAL weekly data."""
    print("Fetching REAL weekly data from Polygon...")
    if polygon_client is None: return _emergency_mock_data(periods=300)
    try:
        aggs = polygon_client.get_aggs(SYMBOL, 1, "week",
            (datetime.date.today() - datetime.timedelta(weeks=300)).isoformat(),
            datetime.date.today().isoformat(), limit=300, sort="desc")
        df = _process_polygon_aggs(aggs)
        if df.empty: return _emergency_mock_data(periods=300)
        return df
    except Exception as e:
        print(f"Error in get_weekly_data: {e}")
        return _emergency_mock_data(periods=300)

def get_daily_data():
    """Fetches REAL daily data."""
    print("Fetching REAL daily data from Polygon...")
    if polygon_client is None: return _emergency_mock_data(periods=200)
    try:
        aggs = polygon_client.get_aggs(SYMBOL, 1, "day",
            (datetime.date.today() - datetime.timedelta(days=200)).isoformat(),
            datetime.date.today().isoformat(), limit=200, sort="desc")
        df = _process_polygon_aggs(aggs)
        if df.empty: return _emergency_mock_data(periods=200)
        return df
    except Exception as e:
        print(f"Error in get_daily_data: {e}")
        return _emergency_mock_data(periods=200)

def get_h4_data():
    """Fetches REAL 4-hour data."""
    print("Fetching REAL 4-hour data from Polygon...")
    if polygon_client is None: return _emergency_mock_data(periods=100)
    try:
        aggs = polygon_client.get_aggs(SYMBOL, 4, "hour",
            (datetime.date.today() - datetime.timedelta(days=100)).isoformat(),
            datetime.date.today().isoformat(), limit=100, sort="desc")
        df = _process_polygon_aggs(aggs)
        if df.empty: return _emergency_mock_data(periods=100)
        return df
    except Exception as e:
        print(f"Error in get_h4_data: {e}")
        return _emergency_mock_data(periods=100)
    
def get_h1_data():
    """Fetches REAL 1-hour data for LSTM training."""
    print("Fetching REAL 1-hour data from Polygon...")
    if polygon_client is None: return _emergency_mock_data(periods=500) # Need more data for LSTM
    try:
        # We fetch 5000 points, the max allowed, for a good training history
        aggs = polygon_client.get_aggs(SYMBOL, 1, "hour",
            (datetime.date.today() - datetime.timedelta(days=300)).isoformat(), # Approx 300 days
            datetime.date.today().isoformat(), limit=5000, sort="desc")
        df = _process_polygon_aggs(aggs)
        if df.empty: return _emergency_mock_data(periods=500)
        return df
    except Exception as e:
        print(f"Error in get_h1_data: {e}")
        return _emergency_mock_data(periods=500)