import pandas as pd
import pandas_ta as ta
from . import preprocessor # Import our existing preprocessor

# (This file no longer needs a print statement)

def analyze_weekly(df_raw_w):
    """Analyzes the Weekly chart for the 200-week SMA context."""
    if df_raw_w.empty:
        return {"verdict": "Weekly Data Not Available"}
        
    df_w = preprocessor.clean_data(df_raw_w)
    
    df_w.ta.sma(length=200, append=True)
    df_w.ffill(inplace=True)
    
    if 'SMA_200' not in df_w.columns or df_w['SMA_200'].isnull().all():
        return {"verdict": "Not enough data for 200W SMA"}

    last_price = df_w['close'].iloc[-1]
    last_sma = df_w['SMA_200'].iloc[-1]
    
    if last_price > last_sma:
        verdict = f"Bullish (Price > {last_sma:.2f} 200W SMA)"
    else:
        verdict = f"Bearish (Price < {last_sma:.2f} 200W SMA)"
        
    return {"verdict": verdict, "sma_200w": last_sma, "price": last_price}

def _calculate_pivot_points(df_daily):
    """Calculates classic Pivot Points using the PREVIOUS day's data."""
    if len(df_daily) < 2: 
        return {}
        
    # Use previous day's High, Low, Close
    prev = df_daily.iloc[-2]
    H, L, C = prev['high'], prev['low'], prev['close']
    
    P = (H + L + C) / 3
    R1 = (2 * P) - L
    S1 = (2 * P) - H
    
    return {
        "P": P,   # Central Pivot
        "R1": R1, # Retail Resistance 1
        "S1": S1  # Retail Support 1
    }

def analyze_daily_h4(df_raw_d, df_raw_h4):
    """Analyzes D1 (Pivots, VPoC) and H4 (RSI)."""
    results = {
        "pivot_points": {},
        "vpoc": 0.0,
        "h4_rsi_status": "N/A",
        "h4_rsi_value": 0
    }
    
    # --- Daily (D1) Analysis ---
    if not df_raw_d.empty:
        df_d = preprocessor.clean_data(df_raw_d)
        
        # 1. Calculate Pivot Points (Retail)
        results["pivot_points"] = _calculate_pivot_points(df_d)
        
        # 2. Calculate VPoC (Institutional)
        try:
            df_d['volume'] = df_d['volume'].astype(float)
            
            # --- FINAL FIX! MANUAL VPOC CALCULATION ---
            # 1. Take the last 30 days of data
            df_30d = df_d.iloc[-30:]
            
            # 2. Create price "bins" (round to nearest dollar)
            price_bins = df_30d['close'].round(0)
            
            # 3. Group by those bins and sum the volume
            volume_at_price = df_30d.groupby(price_bins)['volume'].sum()
            
            # 4. VPoC is the "bin" (price) with the maximum volume
            if not volume_at_price.empty:
                results["vpoc"] = volume_at_price.idxmax()
            # --- END OF MANUAL CALCULATION ---
                
        except Exception as e:
            print(f"Error calculating Volume Profile: {e}")
            results["vpoc"] = 0.0 # Set to 0 on failure

    # --- 4-Hour (H4) Analysis ---
    if not df_raw_h4.empty:
        df_h4 = preprocessor.clean_data(df_raw_h4)
        
        df_h4.ta.rsi(length=14, append=True)
        df_h4.ffill(inplace=True)
        
        if 'RSI_14' in df_h4.columns:
            last_rsi = df_h4['RSI_14'].iloc[-1]
            results["h4_rsi_value"] = last_rsi
            
            if last_rsi > 70: 
                results["h4_rsi_status"] = "Overbought (>70)"
            elif last_rsi < 30: 
                results["h4_rsi_status"] = "Oversold (<30)"
            else: 
                results["h4_rsi_status"] = "Neutral (30-70)"
            
    return results