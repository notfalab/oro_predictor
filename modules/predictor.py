import joblib
import os
import numpy as np
import pandas as pd

# --- Model Loading ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "short_term_model.joblib")
_model = None

# Esta es la lista de features que el modelo (v3) fue entrenado para esperar.
# DEBE ser idéntica a la del script de entrenamiento.
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume', 'EMA_14', 'EMA_50', 
    'SMA_200', 'RSI_14', 'MACD_12_26_9', 'ATRr_14', 
    'dxy_index', 'real_yield',
    'dist_from_sma200w', 'dist_from_vpoc', 'dist_from_s1', 'dist_from_r1',
    'h4_rsi', 'hour_of_day', 'day_of_week'
]

def load_model():
    """Loads the short-term (XGBoost) model from disk."""
    global _model
    if _model is not None: return _model

    try:
        print(f"Loading 'Confluence Model' (v3) from: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        print("Confluence Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run 'python3 training/train_models.py' first.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    return _model

# --- Prediction Functions ---

def make_short_term_prediction(latest_features_df, macro_data, weekly_analysis, daily_h4_analysis):
    """
    (¡NUEVA FIRMA!)
    Genera una predicción usando el set completo de features de confluencia.
    """
    model = load_model()
    
    if model is None:
        return {"signal": "No Model", "confidence": 0, "recommendation": "Error: Model not loaded."}
        
    if latest_features_df is None or latest_features_df.empty:
        return {"signal": "Calculating...", "confidence": 0, "recommendation": "Waiting for technical data..."}

    # --- INGENIERÍA DE FEATURES EN TIEMPO REAL ---
    # 1. Prepara el DataFrame de entrada
    final_input_df = latest_features_df.copy()
    
    # 2. Añadir features Fundamentales (P1)
    final_input_df['dxy_index'] = macro_data.get('usd_index', 105.0)
    final_input_df['real_yield'] = macro_data.get('real_yield', 1.0)
    
    # 3. Añadir features de Contexto Técnico (P2)
    current_price = final_input_df['close'].iloc[0]
    sma_200w = weekly_analysis.get('sma_200w', current_price)
    vpoc = daily_h4_analysis.get('vpoc', current_price)
    s1 = daily_h4_analysis.get('pivot_points', {}).get('S1', current_price)
    r1 = daily_h4_analysis.get('pivot_points', {}).get('R1', current_price)
    
    final_input_df['dist_from_sma200w'] = current_price - sma_200w
    final_input_df['dist_from_vpoc'] = current_price - vpoc
    final_input_df['dist_from_s1'] = current_price - s1
    final_input_df['dist_from_r1'] = current_price - r1
    final_input_df['h4_rsi'] = daily_h4_analysis.get('h4_rsi_value', 50)
    
    # 4. Añadir features de Sesión (Tiempo)
    # (Asegurarse de que el índice es datetime)
    if not pd.api.types.is_datetime64_any_dtype(final_input_df.index):
        final_input_df.index = pd.to_datetime(final_input_df.index)
    final_input_df['hour_of_day'] = final_input_df.index.hour
    final_input_df['day_of_week'] = final_input_df.index.dayofweek
    # --- FIN DE INGENIERÍA DE FEATURES ---

    # 5. Reordenar y seleccionar solo las columnas necesarias
    try:
        final_input_for_model = final_input_df[FEATURE_COLUMNS]
    except KeyError as e:
        print(f"Error: Missing columns for model: {e}")
        # Imprimir qué columnas faltan
        missing_cols = [col for col in FEATURE_COLUMNS if col not in final_input_df.columns]
        print(f"Missing: {missing_cols}")
        return {"signal": "Column Error", "confidence": 0, "recommendation": str(e)}

    # --- PREDICCIÓN ---
    try:
        probabilities = model.predict_proba(final_input_for_model)
        prob_bullish = probabilities[0][1] 
        
        signal = "Bullish" if prob_bullish > 0.5 else "Bearish"
        confidence = abs(prob_bullish - 0.5) * 2 
        
        rsi = latest_features_df['RSI_14'].iloc[0]
        recommendation = f"'Confluence Model' Signal: {signal}"
        
        if signal == "Bullish" and rsi < 30:
            recommendation = "Strong Buy (Confluence Model Bullish + M1 RSI Oversold)"
        elif signal == "Bearish" and rsi > 70:
            recommendation = "Strong Sell (Confluence Model Bearish + M1 RSI Overbought)"

        return {
            "signal": signal,
            "confidence": f"{confidence:.1%}",
            "recommendation": recommendation
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"signal": "Error", "confidence": 0, "recommendation": str(e)}