import pandas as pd
import numpy as np
import joblib
import sys
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Path hack to import from the root /modules folder ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
# --- End of hack ---

from modules import data_fetcher, preprocessor, multi_timeframe

# --- Training Constants ---

# ¡NUESTRAS FEATURES DE CONFLUENCIA!
FEATURE_COLUMNS = [
    # Technical Features (M1)
    'open', 'high', 'low', 'close', 'volume',
    'EMA_14', 'EMA_50', 'SMA_200', 'RSI_14',
    'MACD_12_26_9', 
    'ATRr_14',
    
    # Fundamental Features (Paso 1)
    'dxy_index',
    'real_yield',
    
    # Technical Context Features (Paso 2)
    'dist_from_sma200w', # Distancia a la SMA 200 Semanal
    'dist_from_vpoc',    # Distancia al VPoC Diario
    'dist_from_s1',      # Distancia al Soporte 1
    'dist_from_r1',      # Distancia a la Resistencia 1
    'h4_rsi',            # Valor del RSI de 4 Horas
    
    # Time/Session Features
    'hour_of_day',
    'day_of_week'
]

PREDICTION_HORIZON_MINUTES = 5
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "short_term_model.joblib")


def create_target_variable(df, horizon):
    """Creates the target variable (y)."""
    df['future_price'] = df['close'].shift(-horizon)
    df['target_signal'] = (df['future_price'] > df['close']).astype(int)
    return df

def main():
    """Main function to train and save the model."""
    
    print("Starting 'Confluence Model' (v3) training process...")
    
    # --- 1. Obtener Datos Base (M1) ---
    print("Step 1: Fetching M1 historical data (5000 points)...")
    df_raw = data_fetcher.get_initial_historical_data(limit=5000)
    
    # --- 2. Preprocesar Datos M1 ---
    print("Step 2: Cleaning M1 data and creating technical features...")
    df_clean = preprocessor.clean_data(df_raw)
    df_features = preprocessor.create_technical_features(df_clean)
    
    # --- 3. Crear Variable Objetivo ---
    print(f"Step 3: Creating target variable ({PREDICTION_HORIZON_MINUTES} min horizon)...")
    df_labeled = create_target_variable(df_features, PREDICTION_HORIZON_MINUTES)
    
    # --- 4. Obtener Datos de Contexto (P1 & P2) ---
    print("Step 4: Fetching context data (Macro, W1, D1, H4)...")
    macro_data = data_fetcher.get_macro_and_sentiment()
    df_w = data_fetcher.get_weekly_data()
    df_d = data_fetcher.get_daily_data()
    df_h4 = data_fetcher.get_h4_data()
    
    # Analizar el contexto
    weekly_analysis = multi_timeframe.analyze_weekly(df_w)
    daily_h4_analysis = multi_timeframe.analyze_daily_h4(df_d, df_h4)

    # --- 5. INGENIERÍA DE FEATURES (¡EL PASO CLAVE!) ---
    print("Step 5: Engineering confluence features...")
    
    # Añadir features Fundamentales (P1)
    df_labeled['dxy_index'] = macro_data.get('usd_index', 105.0)
    df_labeled['real_yield'] = macro_data.get('real_yield', 1.0)
    
    # Añadir features de Contexto Técnico (P2)
    # (Usamos valores constantes del análisis más reciente para todo el set de entrenamiento)
    sma_200w = weekly_analysis.get('sma_200w', df_labeled['close'].mean())
    vpoc = daily_h4_analysis.get('vpoc', df_labeled['close'].mean())
    s1 = daily_h4_analysis.get('pivot_points', {}).get('S1', df_labeled['close'].mean())
    r1 = daily_h4_analysis.get('pivot_points', {}).get('R1', df_labeled['close'].mean())
    h4_rsi_val = daily_h4_analysis.get('h4_rsi_value', 50)
    
    df_labeled['dist_from_sma200w'] = df_labeled['close'] - sma_200w
    df_labeled['dist_from_vpoc'] = df_labeled['close'] - vpoc
    df_labeled['dist_from_s1'] = df_labeled['close'] - s1
    df_labeled['dist_from_r1'] = df_labeled['close'] - r1
    df_labeled['h4_rsi'] = h4_rsi_val # Constante para todo el set
    
    # Añadir features de Sesión (Tiempo)
    # (Asegurarse de que el índice es datetime)
    if not pd.api.types.is_datetime64_any_dtype(df_labeled.index):
        df_labeled.index = pd.to_datetime(df_labeled.index)
        
    df_labeled['hour_of_day'] = df_labeled.index.hour
    df_labeled['day_of_week'] = df_labeled.index.dayofweek

    # --- 6. Preparar Datos Finales ---
    print("Step 6: Preparing final data for model...")
    df_final = df_labeled.dropna(subset=FEATURE_COLUMNS + ['target_signal'])
    
    if df_final.empty:
        print("Error: No data remaining after feature engineering. Aborting.")
        return

    X = df_final[FEATURE_COLUMNS]
    y = df_final['target_signal']
    
    print(f"Data ready: {len(X)} samples for training.")
    
    # --- 7. Dividir Datos (Train/Test) ---
    print("Step 7: Splitting data (70% train, 30% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    
    # --- 8. Entrenar Modelo (XGBoost) ---
    print("Step 8: Training 'Confluence Model' (v3)...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # --- 9. Validar Modelo ---
    print("Step 9: Validating model on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n--- Backtest Validation ---")
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    
    # --- 10. Guardar Modelo ---
    print(f"Step 10: Saving 'Confluence Model' (v3) to {MODEL_PATH}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    
    print("\nTraining complete and 'Confluence Model' (v3) saved!")

if __name__ == "__main__":
    main()