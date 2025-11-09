import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Path hack to import from the root /modules folder ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
# --- End of hack ---

from modules import data_fetcher, preprocessor

# --- LSTM Constants ---
# How many past hours will the model look at?
TIME_STEP = 72  # Look at the last 72 hours (3 days)
# How many hours in the future to predict?
PREDICTION_HORIZON_HOURS = 4  # Predict the price 4 hours from now

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "long_term_model.h5") # Keras model
SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.joblib") # We must save the scaler

def create_sequences(data, time_step, horizon):
    """
    Converts time series data into sequences (X) and targets (y).
    X = [sample 1, sample 2, ..., sample 72]
    y = [sample 76]  (4 hours ahead)
    """
    X, y = [], []
    for i in range(len(data) - time_step - horizon + 1):
        X.append(data[i:(i + time_step), 0]) # Get [0...71] as X
        y.append(data[i + time_step + horizon - 1, 0]) # Get [75] as y
    return np.array(X), np.array(y)

def main():
    """Main function to train and save the LSTM model."""
    
    print("Starting LSTM (Long-Term Pattern) training process...")

    # --- 1. Get Data (H1) ---
    print("Step 1: Fetching H1 historical data (up to 5000 points)...")
    # We need a lot of data for an LSTM
    df_raw = data_fetcher.get_h1_data()
    if df_raw.empty:
        print("Error: No H1 data fetched. Aborting LSTM training.")
        return

    # --- 2. Preprocess Data ---
    print("Step 2: Cleaning data and selecting 'close' price...")
    df_clean = preprocessor.clean_data(df_raw)
    
    # LSTM only needs the 'close' price for this simple model
    data = df_clean[['close']].values

    # --- 3. Scale Data ---
    print("Step 3: Scaling data (MinMaxScaler)...")
    # LSTMs are very sensitive to scale. We scale data between 0 and 1.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # --- 4. Create Sequences ---
    print(f"Step 4: Creating sequences (TimeStep={TIME_STEP}, Horizon={PREDICTION_HORIZON_HOURS})...")
    X, y = create_sequences(scaled_data, TIME_STEP, PREDICTION_HORIZON_HOURS)
    
    if len(X) == 0:
        print("Error: Not enough data to create sequences. Need more H1 data.")
        return

    # Reshape X for LSTM input: [samples, time_steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # --- 5. Split Data (Train/Test) ---
    # We use a 90/10 split for LSTMs, as they need more training data.
    split_index = int(len(X) * 0.9)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    print(f"Data ready: {len(X_train)} training sequences, {len(X_test)} testing sequences.")

    # --- 6. Build LSTM Model ---
    print("Step 6: Building LSTM model (Keras)...")
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(TIME_STEP, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=50))
    model.add(Dense(units=1)) # Output layer: 1 value (the predicted price)

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # --- 7. Train Model ---
    print("Step 7: Training LSTM model...")
    # Stop training if the validation loss doesn't improve
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Save the best model found during training
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min')

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # --- 8. Save Scaler ---
    print(f"Step 8: Saving LSTM scaler to {SCALER_PATH}...")
    # We MUST save the scaler, otherwise we can't make new predictions
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    
    print("\nLSTM Training complete and model/scaler saved!")

if __name__ == "__main__":
    main()