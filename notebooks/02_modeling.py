# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("../data/processed/raw_combined.csv", parse_dates=["Date"])
df = df.sort_values(["Ticker", "Date"])

tsla = df[df["Ticker"] == "TSLA"].set_index("Date")["Adj Close"]
tsla.head()

# %%
# --- Chronological train/test split ---
train_end = "2024-12-31"
train = tsla[tsla.index <= train_end]
test = tsla[tsla.index > train_end]

print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} rows)")
print(f"Test:  {test.index.min()} to {test.index.max()} ({len(test)} rows)")

# %%
# --- ARIMA model via auto_arima ---
arima_model = pm.auto_arima(
    train,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    trace=True
)

raw_forecast, arima_conf_int = arima_model.predict(n_periods=len(test), return_conf_int=True)

# FIX: use .values to avoid index-alignment NaN bug (pmdarima returns its own integer index)
arima_forecast = pd.Series(raw_forecast.values, index=test.index)

print(arima_model.summary())
print("ARIMA order:", arima_model.order)

# %%
# --- Plot ARIMA forecast vs actual ---
plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual (Test)")
plt.plot(test.index, arima_forecast, label="ARIMA Forecast")
plt.legend()
plt.title("TSLA: ARIMA Forecast vs Actual")
plt.savefig("arima_forecast.png")
plt.show()

# %%
# --- Prepare sequences for LSTM ---
def make_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

window = 60
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))

X_train, y_train = make_sequences(train_scaled, window)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# %%
# --- Build and train LSTM ---
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(window, 1)),
    LSTM(50),
    Dense(1)
])
lstm_model.compile(optimizer="adam", loss="mse")

history = lstm_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    verbose=1
)

# %%
# --- Generate LSTM forecast for the test period ---
full_scaled = scaler.transform(pd.concat([train, test]).values.reshape(-1, 1))
inputs = full_scaled[len(train) - window:]

X_test = []
for i in range(window, len(inputs)):
    X_test.append(inputs[i-window:i, 0])
X_test = np.array(X_test).reshape(-1, window, 1)

lstm_preds_scaled = lstm_model.predict(X_test)
lstm_preds = scaler.inverse_transform(lstm_preds_scaled).flatten()
lstm_forecast = pd.Series(lstm_preds, index=test.index)

# %%
# --- Plot LSTM forecast vs actual ---
plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual (Test)")
plt.plot(test.index, lstm_forecast, label="LSTM Forecast")
plt.legend()
plt.title("TSLA: LSTM Forecast vs Actual")
plt.savefig("lstm_forecast.png")
plt.show()

# %%
# --- Combined comparison plot ---
plt.figure(figsize=(14, 6))
plt.plot(train.index, train, label="Train", color="gray", alpha=0.5)
plt.plot(test.index, test, label="Actual (Test)", color="black")
plt.plot(test.index, arima_forecast, label="ARIMA Forecast", linestyle="--")
plt.plot(test.index, lstm_forecast, label="LSTM Forecast", linestyle="--")
plt.legend()
plt.title("TSLA: ARIMA vs LSTM Forecast Comparison")
plt.savefig("model_comparison.png")
plt.show()

# %%
# --- Evaluate both models ---
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate(y_true, y_pred, name=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    m = mape(y_true.values, y_pred.values)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "MAPE": m}

results = pd.DataFrame([
    evaluate(test, arima_forecast, "ARIMA"),
    evaluate(test, lstm_forecast, "LSTM"),
])
print(results)
# %%
