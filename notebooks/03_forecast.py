# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm

df = pd.read_csv("../data/processed/raw_combined.csv", parse_dates=["Date"])
df = df.sort_values(["Ticker", "Date"])

tsla = df[df["Ticker"] == "TSLA"].set_index("Date")["Adj Close"]

# %%
# --- Refit ARIMA on ALL available historical data (not just the old train split) ---
# For future forecasting, we want the model to use everything we know up to today
arima_model = pm.auto_arima(
    tsla,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    trace=True
)

print(arima_model.summary())
print("ARIMA order:", arima_model.order)

# %%
# --- Generate future forecast (6-12 months = ~126-252 trading days) ---
n_future = 180  # ~9 months of trading days as a middle ground

raw_forecast, conf_int = arima_model.predict(n_periods=n_future, return_conf_int=True)
future_forecast = pd.Series(raw_forecast.values, index=pd.bdate_range(
    start=tsla.index[-1], periods=n_future + 1
)[1:])

conf_int_df = pd.DataFrame(
    conf_int, columns=["lower", "upper"], index=future_forecast.index
)

print(future_forecast.head())
print(conf_int_df.head())

# %%
# --- Plot historical + future forecast with confidence intervals ---
plt.figure(figsize=(14, 6))
plt.plot(tsla.index, tsla, label="Historical", color="black")
plt.plot(future_forecast.index, future_forecast, label="Future Forecast", color="red")
plt.fill_between(
    future_forecast.index,
    conf_int_df["lower"],
    conf_int_df["upper"],
    color="red", alpha=0.15, label="95% Confidence Interval"
)
plt.legend()
plt.title("TSLA: 9-Month Future Forecast with Confidence Intervals")
plt.savefig("future_forecast.png")
plt.show()

# %%
# --- Zoomed-in view: last 2 years of history + full forecast, for clarity ---
zoom_start = tsla.index[-1] - pd.Timedelta(days=730)
tsla_zoom = tsla[tsla.index >= zoom_start]

plt.figure(figsize=(14, 6))
plt.plot(tsla_zoom.index, tsla_zoom, label="Historical (last 2yr)", color="black")
plt.plot(future_forecast.index, future_forecast, label="Future Forecast", color="red")
plt.fill_between(
    future_forecast.index,
    conf_int_df["lower"],
    conf_int_df["upper"],
    color="red", alpha=0.15, label="95% Confidence Interval"
)
plt.legend()
plt.title("TSLA: Forecast Zoomed View")
plt.savefig("future_forecast_zoomed.png")
plt.show()

# %%
# --- Analyze how CI width changes over the forecast horizon ---
ci_width = conf_int_df["upper"] - conf_int_df["lower"]

plt.figure(figsize=(12, 4))
plt.plot(future_forecast.index, ci_width)
plt.title("Confidence Interval Width Over Forecast Horizon")
plt.ylabel("CI Width ($)")
plt.savefig("ci_width_growth.png")
plt.show()

print(f"CI width at day 1:   ${ci_width.iloc[0]:.2f}")
print(f"CI width at day 30:  ${ci_width.iloc[29]:.2f}")
print(f"CI width at day 90:  ${ci_width.iloc[89]:.2f}")
print(f"CI width at day 180: ${ci_width.iloc[-1]:.2f}")
print(f"\nGrowth factor day1->day180: {ci_width.iloc[-1] / ci_width.iloc[0]:.2f}x")

# %%
# --- Summary stats for opportunity/risk discussion ---
last_price = tsla.iloc[-1]
final_forecast_price = future_forecast.iloc[-1]
pct_change = (final_forecast_price - last_price) / last_price * 100

print(f"Last known price: ${last_price:.2f}")
print(f"Forecast price in {n_future} trading days: ${final_forecast_price:.2f}")
print(f"Implied change: {pct_change:+.2f}%")
print(f"\n95% CI at end of horizon: [${conf_int_df['lower'].iloc[-1]:.2f}, ${conf_int_df['upper'].iloc[-1]:.2f}]")
# %%
