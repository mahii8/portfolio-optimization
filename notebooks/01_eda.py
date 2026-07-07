# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("../data/processed/raw_combined.csv", parse_dates=["Date"])
df.head()

# %%
# --- Check missing values and dtypes ---
print(df.isna().sum())
print(df.dtypes)

# %%
# --- Clean: sort and fill missing values ---
df = df.sort_values(["Ticker", "Date"])
cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
df[cols] = df.groupby("Ticker")[cols].transform(lambda g: g.ffill().bfill())

# --- Daily returns ---
df["Daily Return"] = df.groupby("Ticker")["Adj Close"].pct_change()

# %%
# --- Plot closing prices ---
fig, ax = plt.subplots(figsize=(12, 5))
for t in df["Ticker"].unique():
    sub = df[df["Ticker"] == t]
    ax.plot(sub["Date"], sub["Adj Close"], label=t)
ax.legend()
ax.set_title("Adjusted Close Price 2015-2026")
plt.savefig("close_prices.png")
plt.show()

# %%
# --- Volatility: rolling mean/std ---
for t in df["Ticker"].unique():
    sub = df[df["Ticker"] == t].set_index("Date")
    roll_std = sub["Daily Return"].rolling(30).std()
    plt.figure(figsize=(12, 4))
    plt.plot(roll_std, label=f"{t} 30d rolling std")
    plt.title(f"{t} Rolling Volatility")
    plt.legend()
    plt.savefig(f"volatility_{t}.png")
    plt.show()

# %%
# --- Outlier detection (returns > 3 std) ---
outliers = df[np.abs(df["Daily Return"]) > 3 * df.groupby("Ticker")["Daily Return"].transform("std")]
print(outliers[["Date", "Ticker", "Daily Return"]].sort_values("Daily Return"))

# %%
# --- ADF stationarity test ---
for t in df["Ticker"].unique():
    sub = df[df["Ticker"] == t]
    price_p = adfuller(sub["Adj Close"].dropna())[1]
    ret_p = adfuller(sub["Daily Return"].dropna())[1]
    print(f"{t}: price p-value={price_p:.4f} (non-stationary if >0.05), "
          f"returns p-value={ret_p:.4f} (stationary if <0.05)")

# %%
# --- Risk metrics: VaR (95%) and Sharpe ratio ---
rf = 0.0  # risk-free rate, simplified
summary = []
for t in df["Ticker"].unique():
    r = df[df["Ticker"] == t]["Daily Return"].dropna()
    var_95 = np.percentile(r, 5)
    sharpe = (r.mean() - rf / 252) / r.std() * np.sqrt(252)
    summary.append({"Ticker": t, "VaR_95": var_95, "Sharpe": sharpe})

print(pd.DataFrame(summary))
# %%
