# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data/processed/raw_combined.csv", parse_dates=["Date"])
df = df.sort_values(["Ticker", "Date"])

tsla = df[df["Ticker"] == "TSLA"].set_index("Date")["Adj Close"]
bnd = df[df["Ticker"] == "BND"].set_index("Date")["Adj Close"]
spy = df[df["Ticker"] == "SPY"].set_index("Date")["Adj Close"]

prices = pd.concat([tsla, bnd, spy], axis=1)
prices.columns = ["TSLA", "BND", "SPY"]
prices = prices.dropna()

# %%
# --- Define backtest period ---
backtest_start, backtest_end = "2025-01-01", "2026-06-30"

returns = prices.pct_change().dropna()
bt = returns.loc[backtest_start:backtest_end]

print(f"Backtest period: {bt.index.min()} to {bt.index.max()} ({len(bt)} trading days)")

# %%
# --- Define strategy weights (from Task 4: historical-return Max Sharpe) ---
strategy_weights = pd.Series({"TSLA": 0.05361, "BND": 0.55849, "SPY": 0.38790})

# Buy-and-hold: weights fixed at start, no rebalancing (simplest backtest assumption)
strategy_daily = (bt * strategy_weights).sum(axis=1)
strategy_cum = (1 + strategy_daily).cumprod()

# %%
# --- Benchmark: standard 60/40 SPY/BND ---
benchmark_weights = pd.Series({"TSLA": 0.0, "BND": 0.40, "SPY": 0.60})
benchmark_daily = (bt * benchmark_weights).sum(axis=1)
benchmark_cum = (1 + benchmark_daily).cumprod()

# %%
# --- Plot cumulative returns ---
plt.figure(figsize=(12, 6))
plt.plot(strategy_cum, label="Optimized Strategy (5.4% TSLA / 55.8% BND / 38.8% SPY)")
plt.plot(benchmark_cum, label="60/40 SPY/BND Benchmark")
plt.legend()
plt.title("Cumulative Returns: Strategy vs Benchmark (2025-01 to 2026-06)")
plt.ylabel("Growth of $1")
plt.savefig("backtest_cumulative_returns.png")
plt.show()

# %%
# --- Performance metrics ---
def perf_metrics(daily_returns, name):
    total_return = (1 + daily_returns).prod() - 1
    ann_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    cum = (1 + daily_returns).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        "Strategy": name,
        "Total Return": total_return,
        "Annualized Return": ann_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }

results = pd.DataFrame([
    perf_metrics(strategy_daily, "Optimized Strategy"),
    perf_metrics(benchmark_daily, "60/40 Benchmark"),
])
print(results.to_string(index=False))

# %%
# --- Drawdown comparison plot ---
strategy_dd = (strategy_cum - strategy_cum.cummax()) / strategy_cum.cummax()
benchmark_dd = (benchmark_cum - benchmark_cum.cummax()) / benchmark_cum.cummax()

plt.figure(figsize=(12, 4))
plt.plot(strategy_dd, label="Optimized Strategy")
plt.plot(benchmark_dd, label="60/40 Benchmark")
plt.legend()
plt.title("Drawdown Comparison")
plt.ylabel("Drawdown")
plt.savefig("backtest_drawdown.png")
plt.show()
# %%
