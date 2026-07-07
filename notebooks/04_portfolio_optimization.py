# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting

df = pd.read_csv("../data/processed/raw_combined.csv", parse_dates=["Date"])
df = df.sort_values(["Ticker", "Date"])

tsla = df[df["Ticker"] == "TSLA"].set_index("Date")["Adj Close"]
bnd = df[df["Ticker"] == "BND"].set_index("Date")["Adj Close"]
spy = df[df["Ticker"] == "SPY"].set_index("Date")["Adj Close"]

prices = pd.concat([tsla, bnd, spy], axis=1)
prices.columns = ["TSLA", "BND", "SPY"]
prices = prices.dropna()
prices.head()

# %%
# --- Expected returns ---
# BND and SPY: historical annualized mean return (reasonable for stable/diversified assets)
hist_mu = expected_returns.mean_historical_return(prices[["BND", "SPY"]])

# TSLA: derived from our Task 3 ARIMA forecast, annualized
# (recall: forecast was essentially flat, implying ~0% expected return over the horizon)
last_price = tsla.iloc[-1]
n_future = 180
forecast_final_price = 411.84  # from Task 3 forecast
tsla_forecast_return = (forecast_final_price / last_price) ** (252 / n_future) - 1

mu = pd.Series({
    "TSLA": tsla_forecast_return,
    "BND": hist_mu["BND"],
    "SPY": hist_mu["SPY"]
})
print("Expected annual returns (mu):")
print(mu)

# %%
# --- Covariance matrix ---
S = risk_models.sample_cov(prices)
print("Covariance matrix:")
print(S)

# %%
# --- Covariance heatmap ---
plt.figure(figsize=(6, 5))
sns.heatmap(S, annot=True, cmap="coolwarm", fmt=".4f")
plt.title("Asset Covariance Matrix")
plt.tight_layout()
plt.savefig("covariance_heatmap.png")
plt.show()

# %%
# --- Max Sharpe portfolio ---
ef_sharpe = EfficientFrontier(mu, S)
weights_sharpe = ef_sharpe.max_sharpe()
cleaned_weights_sharpe = ef_sharpe.clean_weights()
print("Max Sharpe weights:", cleaned_weights_sharpe)
perf_sharpe = ef_sharpe.portfolio_performance(verbose=True)

# %%
# --- Min volatility portfolio ---
ef_minvol = EfficientFrontier(mu, S)
weights_minvol = ef_minvol.min_volatility()
cleaned_weights_minvol = ef_minvol.clean_weights()
print("Min Volatility weights:", cleaned_weights_minvol)
perf_minvol = ef_minvol.portfolio_performance(verbose=True)

# %%
# --- Efficient Frontier plot ---
fig, ax = plt.subplots(figsize=(10, 7))
ef_plot = EfficientFrontier(mu, S)
plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)

ax.scatter(perf_sharpe[1], perf_sharpe[0], marker="*", s=300, c="red", label="Max Sharpe", zorder=5)
ax.scatter(perf_minvol[1], perf_minvol[0], marker="*", s=300, c="green", label="Min Volatility", zorder=5)
ax.legend()
ax.set_title("Efficient Frontier: TSLA, BND, SPY")
plt.tight_layout()
plt.savefig("efficient_frontier.png")
plt.show()

# %%
# --- Summary table ---
summary = pd.DataFrame({
    "Max Sharpe": {**cleaned_weights_sharpe, "Expected Return": perf_sharpe[0],
                   "Volatility": perf_sharpe[1], "Sharpe Ratio": perf_sharpe[2]},
    "Min Volatility": {**cleaned_weights_minvol, "Expected Return": perf_minvol[0],
                        "Volatility": perf_minvol[1], "Sharpe Ratio": perf_minvol[2]}
})
print(summary)
# %%
# %%
# --- SCENARIO 2: Use TSLA's historical mean return instead of the flat ARIMA forecast ---
hist_mu_all = expected_returns.mean_historical_return(prices)
print("Historical annualized returns (all assets):")
print(hist_mu_all)

mu_historical = hist_mu_all.copy()  # TSLA, BND, SPY all using historical mean

# %%
# --- Max Sharpe (historical-return scenario) ---
ef_sharpe_hist = EfficientFrontier(mu_historical, S)
weights_sharpe_hist = ef_sharpe_hist.max_sharpe()
cleaned_weights_sharpe_hist = ef_sharpe_hist.clean_weights()
print("Max Sharpe weights (historical returns):", cleaned_weights_sharpe_hist)
perf_sharpe_hist = ef_sharpe_hist.portfolio_performance(verbose=True)

# %%
# --- Min Volatility (historical-return scenario) ---
# Note: min-vol is independent of expected returns, so this should be IDENTICAL to Scenario 1
ef_minvol_hist = EfficientFrontier(mu_historical, S)
weights_minvol_hist = ef_minvol_hist.min_volatility()
cleaned_weights_minvol_hist = ef_minvol_hist.clean_weights()
print("Min Volatility weights (historical returns):", cleaned_weights_minvol_hist)
perf_minvol_hist = ef_minvol_hist.portfolio_performance(verbose=True)

# %%
# --- Side-by-side comparison table: forecast-based vs historical-based expected returns ---
comparison = pd.DataFrame({
    "Max Sharpe (ARIMA forecast)": {**cleaned_weights_sharpe, "Expected Return": perf_sharpe[0],
                                      "Volatility": perf_sharpe[1], "Sharpe Ratio": perf_sharpe[2]},
    "Max Sharpe (Historical)": {**cleaned_weights_sharpe_hist, "Expected Return": perf_sharpe_hist[0],
                                  "Volatility": perf_sharpe_hist[1], "Sharpe Ratio": perf_sharpe_hist[2]},
    "Min Vol (both scenarios)": {**cleaned_weights_minvol, "Expected Return": perf_minvol[0],
                                   "Volatility": perf_minvol[1], "Sharpe Ratio": perf_minvol[2]},
})
print(comparison)

# %%
# --- Plot both efficient frontiers side-by-side for visual comparison ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ef_plot1 = EfficientFrontier(mu, S)
plotting.plot_efficient_frontier(ef_plot1, ax=axes[0], show_assets=True)
axes[0].scatter(perf_sharpe[1], perf_sharpe[0], marker="*", s=300, c="red", label="Max Sharpe", zorder=5)
axes[0].scatter(perf_minvol[1], perf_minvol[0], marker="*", s=300, c="green", label="Min Vol", zorder=5)
axes[0].set_title("Scenario 1: TSLA return from ARIMA forecast (~0%)")
axes[0].legend()

ef_plot2 = EfficientFrontier(mu_historical, S)
plotting.plot_efficient_frontier(ef_plot2, ax=axes[1], show_assets=True)
axes[1].scatter(perf_sharpe_hist[1], perf_sharpe_hist[0], marker="*", s=300, c="red", label="Max Sharpe", zorder=5)
axes[1].scatter(perf_minvol_hist[1], perf_minvol_hist[0], marker="*", s=300, c="green", label="Min Vol", zorder=5)
axes[1].set_title("Scenario 2: TSLA return from historical mean")
axes[1].legend()

plt.tight_layout()
plt.savefig("efficient_frontier_comparison.png")
plt.show()
# %%
