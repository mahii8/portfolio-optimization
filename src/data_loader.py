import yfinance as yf
import pandas as pd
import time

TICKERS = ["TSLA", "BND", "SPY"]
START = "2015-01-01"
END = "2026-06-30"

def fetch_one(ticker, start, end, max_retries=5):
    for attempt in range(1, max_retries + 1):
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if df is not None and not df.empty:
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df["Ticker"] = ticker
            return df
        print(f"Attempt {attempt}/{max_retries} failed for {ticker}, retrying...")
        time.sleep(5)
    raise RuntimeError(f"Failed to fetch {ticker} after {max_retries} attempts")

def fetch_data(tickers=TICKERS, start=START, end=END):
    data = {}
    for t in tickers:
        data[t] = fetch_one(t, start, end)
    return data

def combine(data: dict) -> pd.DataFrame:
    combined = pd.concat(data.values())
    combined.index.name = "Date"
    return combined.reset_index()

if __name__ == "__main__":
    data = fetch_data()
    combined = combine(data)
    combined.to_csv("data/processed/raw_combined.csv", index=False)
    print(combined.groupby("Ticker").agg({"Close": ["min","max","mean"]}))
