import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ROOT_DIR, OUTPUT_DIR
csv_path = os.path.join(ROOT_DIR, "data", "working_sample.csv")
df_tickers = pd.read_csv(csv_path)
TICKERS = dict(zip(df_tickers["ticker"], df_tickers["sector"]))

FINANCIAL_DATA_PATH = os.path.join(OUTPUT_DIR, "financial_data.csv")

def get_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fin = stock.financials
        rev = fin.loc["Total Revenue"] if "Total Revenue" in fin.index else None
        rd = (
            fin.loc["Research And Development"]
            if "Research And Development" in fin.index
            else None
        )
        ebit = fin.loc["Operating Income"] if "Operating Income" in fin.index else None
        rows = []
        if rev is not None:
            for date, revenue in rev.items():
                year = date.year
                rd_val = rd[date] if rd is not None else np.nan
                ebit_val = ebit[date] if ebit is not None else np.nan

                rows.append({
                    "ticker": ticker,
                    "year": year,
                    "revenue": revenue,
                    "rd": rd_val,
                    "ebit": ebit_val,
                    "sector": info.get("sector", "Unknown"),
                    "mktcap": info.get("marketCap", np.nan),
                    "pe": info.get("trailingPE", np.nan),
                })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error {ticker}: {e}")
        return pd.DataFrame()

dfs = []
for ticker in TICKERS:
    df = get_financials(ticker)
    dfs.append(df)

df_fin = pd.concat(dfs, ignore_index=True)
df_fin = df_fin.sort_values(["ticker", "year"])
df_fin["rev_growth"] = df_fin.groupby("ticker")["revenue"].pct_change()
df_fin["rd_intensity"] = df_fin["rd"] / df_fin["revenue"]
df_fin["op_margin"] = df_fin["ebit"] / df_fin["revenue"]

sector_pe = (
    df_fin.groupby(["sector", "year"])["pe"]
    .median()
    .reset_index()
    .rename(columns={"pe": "sector_pe"})
)

df_fin = df_fin.merge(sector_pe, on=["sector", "year"], how="left")
df_fin["pe_premium"] = df_fin["pe"] - df_fin["sector_pe"]
df_fin["post_2022"] = (df_fin["year"] >= 2023).astype(int)

df_fin.to_csv(FINANCIAL_DATA_PATH, index=False)