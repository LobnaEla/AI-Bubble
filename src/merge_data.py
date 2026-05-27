import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ROOT_DIR, OUTPUT_DIR, DATA_DIR

EDGAR_PATH = os.path.join(DATA_DIR, "edgar_panel_fixed.csv")
FINANCIAL_PATH = os.path.join(DATA_DIR, "financial_data.csv")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "merged_panel.csv")

print("Loading financial data ...")
df_fin = pd.read_csv(FINANCIAL_PATH)

print("Loading EDGAR text panel ...")
df_edgar = pd.read_csv(EDGAR_PATH)

df_fin["year"] = df_fin["year"].astype(int)
df_edgar["fiscal_year"] = df_edgar["fiscal_year"].astype(int)
df_edgar["ticker"] = df_edgar["ticker"].str.strip().str.upper()
df_fin["ticker"] = df_fin["ticker"].str.strip().str.upper()

fin_tickers = set(df_fin["ticker"].unique())
df_edgar_54 = df_edgar[df_edgar["ticker"].isin(fin_tickers)].copy()

print(f"EDGAR rows for 54 firms: {len(df_edgar_54)}")
print(f"Financial rows: {len(df_fin)}")

EDGAR_COLS = [
    "ticker", "fiscal_year",
    "name",
    "sector",
    "sub_industry",
    "group",
    "filing_date",
    "mda_word_count",
    "ai_keyword_hits",
    "ai_density_per1k",
    "speculative_hits",
    "speculative_score",
    "operational_hits",
    "operational_score",
    "spec_vs_oper_ratio",
    "ai_density_2023",
    "data_quality",
]

df_edgar_sel = df_edgar_54[EDGAR_COLS].copy()
df_edgar_sel.rename(columns={"sector": "sector_edgar"}, inplace=True)

df_merged = df_fin.merge(
    df_edgar_sel,
    left_on=["ticker", "year"],
    right_on=["ticker", "fiscal_year"],
    how="left",
)

df_merged.drop(columns=["fiscal_year"], inplace=True)

total_rows = len(df_merged)
matched_rows = df_merged["ai_density_per1k"].notna().sum()
unmatched = total_rows - matched_rows

print("\nMerge diagnostics")
print(f"Total rows in merged panel: {total_rows}")
print(f"Rows with EDGAR text data: {matched_rows} (fiscal years 2021-2024)")
print(f"Rows without EDGAR data: {unmatched} (fiscal years 2025-2026, not yet filed)")

low_quality = df_merged[df_merged["data_quality"].isin(["LOW", "FAILED"])].shape[0]
print(f"LOW/FAILED quality rows: {low_quality} (flag; not dropped here)")

sector_mismatch = df_merged[
    df_merged["sector_edgar"].notna() &
    (df_merged["sector"] != df_merged["sector_edgar"])
]
if len(sector_mismatch) > 0:
    print(f"\nSector label mismatches between yfinance and EDGAR: {len(sector_mismatch)} rows")
    print(sector_mismatch[["ticker", "year", "sector", "sector_edgar"]].drop_duplicates().to_string(index=False))
else:
    print("Sector labels: fully consistent between yfinance and EDGAR.")

LEAD_COLS = [
    "ticker", "name", "year", "filing_date",
    "sector", "sector_edgar", "sub_industry", "group",
    "data_quality",
]
REMAINING = [c for c in df_merged.columns if c not in LEAD_COLS]
df_merged = df_merged[LEAD_COLS + REMAINING]

df_merged.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
print(f"Shape: {df_merged.shape[0]} rows x {df_merged.shape[1]} columns")
print("\nColumn list:\n  " + "\n  ".join(df_merged.columns.tolist()))