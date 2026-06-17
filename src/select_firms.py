import pandas as pd
import numpy as np
import re
import time
import os
import sys
from edgar import Company, set_identity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR, RAW_DIR, SCREENING_YEAR,
    FOCUS_SECTORS, AI_SCREENING_KEYWORDS,
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

set_identity("lobna.elabed@telecom-paris.fr")


# S&P 500 list
def get_sp500():
    print("Downloading S&P 500 list :")
    url = ("https://raw.githubusercontent.com/datasets/"
           "s-and-p-500-companies/main/data/constituents.csv")
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Symbol": "ticker", "Security": "name",
        "GICS Sector": "sector", "GICS Sub-Industry": "sub_industry",
    })

    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    print(f"  Got {len(df)} firms")
    return df[["ticker", "name", "sector", "sub_industry"]]


sp500 = get_sp500()
sp500.to_csv(os.path.join(RAW_DIR, "sp500_list.csv"), index=False)
print(f"Saved sp500_list.csv\n")

sp500_filtered = sp500[sp500["sector"].isin(FOCUS_SECTORS)].copy().reset_index(drop=True)
print(f"Firms in target sectors: {len(sp500_filtered)}")
print(sp500_filtered["sector"].value_counts().to_string())


# AI keyword counting
def count_ai_mentions(text):
    if not text or len(text) < 100:
        return np.nan
    text_lower = text.lower()
    total = 0
    for kw in AI_SCREENING_KEYWORDS:
        kw = kw.strip()
        if " " in kw:
            total += text_lower.count(kw)
        else:
            total += len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower))
    wc = max(len(text_lower.split()), 1)
    return round((total / wc) * 1000, 4)


CHECKPOINT = os.path.join(RAW_DIR, f"ai_scores_{SCREENING_YEAR}_checkpoint.csv")

if os.path.exists(CHECKPOINT):
    done_df = pd.read_csv(CHECKPOINT)
    done_tickers = set(done_df["ticker"].tolist())
    print(f"\nResuming from checkpoint: {len(done_tickers)} already done")
else:
    done_df = pd.DataFrame()
    done_tickers = set()

new_rows = []

for idx, row in sp500_filtered.iterrows():
    ticker = row["ticker"]
    if ticker in done_tickers:
        continue

    print(f"[{idx+1}/{len(sp500_filtered)}] {ticker:<6} — {row['name']}")

    entry = {
        "ticker": ticker, "name": row["name"],
        "sector": row["sector"], "sub_industry": row["sub_industry"],
        f"ai_density_{SCREENING_YEAR}": np.nan,
        "mda_length": 0, "note": "",
    }

    try:
        company = Company(ticker)
        filings = company.get_filings(form="10-K")

        # try exact year first, then +-1
        target = None
        for f in filings:
            if f.filing_date.year == SCREENING_YEAR:
                target = f
                break
        if target is None:
            for f in filings:
                if f.filing_date.year in [SCREENING_YEAR - 1, SCREENING_YEAR + 1]:
                    target = f
                    entry["note"] = f"used {f.filing_date.year} filing"
                    break

        if target is None:
            entry["note"] = "no_filing_found"
            new_rows.append(entry)
            continue

        tenk = target.obj()
        mda = tenk["item7"] if "item7" in dir(tenk) else None

        if not mda or len(str(mda)) < 300:
            mda = target.text()  # fallback to full text
            entry["note"] += " full_text_fallback"

        mda = str(mda)
        ai_density = count_ai_mentions(mda)
        entry[f"ai_density_{SCREENING_YEAR}"] = ai_density
        entry["mda_length"] = len(mda)
        if not entry["note"]:
            entry["note"] = "ok"

        print(f"  AI density: {ai_density:.3f}/1000w  ({len(mda):,} chars)")

    except Exception as e:
        entry["note"] = f"error: {str(e)[:80]}"
        print(f"  {entry['note']}")

    new_rows.append(entry)

    # save after each firm in case it crashes
    checkpoint_df = pd.concat([done_df, pd.DataFrame(new_rows)], ignore_index=True)
    checkpoint_df.to_csv(CHECKPOINT, index=False)
    time.sleep(0.6)



score_col = f"ai_density_{SCREENING_YEAR}"
all_firms = pd.read_csv(CHECKPOINT)
scored = all_firms.dropna(subset=[score_col]).copy()

scored["ai_rank_in_sector"] = scored.groupby("sector")[score_col].rank(
    method="first", ascending=False
)
all_firms.to_csv(os.path.join(RAW_DIR, "firm_groups_all_scored.csv"), index=False)


print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
print(f"Firms in target sectors:   {len(sp500_filtered)}")
print(f"Successfully scored:       {len(scored)}")
print(f"EDGAR errors:              {len(all_firms) - len(scored)}")
print()
print("By sector:")
print(scored["sector"].value_counts().to_string())
print()
print(f"AI density {SCREENING_YEAR} stats:")
print(scored[score_col].describe().round(4).to_string())
print()
print(f"Top 10 (highest AI mentions):")
print(scored.nlargest(10, score_col)[["ticker", "name", "sector", score_col]]
      .to_string(index=False))
print()
print(f"Saved: {RAW_DIR}/firm_groups_all_scored.csv  ({len(all_firms)} firms)")