import pandas as pd
import numpy as np
import re
import time
import os, sys
from edgar import Company, set_identity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR,
    SCREENING_YEAR,
    FOCUS_SECTORS,
    SECTOR_SELECTION,
    AI_SCREENING_KEYWORDS,
)

os.makedirs(DATA_DIR, exist_ok=True)

set_identity("lobna.elabed@telecom-paris.fr")

def get_sp500():
    print("GitHub raw CSV...")
    try:
        url = (
            "https://raw.githubusercontent.com/datasets/"
            "s-and-p-500-companies/main/data/constituents.csv"
        )
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        df = df.rename(columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry":"sub_industry",
        })
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        print(f"Got {len(df)} firms from GitHub")
        return df[["ticker", "name", "sector", "sub_industry"]]
    except Exception as e:
        print(f"Method failed: {e}")

sp500 = get_sp500()
sp500.to_csv(os.path.join(DATA_DIR, "sp500_list.csv"), index=False)
print(f"\nSaved sp500_list.csv — {len(sp500)} firms")
print(sp500["sector"].value_counts().to_string())

sp500_filtered = sp500[sp500["sector"].isin(FOCUS_SECTORS)].copy()
sp500_filtered = sp500_filtered.reset_index(drop=True)

print(f"\nFiltered: {len(sp500_filtered)} firms in target sectors")

def count_ai_mentions(text):
    if not text or len(text) < 100:
        return np.nan
    text_lower = text.lower()
    total = 0
    for kw in AI_SCREENING_KEYWORDS:
        if " " in kw.strip():
            total += text_lower.count(kw)
        else:
            total += len(re.findall(r'\b' + re.escape(kw.strip()) + r'\b',
                                    text_lower))
    word_count = max(len(text_lower.split()), 1)
    return round((total / word_count) * 1000, 4)

CHECKPOINT = os.path.join(DATA_DIR, f"ai_scores_{SCREENING_YEAR}_checkpoint.csv")

if os.path.exists(CHECKPOINT):
    done_df = pd.read_csv(CHECKPOINT)
    done_tickers = set(done_df["ticker"].tolist())
    print(f"\nResuming from checkpoint: {len(done_tickers)} firms already done")
else:
    done_df = pd.DataFrame()
    done_tickers = set()

new_rows = []

for idx, row in sp500_filtered.iterrows():
    ticker = row["ticker"]

    if ticker in done_tickers:
        continue

    print(f"[{idx + 1}/{len(sp500_filtered)}] {ticker:6s} - {row['name']}")

    entry = {
        "ticker": ticker,
        "name": row["name"],
        "sector": row["sector"],
        "sub_industry": row["sub_industry"],
        f"ai_density_{SCREENING_YEAR}": np.nan,
        "mda_length": 0,
        "note": "",
    }

    try:
        company = Company(ticker)
        filings = company.get_filings(form="10-K")

        target = None
        for f in filings:
            if f.filing_date.year == SCREENING_YEAR:
                target = f
                break

        if target is None:
            for f in filings:
                if f.filing_date.year in [SCREENING_YEAR-1, SCREENING_YEAR+1]:
                    target = f
                    entry["note"] = f"used {f.filing_date.year} filing"
                    break

        if target is None:
            print(f"No filing found near {SCREENING_YEAR}")
            entry["note"] = "no_filing_found"
            new_rows.append(entry)
            continue

        tenk = target.obj()
        mda = tenk["item7"] if "item7" in dir(tenk) else None

        if not mda or len(str(mda)) < 300:
            mda = target.text()
            entry["note"] += " full_text_fallback"

        mda = str(mda)
        ai_density = count_ai_mentions(mda)

        entry[f"ai_density_{SCREENING_YEAR}"] = ai_density
        entry["mda_length"] = len(mda)
        if not entry["note"]:
            entry["note"] = "ok"

        print(f"AI density: {ai_density:.3f}/1000w ({len(mda):,} chars)")

    except Exception as e:
        entry["note"] = f"error: {str(e)[:80]}"
        print(entry["note"])

    new_rows.append(entry)

    checkpoint_df = pd.concat([done_df, pd.DataFrame(new_rows)], ignore_index=True)
    checkpoint_df.to_csv(CHECKPOINT, index=False)

    time.sleep(0.6)

score_col = f"ai_density_{SCREENING_YEAR}"

scores = pd.read_csv(CHECKPOINT)
scores = scores.dropna(subset=[score_col])
scores = scores[scores[score_col] > 0].copy()

scores["rank_desc_in_sector"] = scores.groupby("sector")[score_col].rank(
    method="first",
    ascending=False
)

scores["rank_asc_in_sector"] = scores.groupby("sector")[score_col].rank(
    method="first",
    ascending=True
)

scores["group"] = "middle"

selected_parts = []

for sector, cfg in SECTOR_SELECTION.items():
    sector_df = scores[scores["sector"] == sector].copy()

    if sector_df.empty:
        print(f"\nNo firms found for sector: {sector}")
        continue

    n_treat = cfg.get("treatment", 0)
    n_ctrl = cfg.get("control", 0)

    treatment = (
        sector_df
        .sort_values(score_col, ascending=False)
        .head(n_treat)
        .copy()
    )
    treatment["group"] = "treatment"

    control = (
        sector_df
        .sort_values(score_col, ascending=True)
        .head(n_ctrl)
        .copy()
    )
    control["group"] = "control"

    selected_parts.extend([treatment, control])

    print(f"\n{sector}")
    print(f"Available firms: {len(sector_df)}")
    print(f"Selected treatment: {len(treatment)}")
    print(f"Selected control: {len(control)}")

    print("\nTreatment:")
    print(treatment[["ticker", "name", score_col]].to_string(index=False))

    print("\nControl:")
    print(control[["ticker", "name", score_col]].to_string(index=False))

working_sample = pd.concat(selected_parts, ignore_index=True)

working_sample = working_sample.drop_duplicates(subset=["ticker"])

scores.to_csv(os.path.join(DATA_DIR, "firm_groups_all_scored.csv"), index=False)

working_sample.to_csv(os.path.join(DATA_DIR, "working_sample.csv"), index=False)

print(f"\nFinal working sample: {len(working_sample)} firms")
print(working_sample[["ticker", "name", "sector", score_col, "group"]].sort_values(["sector", "group", score_col], ascending=[True, True, False]).to_string(index=False))

print("\nSector balance")
print(pd.crosstab(working_sample["sector"], working_sample["group"]))

print("\nSaved:")
print(f"- {DATA_DIR}firm_groups_all_scored.csv")
print(f"- {DATA_DIR}working_sample.csv")

tickers = working_sample["ticker"].tolist()
print("\nPaste this into script 01")
print(f"TICKERS = {tickers}")