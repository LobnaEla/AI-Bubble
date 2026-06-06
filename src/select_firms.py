"""
SELECT FIRMS
============
Sélectionne toutes les firmes S&P 500 des secteurs cibles
et calcule leur densité AI dans le 10-K de SCREENING_YEAR (2021).

CE QUE CE SCRIPT FAIT :
  1. Télécharge la liste S&P 500
  2. Filtre sur les secteurs dans FOCUS_SECTORS
  3. Pour chaque firme, télécharge son 10-K de 2021 via EDGAR
  4. Compte les mentions AI → ai_density_2021 (variable de contrôle)
  5. Sauvegarde TOUTES les firmes scorées (pas de filtre top/bottom)

POURQUOI ai_density_2021 ?
  C'est le niveau AI "de base" de chaque firme avant ChatGPT.
  Utilisé comme variable de contrôle dans la régression pour
  distinguer les firmes naturellement tech-heavy des autres.

OUTPUT :
  data/raw/sp500_list.csv              ← liste complète S&P 500
  data/raw/firm_groups_all_scored.csv  ← toutes les firmes avec leur score AI

USAGE :
  cd AI_BUBBLE
  python src/select_firms.py
"""

import pandas as pd
import numpy as np
import re
import time
import os
import sys
from edgar import Company, set_identity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR,
    RAW_DIR,
    SCREENING_YEAR,
    FOCUS_SECTORS,
    AI_SCREENING_KEYWORDS,
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR,  exist_ok=True)

set_identity("lobna.elabed@telecom-paris.fr")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Liste S&P 500
# ─────────────────────────────────────────────────────────────────────────────

def get_sp500():
    print("Téléchargement liste S&P 500...")
    url = (
        "https://raw.githubusercontent.com/datasets/"
        "s-and-p-500-companies/main/data/constituents.csv"
    )
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Symbol"          : "ticker",
        "Security"        : "name",
        "GICS Sector"     : "sector",
        "GICS Sub-Industry": "sub_industry",
    })
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    print(f"  {len(df)} firmes récupérées")
    return df[["ticker", "name", "sector", "sub_industry"]]


sp500 = get_sp500()
sp500.to_csv(os.path.join(RAW_DIR, "sp500_list.csv"), index=False)
print(f"Sauvegardé : sp500_list.csv\n")

sp500_filtered = sp500[sp500["sector"].isin(FOCUS_SECTORS)].copy().reset_index(drop=True)
print(f"Firmes dans les secteurs cibles : {len(sp500_filtered)}")
print(sp500_filtered["sector"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 2. Comptage mentions AI dans le 10-K de SCREENING_YEAR
# ─────────────────────────────────────────────────────────────────────────────

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
    word_count = max(len(text_lower.split()), 1)
    return round((total / word_count) * 1000, 4)


CHECKPOINT = os.path.join(RAW_DIR, f"ai_scores_{SCREENING_YEAR}_checkpoint.csv")

if os.path.exists(CHECKPOINT):
    done_df      = pd.read_csv(CHECKPOINT)
    done_tickers = set(done_df["ticker"].tolist())
    print(f"\nReprise depuis checkpoint : {len(done_tickers)} firmes déjà traitées")
else:
    done_df      = pd.DataFrame()
    done_tickers = set()

new_rows = []

for idx, row in sp500_filtered.iterrows():
    ticker = row["ticker"]

    if ticker in done_tickers:
        continue

    print(f"[{idx+1}/{len(sp500_filtered)}] {ticker:<6} — {row['name']}")

    entry = {
        "ticker"                      : ticker,
        "name"                        : row["name"],
        "sector"                      : row["sector"],
        "sub_industry"                : row["sub_industry"],
        f"ai_density_{SCREENING_YEAR}": np.nan,
        "mda_length"                  : 0,
        "note"                        : "",
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
                if f.filing_date.year in [SCREENING_YEAR - 1, SCREENING_YEAR + 1]:
                    target = f
                    entry["note"] = f"used {f.filing_date.year} filing"
                    break

        if target is None:
            entry["note"] = "no_filing_found"
            new_rows.append(entry)
            continue

        tenk = target.obj()
        mda  = tenk["item7"] if "item7" in dir(tenk) else None

        if not mda or len(str(mda)) < 300:
            mda = target.text()
            entry["note"] += " full_text_fallback"

        mda        = str(mda)
        ai_density = count_ai_mentions(mda)

        entry[f"ai_density_{SCREENING_YEAR}"] = ai_density
        entry["mda_length"]                   = len(mda)
        if not entry["note"]:
            entry["note"] = "ok"

        print(f"  AI density : {ai_density:.3f}/1000w  ({len(mda):,} chars)")

    except Exception as e:
        entry["note"] = f"error: {str(e)[:80]}"
        print(f"  {entry['note']}")

    new_rows.append(entry)

    checkpoint_df = pd.concat([done_df, pd.DataFrame(new_rows)], ignore_index=True)
    checkpoint_df.to_csv(CHECKPOINT, index=False)
    time.sleep(0.6)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Assembler et sauvegarder
# ─────────────────────────────────────────────────────────────────────────────

score_col = f"ai_density_{SCREENING_YEAR}"
all_firms = pd.read_csv(CHECKPOINT)

# Firmes scorées avec succès
scored = all_firms.dropna(subset=[score_col]).copy()

# Ranking intra-secteur (informatif)
scored["ai_rank_in_sector"] = scored.groupby("sector")[score_col].rank(
    method="first", ascending=False
)

all_firms.to_csv(os.path.join(RAW_DIR, "firm_groups_all_scored.csv"), index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Rapport
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("RÉSULTAT FINAL")
print(f"{'='*65}")
print(f"Firmes dans les secteurs cibles : {len(sp500_filtered)}")
print(f"Firmes scorées avec succès      : {len(scored)}")
print(f"Firmes avec erreur EDGAR        : {len(all_firms) - len(scored)}")
print()
print("Firmes par secteur :")
print(scored["sector"].value_counts().to_string())
print()
print(f"AI density {SCREENING_YEAR} — statistiques :")
print(scored[score_col].describe().round(4).to_string())
print()
print(f"Top 10 firmes (plus de mentions AI) :")
print(
    scored.nlargest(10, score_col)[["ticker", "name", "sector", score_col]]
    .to_string(index=False)
)
print()
print(f"Fichier sauvegardé :")
print(f"  {RAW_DIR}/firm_groups_all_scored.csv  ({len(all_firms)} firmes)")
print()
print(f"Prochaine étape :")
print(f"  python src/financial_data.py")
print(f"  (récupère les données roic.ai pour les {len(scored)} firmes scorées)")