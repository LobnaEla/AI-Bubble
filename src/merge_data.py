"""
MERGE DATA
==========
Fusionne edgar_panel.csv × financial_data.csv (roic.ai).

LOGIQUE :
  - Prend uniquement les tickers COMMUNS aux deux fichiers
  - pe et pe_premium viennent de roic.ai → time-varying ✓ (plus de pe_proxy)
  - Années 2018–2026

INPUT  :
  data/processed/edgar_panel.csv   → texte NLP (151+ firmes, 2018-2024)
  data/processed/financial_data.csv   → données roic.ai (pe historique ✓)

OUTPUT :
  data/processed/merged_panel.csv

USAGE :
  cd AI_BUBBLE
  python src/merge_data.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

DATA_DIR   = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDGAR_FILE  = DATA_DIR / "edgar_panel.csv"
OUTPUT_FILE = DATA_DIR / "merged_panel.csv"

for fname in ["financial_data.csv", "financial_data.csv"]:
    FIN_FILE = DATA_DIR / fname
    if FIN_FILE.exists():
        break

VALID_YEARS = list(range(2018, 2027))


def main():
    print("=" * 65)
    print("MERGE DATA — edgar × financial_data (roic.ai)")
    print("=" * 65)

    edgar = pd.read_csv(EDGAR_FILE)
    fin   = pd.read_csv(FIN_FILE)

    print(f"\nEdgar  : {edgar.shape[0]} rows | "
          f"{edgar['ticker'].nunique()} firmes | "
          f"années {sorted(edgar['fiscal_year'].unique())}")
    print(f"Financ : {fin.shape[0]} rows | "
          f"{fin['ticker'].nunique()} firmes | "
          f"source : {FIN_FILE.name}")

    # ── 1. Tickers communs ────────────────────────────────────────────────────
    edgar_tickers = set(edgar["ticker"].unique())
    fin_tickers   = set(fin["ticker"].unique())
    common        = edgar_tickers & fin_tickers
    only_edgar    = edgar_tickers - fin_tickers
    only_fin      = fin_tickers   - edgar_tickers

    print(f"\n[1] Intersection des tickers :")
    print(f"    Edgar seulement  : {len(only_edgar)} firmes")
    print(f"    Financ seulement : {len(only_fin)} firmes")
    print(f"    Communs          : {len(common)} firmes  ← on garde ces-là")

    edgar = edgar[edgar["ticker"].isin(common)].copy()
    fin   = fin[fin["ticker"].isin(common)].copy()

    # ── 2. Filtrer edgar HIGH quality ─────────────────────────────────────────
    n_before = len(edgar)
    edgar    = edgar[edgar["data_quality"] == "HIGH"].copy()
    edgar    = edgar.rename(columns={"fiscal_year": "year"})
    print(f"\n[2] Edgar HIGH quality : {n_before} → {len(edgar)} rows "
          f"({edgar['ticker'].nunique()} firmes)")

    # ── 3. Filtrer financial années valides ───────────────────────────────────
    n_before = len(fin)
    fin      = fin[fin["year"].isin(VALID_YEARS)].copy()
    print(f"[3] Financial {VALID_YEARS[0]}–{VALID_YEARS[-1]} : "
          f"{n_before} → {len(fin)} rows")

    # ── 4. Vérifier que le P/E est time-varying ───────────────────────────────
    if "pe" in fin.columns:
        pe_var = fin.groupby("ticker")["pe"].nunique()
        n_var  = (pe_var > 1).sum()
        print(f"\n[4] P/E time-varying : {n_var}/{fin['ticker'].nunique()} firmes")
        if n_var == 0:
            print("    ⚠  P/E identique pour toutes les années !")
            print("       Vérifier que financial_data.py a tourné avec roic.ai")
        else:
            print("    ✓ Bon pour la régression panel")

    # ── 5. Colonnes financières à garder ─────────────────────────────────────
    fin_cols_want = [
        "ticker", "year",
        "pe", "pe_avg", "pe_premium", "sector_pe",
        "revenue", "ebit", "rd",
        "op_margin", "rd_intensity", "rev_growth",
        "mktcap", "ev", "eps", "net_income",
        "log_revenue", "log_mktcap",
        "post_2022",
    ]
    fin_cols = [c for c in fin_cols_want if c in fin.columns]
    print(f"\n[5] Colonnes financières retenues : {len(fin_cols)}")

    # ── 6. Merge inner join ───────────────────────────────────────────────────
    panel = edgar.merge(fin[fin_cols], on=["ticker", "year"], how="inner")
    print(f"\n[6] Panel mergé :")
    print(f"    Rows   : {panel.shape[0]}")
    print(f"    Firmes : {panel['ticker'].nunique()}")
    print(f"    Années : {sorted(panel['year'].unique())}")

    # ── 7. Variables additionnelles ───────────────────────────────────────────
    for col, new_col in [("mktcap", "log_mktcap"), ("revenue", "log_revenue")]:
        if col in panel.columns and new_col not in panel.columns:
            panel[new_col] = np.log(panel[col].clip(lower=1))

    if "post_2022" not in panel.columns:
        panel["post_2022"] = (panel["year"] > 2022).astype(int)

    if "spec_vs_oper_ratio" in panel.columns:
        panel["spec_ratio_x_post"] = (
            panel["spec_vs_oper_ratio"] * panel["post_2022"]
        )

    # ── 8. Variable dépendante ────────────────────────────────────────────────
    y_col = next(
        (c for c in ["pe_premium", "pe_premium_proxy"]
         if c in panel.columns and panel[c].notna().sum() > 0),
        None
    )
    print(f"\n[7] Variable dépendante : {y_col or '⚠ MANQUANTE'}")
    if y_col:
        pvar  = panel.groupby("ticker")[y_col].nunique()
        n_val = panel[y_col].notna().sum()
        print(f"    Valides      : {n_val} / {len(panel)}")
        print(f"    Time-varying : {(pvar > 1).sum()} / {panel['ticker'].nunique()} firmes")

    # ── 9. Valeurs manquantes ─────────────────────────────────────────────────
    key_cols = [
        y_col, "spec_vs_oper_ratio", "speculative_score",
        "operational_score", "log_mktcap", "log_revenue",
        "op_margin", "rev_growth", "rd_intensity", "post_2022",
    ]
    key_cols    = [c for c in key_cols if c and c in panel.columns]
    null_counts = panel[key_cols].isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts):
        print(f"\n[8] Valeurs manquantes :")
        for col, n in null_counts.items():
            print(f"    {col:<30} : {n:>3} NaN ({100*n/len(panel):.1f}%)")
    else:
        print("\n[8] Aucune valeur manquante dans les colonnes clés ✓")

    # ── 10. Sauvegarde ────────────────────────────────────────────────────────
    panel.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Sauvegardé : {OUTPUT_FILE}")
    print(f"  Shape : {panel.shape}")
    print(f"\nProchaine étape : python src/clean_panel.py")


if __name__ == "__main__":
    main()