"""
NETTOYAGE & PRÉPARATION PANEL FINAL
=====================================
INPUT  : data/processed/panel_with_topics.csv
OUTPUT : data/processed/panel_final.csv
         data/outputs/data_summary.txt

USAGE :
    cd AI_BUBBLE
    python src/clean_panel.py
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

# PRIORITÉ : panel_with_topics a TOUT (NLP corrigé + financial roic.ai)
for fname in ["panel_with_topics.csv", "panel_with_specificity.csv",
              "merged_panel_v2.csv", "merged_panel.csv"]:
    INPUT_FILE = DATA_DIR / fname
    if INPUT_FILE.exists():
        break

OUTPUT_FILE  = DATA_DIR / "panel_final.csv"
SUMMARY_FILE = OUTPUT_DIR / "data_summary.txt"

WINSOR_Q    = 0.01
TOPIC_NAMES = ["opportunity", "adoption", "laborsaving", "rd_investment"]

TICKER_DUPLICATES = {
    "GOOG" : "GOOGL",
    "BRK.B": "BRK.A",
    "NWS"  : "NWSA",
    "FOX"  : "FOXA",
}


def winsorize(series, q=0.01):
    return series.clip(series.quantile(q), series.quantile(1 - q))


def main():
    print("=" * 65)
    print("NETTOYAGE & PRÉPARATION — Panel Final")
    print("=" * 65)

    df = pd.read_csv(INPUT_FILE)
    print(f"Input  : {INPUT_FILE.name}  ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"Firmes : {df['ticker'].nunique()}  |  Années : {sorted(df['year'].unique())}")

    # ── Vérification rapide spec_vs_oper_ratio ────────────────────────────────
    if "spec_vs_oper_ratio" in df.columns:
        max_r = df["spec_vs_oper_ratio"].max()
        if max_r > 10:
            print(f"\n⚠  spec_vs_oper_ratio max = {max_r:.0f} (ancien ratio cassé)")
            print("   → Relancer tense_specificity.py d'abord !")
            return
        else:
            print(f"   spec_vs_oper_ratio : OK (max={max_r:.4f}, borné [0,1])")

    # ── 1. Filtre data_quality ────────────────────────────────────────────────
    if "data_quality" in df.columns:
        n_before = len(df)
        df = df[df["data_quality"] == "HIGH"].copy()
        print(f"\n[1] Filtre data_quality=HIGH : {n_before} → {len(df)}")
    else:
        print("\n[1] data_quality absent — pas de filtre")

    # ── 2. Exclure firmes AI density = 0 sur toute la période ─────────────────
    ai_col = next((c for c in ["ai_density_per1k", "ai_density_2021"]
                   if c in df.columns), None)
    if ai_col:
        n_before = df["ticker"].nunique()
        active   = df.groupby("ticker")[ai_col].sum()
        active   = active[active > 0].index
        df       = df[df["ticker"].isin(active)].copy()
        print(f"\n[2] Exclure firmes AI=0 partout : "
              f"{n_before} → {df['ticker'].nunique()} firmes "
              f"({n_before - df['ticker'].nunique()} exclues)")

    # ── 3. Dédupliquer tickers ────────────────────────────────────────────────
    tickers_present = set(df["ticker"].unique())
    to_drop = [t for t, keep in TICKER_DUPLICATES.items()
               if t in tickers_present and keep in tickers_present]
    if to_drop:
        n_before = len(df)
        df = df[~df["ticker"].isin(to_drop)].copy()
        print(f"\n[3] Doublons supprimés : {to_drop}  ({n_before} → {len(df)} rows)")
    else:
        print(f"\n[3] Aucun doublon ✓")

    # ── 4. Winsorisation pe_premium ───────────────────────────────────────────
    y_col = next((c for c in ["pe_premium", "pe_premium_proxy"]
                  if c in df.columns and df[c].notna().sum() > 0), None)
    if y_col:
        n_valid    = df[y_col].notna().sum()
        valid_mask = df[y_col].notna()
        raw_min, raw_max = df[y_col].min(), df[y_col].max()
        df.loc[valid_mask, y_col] = winsorize(df.loc[valid_mask, y_col], WINSOR_Q)
        pe_var = df.groupby("ticker")[y_col].nunique()
        print(f"\n[4] Winsorisation {y_col} (q={WINSOR_Q}) :")
        print(f"    Avant : [{raw_min:.1f}, {raw_max:.1f}]")
        print(f"    Après : [{df[y_col].min():.1f}, {df[y_col].max():.1f}]")
        print(f"    Valides : {n_valid}/{len(df)}  |  "
              f"Time-varying : {(pe_var>1).sum()}/{df['ticker'].nunique()} firmes")
    else:
        print("\n[4] ⚠ pe_premium absent — lancer financial_data.py + merge_data.py")

    # ── 5. Log-transformations ────────────────────────────────────────────────
    print("\n[5] Log-transformations :")
    for col, new_col in [("mktcap", "log_mktcap"), ("revenue", "log_revenue")]:
        if col in df.columns:
            if new_col not in df.columns:
                df[new_col] = np.log(df[col].clip(lower=1))
            print(f"    {new_col} : mean={df[new_col].mean():.2f}")

    # ── 6. post_2022 ──────────────────────────────────────────────────────────
    if "post_2022" not in df.columns:
        df["post_2022"] = (df["year"] > 2022).astype(int)
    pre  = (df["post_2022"] == 0).sum()
    post = (df["post_2022"] == 1).sum()
    print(f"\n[6] post_2022 : pre={pre}, post={post}")

    # ── 7. Termes d'interaction ───────────────────────────────────────────────
    print("\n[7] Termes d'interaction :")

    # Recalculer spec_ratio_x_post avec le ratio corrigé
    if "spec_vs_oper_ratio" in df.columns:
        df["spec_ratio_x_post"] = df["spec_vs_oper_ratio"] * df["post_2022"]
        print(f"    spec_ratio_x_post ← β₆  (mean={df['spec_ratio_x_post'].mean():.4f})")

    n_topic_inter = 0
    for t in TOPIC_NAMES:
        col = f"topic_{t}"
        if col in df.columns:
            df[f"{col}_x_post"] = df[col] * df["post_2022"]
            n_topic_inter += 1
    if n_topic_inter:
        print(f"    {n_topic_inter} interactions topics × post_2022")

    # ── 8. Centrage spec_vs_oper_ratio ────────────────────────────────────────
    if "spec_vs_oper_ratio" in df.columns:
        mean_r = df["spec_vs_oper_ratio"].mean()
        df["spec_vs_oper_ratio_c"] = df["spec_vs_oper_ratio"] - mean_r
        print(f"\n[8] spec_vs_oper_ratio centré (mean={mean_r:.4f})")

    # ── 9. Gérer rd_intensity NaN ─────────────────────────────────────────────
    if "rd_intensity" in df.columns:
        n_nan = df["rd_intensity"].isna().sum()
        pct   = 100 * n_nan / len(df)
        if pct > 30:
            # Trop de NaN → imputer à 0 (firmes sans R&D déclaré séparément)
            df["rd_intensity"] = df["rd_intensity"].fillna(0)
            print(f"\n[9] rd_intensity : {n_nan} NaN ({pct:.0f}%) → imputé à 0")
            print("    (firmes qui ne déclarent pas de R&D séparé)")
        else:
            print(f"\n[9] rd_intensity : {n_nan} NaN ({pct:.0f}%)")

    # ── 10. Rapport valeurs manquantes ────────────────────────────────────────
    key_cols = [c for c in [
        y_col, "spec_vs_oper_ratio", "speculative_score", "operational_score",
        "log_mktcap", "log_revenue", "op_margin", "rd_intensity", "rev_growth",
    ] + [f"topic_{t}" for t in TOPIC_NAMES] if c and c in df.columns]

    print(f"\n[10] Valeurs manquantes :")
    any_null = False
    for col in key_cols:
        n = df[col].isna().sum()
        if n > 0:
            print(f"    {col:<30} : {n:>3} NaN ({100*n/len(df):.1f}%)")
            any_null = True
    if not any_null:
        print("    Aucune ✓")

    # ── 11. Sauvegarde ────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Sauvegardé : {OUTPUT_FILE}")
    print(f"  Shape        : {df.shape}")
    print(f"  Firmes       : {df['ticker'].nunique()}")
    print(f"  Années       : {sorted(df['year'].unique())}")

    # Summary
    summary = df[key_cols].describe().round(4)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(f"SUMMARY STATISTICS — panel_final.csv\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Shape  : {df.shape}\nFirmes : {df['ticker'].nunique()}\n")
        f.write(f"Années : {sorted(df['year'].unique())}\n\n")
        f.write(summary.to_string())
    print(f"✓ Summary      : {SUMMARY_FILE}")

    print(f"\n{'─'*65}")
    print("→ python src/regression.py")
    print(f"{'─'*65}")


if __name__ == "__main__":
    main()