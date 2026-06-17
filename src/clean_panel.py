import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

DATA_DIR = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for fname in ["panel_with_topics.csv", "panel_with_specificity.csv",
              "merged_panel_v2.csv", "merged_panel.csv"]:
    INPUT_FILE = DATA_DIR / fname
    if INPUT_FILE.exists():
        break

OUTPUT_FILE = DATA_DIR / "panel_final.csv"
SUMMARY_FILE = OUTPUT_DIR / "data_summary.txt"

WINSOR_Q = 0.01
TOPICS = ["opportunity", "adoption", "laborsaving", "rd_investment"]

# duplicate tickers to drop
DUPES = {"GOOG": "GOOGL", "BRK.B": "BRK.A", "NWS": "NWSA", "FOX": "FOXA"}


def winsorize(s, q=0.01):
    return s.clip(s.quantile(q), s.quantile(1-q))


def main():
    print("=" * 60)
    print("Panel cleaning & preparation")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)
    print(f"Input: {INPUT_FILE.name}  ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"Firms: {df['ticker'].nunique()}  |  Years: {sorted(df['year'].unique())}")

    # sanity check: spec_vs_oper_ratio should be bounded [0,1]
    if "spec_vs_oper_ratio" in df.columns:
        mx = df["spec_vs_oper_ratio"].max()
        if mx > 10:
            print(f"\nspec_vs_oper_ratio max = {mx:.0f} — looks broken, "
                  "rerun tense_specificity.py first")
            return
        print(f"  spec_vs_oper_ratio OK (max={mx:.4f})")

    # data_quality filter
    if "data_quality" in df.columns:
        n0 = len(df)
        df = df[df["data_quality"] == "HIGH"].copy()
        print(f"\nKept data_quality=HIGH: {n0} -> {len(df)}")
    else:
        print("\nNo data_quality column, skipping filter")

    # dropping firms with zero AI density across entire period
    ai_col = next((c for c in ["ai_density_per1k", "ai_density_2021"]
                   if c in df.columns), None)
    if ai_col:
        n_firms_before = df["ticker"].nunique()
        active = df.groupby("ticker")[ai_col].sum()
        active = active[active > 0].index
        df = df[df["ticker"].isin(active)].copy()
        dropped = n_firms_before - df["ticker"].nunique()
        print(f"Dropped {dropped} firms with AI=0 everywhere "
              f"({n_firms_before} -> {df['ticker'].nunique()})")

    # deduplicating tickers (GOOG/GOOGL etc)
    present = set(df["ticker"].unique())
    to_drop = [t for t, keep in DUPES.items()
               if t in present and keep in present]
    if to_drop:
        n0 = len(df)
        df = df[~df["ticker"].isin(to_drop)].copy()
        print(f"Removed duplicate tickers {to_drop}: {n0} -> {len(df)} rows")

    # winsorize pe_premium
    y_col = next((c for c in ["pe_premium", "pe_premium_proxy"]
                  if c in df.columns and df[c].notna().sum() > 0), None)
    if y_col:
        valid = df[y_col].notna()
        raw_range = (df[y_col].min(), df[y_col].max())
        df.loc[valid, y_col] = winsorize(df.loc[valid, y_col], WINSOR_Q)
        pe_var = df.groupby("ticker")[y_col].nunique()
        print(f"\nWinsorized {y_col} (q={WINSOR_Q}): "
              f"[{raw_range[0]:.1f}, {raw_range[1]:.1f}] -> "
              f"[{df[y_col].min():.1f}, {df[y_col].max():.1f}]")
        print(f"  Valid: {valid.sum()}/{len(df)}  |  "
              f"Time-varying: {(pe_var>1).sum()}/{df['ticker'].nunique()}")
    else:
        print("\npe_premium not found — run financial_data.py + merge_data.py")

    # log transforms
    for col, lcol in [("mktcap", "log_mktcap"), ("revenue", "log_revenue")]:
        if col in df.columns and lcol not in df.columns:
            df[lcol] = np.log(df[col].clip(lower=1))
            print(f"{lcol}: mean={df[lcol].mean():.2f}")

    # post_2022 dummy
    if "post_2022" not in df.columns:
        df["post_2022"] = (df["year"] > 2022).astype(int)
    print(f"\npost_2022 split: pre={(df['post_2022']==0).sum()}, "
          f"post={(df['post_2022']==1).sum()}")

    # interaction terms
    if "spec_vs_oper_ratio" in df.columns:
        df["spec_ratio_x_post"] = df["spec_vs_oper_ratio"] * df["post_2022"]

    for t in TOPICS:
        col = f"topic_{t}"
        if col in df.columns:
            df[f"{col}_x_post"] = df[col] * df["post_2022"]

    # centering spec_vs_oper_ratio
    if "spec_vs_oper_ratio" in df.columns:
        mu = df["spec_vs_oper_ratio"].mean()
        df["spec_vs_oper_ratio_c"] = df["spec_vs_oper_ratio"] - mu
        print(f"Centered spec_vs_oper_ratio (mean={mu:.4f})")

    # rd_intensity: imputing NaN to 0 if too many missing
    if "rd_intensity" in df.columns:
        n_nan = df["rd_intensity"].isna().sum()
        pct = 100 * n_nan / len(df)
        if pct > 30:
            df["rd_intensity"] = df["rd_intensity"].fillna(0)
            print(f"rd_intensity: {n_nan} NaN ({pct:.0f}%) -> imputed to 0")
        else:
            print(f"rd_intensity: {n_nan} NaN ({pct:.0f}%)")

    # missing values report
    key_cols = [c for c in [
        y_col, "spec_vs_oper_ratio", "speculative_score", "operational_score",
        "log_mktcap", "log_revenue", "op_margin", "rd_intensity", "rev_growth",
    ] + [f"topic_{t}" for t in TOPICS] if c and c in df.columns]

    print(f"\nMissing values:")
    has_any = False
    for col in key_cols:
        n = df[col].isna().sum()
        if n > 0:
            print(f"  {col}: {n} ({100*n/len(df):.1f}%)")
            has_any = True
    if not has_any:
        print("  None")

    # saving
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"  {df.shape[0]} rows x {df.shape[1]} cols, {df['ticker'].nunique()} firms")
    print(f"  Years: {sorted(df['year'].unique())}")

    summary = df[key_cols].describe().round(4)
    with open(SUMMARY_FILE, "w") as f:
        f.write(f"SUMMARY: panel_final.csv\n{'='*60}\n\n")
        f.write(f"Shape: {df.shape}\nFirms: {df['ticker'].nunique()}\n")
        f.write(f"Years: {sorted(df['year'].unique())}\n\n")
        f.write(summary.to_string())
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()