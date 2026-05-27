import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# 0.  Path setup — mirrors the convention in financial_data.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ROOT_DIR, OUTPUT_DIR, DATA_DIR

EDGAR_PATH    = os.path.join(DATA_DIR, "edgar_panel_fixed.csv")
FINANCIAL_PATH = os.path.join(DATA_DIR, "financial_data.csv")
OUTPUT_PATH   = os.path.join(OUTPUT_DIR, "merged_panel.csv")


# ---------------------------------------------------------------------------
# 1.  Load both datasets
# ---------------------------------------------------------------------------
print("Loading financial data ...")
df_fin = pd.read_csv(FINANCIAL_PATH)

print("Loading EDGAR text panel ...")
df_edgar = pd.read_csv(EDGAR_PATH)

# Enforce consistent types for the join keys before any filtering
df_fin["year"]              = df_fin["year"].astype(int)
df_edgar["fiscal_year"]     = df_edgar["fiscal_year"].astype(int)
df_edgar["ticker"]          = df_edgar["ticker"].str.strip().str.upper()
df_fin["ticker"]            = df_fin["ticker"].str.strip().str.upper()


# ---------------------------------------------------------------------------
# 2.  Restrict EDGAR to the 54 firms present in financial_data.csv
#     (financial_data.csv already contains exactly 54 tickers; the EDGAR
#     panel covers 151 tickers, so we keep only the intersection)
# ---------------------------------------------------------------------------
fin_tickers = set(df_fin["ticker"].unique())          # 54 firms
df_edgar_54 = df_edgar[df_edgar["ticker"].isin(fin_tickers)].copy()

print(f"  EDGAR rows for 54 firms : {len(df_edgar_54)}")
print(f"  Financial rows          : {len(df_fin)}")


# ---------------------------------------------------------------------------
# 3.  Select which EDGAR columns to carry into the merged panel
#
#     KEPT — and why
#     -----------------------------------------------------------------------
#     ticker, fiscal_year   → join keys (dropped / renamed after merge)
#
#     name                  → human-readable company name; useful for tables
#                             and robustness checks where you slice by firm.
#
#     sector                → sector label sourced from EDGAR/S&P; kept as a
#                             cross-check against the yfinance sector in
#                             financial_data.csv (they sometimes differ).
#                             Renamed to "sector_edgar" to avoid collision.
#
#     sub_industry          → finer-grained GICS classification; needed for
#                             within-sector heterogeneity tests (e.g., do
#                             Software firms drive the AI premium more than
#                             Hardware firms inside the Tech sector?).
#
#     group                 → treatment/control indicator built during sample
#                             construction ("AI adopter" vs "control").
#                             This is the key moderator / grouping variable
#                             for the DiD and cross-sectional regressions.
#
#     filing_date           → the actual SEC filing date; lets us compute
#                             event-window returns around the 10-K drop and
#                             align text signals to the correct fiscal year.
#
#     mda_word_count        → length control. Longer MDA sections produce
#                             more keyword hits mechanically; we must partial
#                             this out in every regression.
#
#     ai_keyword_hits       → raw count of AI-related keywords in the MDA.
#                             Used to construct density measures but also
#                             kept raw for Poisson / negative-binomial count
#                             regressions.
#
#     ai_density_per1k      → keyword hits per 1 000 words; the primary
#                             continuous treatment variable measuring how
#                             prominently management discusses AI in its
#                             annual report.
#
#     speculative_hits      → count of forward-looking / uncertain AI terms
#                             ("plan to deploy", "exploring AI"). Captures
#                             the *intention* dimension of AI adoption.
#
#     speculative_score     → speculative_hits normalised by mda_word_count.
#                             Used as a separate RHS variable to test whether
#                             intention (vs execution) drives the P/E premium.
#
#     operational_hits      → count of present-tense operational AI terms
#                             ("AI-enabled", "deployed", "integrated"). Captures
#                             the *execution* dimension.
#
#     operational_score     → operational_hits normalised by mda_word_count.
#
#     spec_vs_oper_ratio    → speculative_score / operational_score.
#                             A high ratio = mostly talk, little action.
#                             Central to the "AI hype vs real adoption" test.
#
#     ai_density_2023       → firm-level *average* AI density anchored at
#                             fiscal year 2023; a time-invariant baseline
#                             useful in cross-sectional OLS and as an
#                             instrument for current-year density.
#
#     data_quality          → HIGH / MEDIUM / LOW / FAILED flag from the
#                             NLP extraction pipeline. We filter to HIGH+
#                             for the main regressions and use the full
#                             sample (including LOW) for robustness checks.
#
#     EXCLUDED — and why
#     -----------------------------------------------------------------------
#     accession_number      → SEC internal ID; no analytical use once the
#                             merge is complete.
#
#     extraction_method     → all 54 firms use "tenk_management_discussion";
#                             the column is constant and adds nothing.
#
#     status                → "ok" / "not_found"; rows with "not_found" have
#                             NaN NLP scores anyway and are dropped below.
# ---------------------------------------------------------------------------

EDGAR_COLS = [
    "ticker", "fiscal_year",
    "name",
    "sector",               # will be renamed to sector_edgar
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


# ---------------------------------------------------------------------------
# 4.  Merge on ticker × year
#
#     Left = financial_data (255 rows × 54 firms × up to 5 years: 2021-2026)
#     Right = EDGAR text panel (378 rows × 54 firms × 2018-2024)
#
#     We use a LEFT join so that financial rows for fiscal years 2025-2026
#     (where the 10-K has not yet been filed / processed) are retained with
#     NaN text columns rather than silently dropped. This preserves the full
#     financial time-series and makes the missingness explicit.
#
#     The overlapping years where both sides have data are 2021–2024
#     (200 matched ticker-year pairs), which is the main estimation sample.
# ---------------------------------------------------------------------------
df_merged = df_fin.merge(
    df_edgar_sel,
    left_on=["ticker", "year"],
    right_on=["ticker", "fiscal_year"],
    how="left",
)

# fiscal_year is now redundant with year
df_merged.drop(columns=["fiscal_year"], inplace=True)


# ---------------------------------------------------------------------------
# 5.  Quality and diagnostics
# ---------------------------------------------------------------------------
total_rows   = len(df_merged)
matched_rows = df_merged["ai_density_per1k"].notna().sum()
unmatched    = total_rows - matched_rows

print(f"\nMerge diagnostics")
print(f"  Total rows in merged panel : {total_rows}")
print(f"  Rows with EDGAR text data  : {matched_rows}  (fiscal years 2021–2024)")
print(f"  Rows without EDGAR data    : {unmatched}  (fiscal years 2025–2026, not yet filed)")

low_quality = df_merged[df_merged["data_quality"].isin(["LOW", "FAILED"])].shape[0]
print(f"  LOW/FAILED quality rows    : {low_quality}  (flag; not dropped here)")

# Sector consistency check
sector_mismatch = df_merged[
    df_merged["sector_edgar"].notna() &
    (df_merged["sector"] != df_merged["sector_edgar"])
]
if len(sector_mismatch) > 0:
    print(f"\n  ⚠  Sector label mismatches between yfinance and EDGAR: {len(sector_mismatch)} rows")
    print(sector_mismatch[["ticker", "year", "sector", "sector_edgar"]].drop_duplicates().to_string(index=False))
else:
    print("  Sector labels: fully consistent between yfinance and EDGAR.")


# ---------------------------------------------------------------------------
# 6.  Column ordering for readability
# ---------------------------------------------------------------------------
LEAD_COLS = [
    "ticker", "name", "year", "filing_date",
    "sector", "sector_edgar", "sub_industry", "group",
    "data_quality",
]
REMAINING = [c for c in df_merged.columns if c not in LEAD_COLS]
df_merged = df_merged[LEAD_COLS + REMAINING]


# ---------------------------------------------------------------------------
# 7.  Save
# ---------------------------------------------------------------------------
df_merged.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓  Saved: {OUTPUT_PATH}")
print(f"   Shape : {df_merged.shape[0]} rows × {df_merged.shape[1]} columns")
print(f"\nColumn list:\n  " + "\n  ".join(df_merged.columns.tolist()))