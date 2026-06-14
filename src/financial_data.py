import sys
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

API_KEY    = config.ROIC_API_KEY
BASE_URL   = "https://api.roic.ai/v2/fundamental"
DATA_DIR   = Path(config.DATA_DIR)
RAW_DIR    = Path(config.RAW_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Source des firmes : firm_groups_all_scored (toutes les firmes scorées)
FIRMS_FILE = RAW_DIR / "firm_groups_all_scored.csv"
if not FIRMS_FILE.exists():
    FIRMS_FILE = DATA_DIR / "firm_groups_all_scored.csv"

OUTPUT_FILE = DATA_DIR / "financial_data.csv"

FISCAL_YEAR_START = 2018
FISCAL_YEAR_END   = 2026
SLEEP_BETWEEN     = 0.3


def api_get(endpoint: str, ticker: str, **extra_params) -> list:
    url    = f"{BASE_URL}/{endpoint}/{ticker}"
    params = {
        "apikey"           : API_KEY,
        "period"           : "annual",
        "fiscal_year_start": FISCAL_YEAR_START,
        "fiscal_year_end"  : FISCAL_YEAR_END,
        "order"            : "ASC",
        "limit"            : 20,
    }
    params.update(extra_params)
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 401:
            print("\n❌ Clé API invalide — vérifier config.ROIC_API_KEY")
            sys.exit(1)
        elif r.status_code == 429:
            print("\n⏳ Rate limit — pause 10s...")
            time.sleep(10)
            return api_get(endpoint, ticker, **extra_params)
        else:
            return []
    except Exception:
        return []


def fetch_income(ticker: str) -> pd.DataFrame:
    data = api_get("income-statement", ticker)
    if not data:
        return pd.DataFrame()
    rows = []
    for d in data:
        try:
            year = int(d.get("fiscal_year", 0))
        except (ValueError, TypeError):
            continue
        if not (FISCAL_YEAR_START <= year <= FISCAL_YEAR_END):
            continue
        rows.append({
            "ticker"    : ticker,
            "year"      : year,
            "revenue"   : d.get("is_sales_revenue_turnover"),
            "ebit"      : d.get("is_oper_income"),
            "rd"        : d.get("is_operating_expenses_r_and_d"),
            "eps"       : d.get("diluted_eps") or d.get("eps"),
            "net_income": d.get("is_net_income"),
            "op_margin" : d.get("oper_margin"),
        })
    return pd.DataFrame(rows)


def fetch_multiples(ticker: str) -> pd.DataFrame:
    data = api_get("multiples", ticker)
    if not data:
        return pd.DataFrame()
    rows = []
    for d in data:
        try:
            year = int(d.get("fiscal_year", 0))
        except (ValueError, TypeError):
            continue
        if not (FISCAL_YEAR_START <= year <= FISCAL_YEAR_END):
            continue
        price  = d.get("pr_last")
        shares = d.get("bs_sh_out")
        ev     = d.get("enterprise_value")
        mktcap = (price * shares) if (price and shares) else ev
        rows.append({
            "ticker" : ticker,
            "year"   : year,
            "pe"     : d.get("pe_ratio"),
            "pe_avg" : d.get("average_price_earnings_ratio"),
            "mktcap" : mktcap,
            "ev"     : ev,
            "price"  : price,
        })
    return pd.DataFrame(rows)


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "year"])
    df["rd_intensity"] = np.where(
        df["revenue"].notna() & (df["revenue"] > 0) & df["rd"].notna(),
        (df["rd"] / df["revenue"]) * 100, np.nan
    )
    df["rev_growth"]  = df.groupby("ticker")["revenue"].pct_change() * 100
    df["log_revenue"] = np.log(df["revenue"].clip(lower=1))
    df["log_mktcap"]  = np.log(df["mktcap"].clip(lower=1))
    df["post_2022"]   = (df["year"] > 2022).astype(int)
    return df


def compute_pe_premium(df: pd.DataFrame) -> pd.DataFrame:
    sector_med = (
        df.dropna(subset=["pe"])
        .groupby(["sector", "year"])["pe"]
        .median()
        .reset_index()
        .rename(columns={"pe": "sector_pe"})
    )
    df = df.merge(sector_med, on=["sector", "year"], how="left")
    df["pe_premium"] = df["pe"] - df["sector_pe"]
    return df


def main():
    print("=" * 65)
    print("FINANCIAL DATA — roic.ai API")
    print(f"Années : {FISCAL_YEAR_START}–{FISCAL_YEAR_END}")
    print("=" * 65)

    if not API_KEY:
        print("❌ ROIC_API_KEY vide dans config.py")
        return

    firms   = pd.read_csv(FIRMS_FILE)
    tickers = firms["ticker"].unique()
    print(f"Source : {FIRMS_FILE.name}  ({len(tickers)} firmes)\n")

    income_rows, multiple_rows, errors = [], [], []

    for i, ticker in enumerate(tickers):
        print(f"[{i+1:>3}/{len(tickers)}] {ticker:<6}", end="  ", flush=True)

        inc  = fetch_income(ticker);   time.sleep(SLEEP_BETWEEN)
        mult = fetch_multiples(ticker); time.sleep(SLEEP_BETWEEN)

        if inc.empty and mult.empty:
            print("⚠  aucune donnée")
            errors.append(ticker)
            continue

        print(f"income={len(inc) if not inc.empty else 0} ans  "
              f"multiples={len(mult) if not mult.empty else 0} ans")

        if not inc.empty:  income_rows.append(inc)
        if not mult.empty: multiple_rows.append(mult)

    if not income_rows and not multiple_rows:
        print("❌ Aucune donnée récupérée — vérifier la clé API")
        return

    df_inc  = pd.concat(income_rows,   ignore_index=True) if income_rows  else pd.DataFrame()
    df_mult = pd.concat(multiple_rows, ignore_index=True) if multiple_rows else pd.DataFrame()

    if not df_inc.empty and not df_mult.empty:
        df = df_inc.merge(df_mult, on=["ticker", "year"], how="outer")
    else:
        df = df_inc if not df_inc.empty else df_mult

    meta_cols = [c for c in ["ticker", "sector", "group"] if c in firms.columns]
    meta = firms[meta_cols].drop_duplicates("ticker")
    df   = df.merge(meta, on="ticker", how="left")
    df   = compute_derived(df)
    df   = compute_pe_premium(df)

    if "pe" in df.columns and df["pe"].notna().sum() > 0:
        cap   = df["pe"].quantile(0.99)
        n_cap = (df["pe"] > cap).sum()
        df["pe"]         = df["pe"].clip(upper=cap)
        df["pe_premium"] = df["pe_premium"].clip(
            lower=df["pe_premium"].quantile(0.01),
            upper=df["pe_premium"].quantile(0.99)
        )
        print(f"\nWinsorisation P/E : {n_cap} valeurs plafonnées à {cap:.1f}")

    print(f"\n{'─'*65}")
    print(f"Firmes OK   : {df['ticker'].nunique()} / {len(tickers)}")
    print(f"Erreurs     : {len(errors)}  {errors if errors else ''}")
    print(f"Firm-years  : {len(df)}")
    print(f"Années      : {sorted(df['year'].unique())}")
    print(f"P/E valides : {df['pe'].notna().sum()} / {len(df)}")
    pe_var = df.groupby("ticker")["pe"].nunique()
    print(f"P/E variable dans le temps : {(pe_var > 1).sum()} / {df['ticker'].nunique()} firmes ✓")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Sauvegardé : {OUTPUT_FILE}  {df.shape}")
    print(f"\nProchaine étape : python src/merge_data.py")


if __name__ == "__main__":
    main()