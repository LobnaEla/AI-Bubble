"""Quick ticker overlap check for the processed CSV files."""

from pathlib import Path

import pandas as pd

try:
	import config

	DATA_DIR = Path(config.DATA_DIR)
except Exception:
	DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"


EDGAR_FILE = DATA_DIR / "edgar_panel_fixed.csv"
FINANCIAL_FILE = DATA_DIR / "financial_data.csv"


def load_tickers(path: Path) -> set[str]:
	df = pd.read_csv(path)
	if "ticker" not in df.columns:
		raise KeyError(f"Colonne 'ticker' introuvable dans {path.name}")
	return set(df["ticker"].dropna().astype(str).str.strip().str.upper().unique())


def main() -> None:
	edgar_tickers = load_tickers(EDGAR_FILE)
	financial_tickers = load_tickers(FINANCIAL_FILE)

	common_tickers = edgar_tickers & financial_tickers

	print(f"edgar_panel_fixed.csv : {len(edgar_tickers)} tickers uniques")
	print(f"financial_data.csv    : {len(financial_tickers)} tickers uniques")
	print(f"Communs               : {len(common_tickers)} tickers")


if __name__ == "__main__":
	main()
