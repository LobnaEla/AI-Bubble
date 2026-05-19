import os

# Root of the project (one level up from src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR       = os.path.join(ROOT_DIR, "data", "processed")
RAW_DIR        = os.path.join(ROOT_DIR, "data", "raw")
OUTPUT_DIR     = os.path.join(ROOT_DIR, "outputs")
FIGURES_DIR    = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR     = os.path.join(OUTPUT_DIR, "tables")

# Create all directories if they don't exist
for d in [DATA_DIR, RAW_DIR, FIGURES_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Parameters ──────────────────────────────────────────────────
SCREENING_YEAR = 2021      # year used to classify firms (pre-ChatGPT)

FOCUS_SECTORS = [
    "Information Technology",
    "Communication Services",
    "Financials",
    "Health Care",
    "Industrials",
    "Energy",
]

SECTOR_SELECTION = {
    "Information Technology":  {"treatment": 8, "control": 8},
    "Communication Services":  {"treatment": 5, "control": 5},
    "Financials":              {"treatment": 4, "control": 4},
    "Health Care":             {"treatment": 4, "control": 4},
    "Industrials":             {"treatment": 3, "control": 3},
    "Energy":                  {"treatment": 3, "control": 3},
}

AI_SCREENING_KEYWORDS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "large language model",
    "generative ai",
    "natural language processing",
    "computer vision",
    "ai-powered",
    "ai-driven",
    "ai-enabled",
    " ai ",
]