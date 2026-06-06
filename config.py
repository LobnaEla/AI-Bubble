import os

# Root of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR   = os.path.join(ROOT_DIR, "data", "processed")
RAW_DIR    = os.path.join(ROOT_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "outputs")

for d in [DATA_DIR, RAW_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Paramètres ───────────────────────────────────────────────────────────────

# Année de screening : utilisée pour calculer ai_density_2021
# (niveau AI de base avant ChatGPT → variable de contrôle dans la régression)
SCREENING_YEAR = 2021

# Secteurs S&P 500 à inclure dans l'étude
FOCUS_SECTORS = [
    "Information Technology",
    "Communication Services",
    "Financials",
    "Health Care",
    "Industrials",
    "Energy",
]

# Mots-clés pour compter les mentions AI dans les 10-K
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