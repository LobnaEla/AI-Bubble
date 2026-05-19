DATA_DIR = "./data/"

# Année utilisée pour classifier les firms
# Si tu veux garder ton choix actuel, mets 2023.
# Si tu veux une sélection plus propre pré-hype, mets 2021 ou moyenne 2018-2022.
SCREENING_YEAR = 2023

FOCUS_SECTORS = [
    "Information Technology",
    "Communication Services",
    "Financials",
    "Health Care",
    "Industrials",
    "Energy",
]

# Nombre de treatment/control à prendre dans chaque secteur
# Tu peux ajuster selon les résultats disponibles.
SECTOR_SELECTION = {
    "Information Technology": {
        "treatment": 8,
        "control": 8,
    },
    "Communication Services": {
        "treatment": 5,
        "control": 5,
    },
    "Financials": {
        "treatment": 4,
        "control": 4,
    },
    "Health Care": {
        "treatment": 4,
        "control": 4,
    },
    "Industrials": {
        "treatment": 4,
        "control": 4,
    },
    "Energy": {
        "treatment": 2,
        "control": 2,
    },
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