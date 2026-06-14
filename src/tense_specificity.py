import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

DATA_DIR = Path(config.DATA_DIR)
RAW_DIR  = Path(config.RAW_DIR)
MDA_DIR  = RAW_DIR / "mda_texts"

# Input : prendre le fichier le plus avancé disponible
for fname in ["merged_panel_v2.csv", "merged_panel.csv"]:
    INPUT_FILE = DATA_DIR / fname
    if INPUT_FILE.exists():
        break

OUTPUT_FILE = DATA_DIR / "panel_with_specificity.csv"

# ── Patterns AI ───────────────────────────────────────────────────────────────
AI_KEYWORDS = [
    r"artificial intelligence", r"machine learning", r"deep learning",
    r"neural network", r"large language model", r"generative ai",
    r"natural language processing", r"computer vision",
    r"ai[- ]powered", r"ai[- ]driven", r"ai[- ]enabled",
    r"\bai\b",
]
AI_PATTERN = re.compile("|".join(AI_KEYWORDS), re.IGNORECASE)

# ── Niveau 1 : forward-looking (spéculatif) ───────────────────────────────────
FUTURE_MARKERS = [
    r"\bwill\b", r"\bplan(?:s|ned)?\s+to\b", r"\bintend(?:s|ed)?\s+to\b",
    r"\bexpect(?:s|ed)?\s+to\b", r"\baim(?:s|ed)?\s+to\b",
    r"\bgoing\s+to\b", r"\bseek(?:s|ing)?\s+to\b",
    r"\bcommit(?:ted|s)?\s+to\b", r"\bprioritiz(?:e|ing)\b",
    r"\bwould\b", r"\bcould\b", r"\bmay\b", r"\bmight\b",
    r"\bfuture\b", r"\bupcoming\b", r"\bnext\s+(?:year|quarter|fiscal)\b",
    r"\bforecast\b", r"\bprojected?\b", r"\bprospect(?:s|ive)?\b",
]
FUTURE_PATTERN = re.compile("|".join(FUTURE_MARKERS), re.IGNORECASE)

# ── Niveau 1 : past/present (opérationnel) ────────────────────────────────────
PAST_MARKERS = [
    r"\bdeployed?\b", r"\bintegrated?\b", r"\blaunched?\b",
    r"\bimplemented?\b", r"\brolled?\s+out\b", r"\bin\s+production\b",
    r"\bwe\s+use\b", r"\bwe\s+have\b", r"\bwe\s+developed\b",
    r"\bwe\s+built\b", r"\bwe\s+invested\b", r"\bwe\s+achieved\b",
    r"\bcurrently\b", r"\balready\b", r"\btoday\b",
    r"\bhas\s+been\b", r"\bhave\s+been\b",
    r"\bwas\s+(?:deployed|launched|implemented)\b",
]
PAST_PATTERN = re.compile("|".join(PAST_MARKERS), re.IGNORECASE)

# ── Niveau 2 : spécificité ────────────────────────────────────────────────────
DOLLAR_PATTERN = re.compile(
    r"\$[\d,\.]+\s*(?:million|billion|M|B|bn|trillion)?\b"
    r"|\b\d[\d,\.]*\s*(?:million|billion|trillion)\s*dollars?\b",
    re.IGNORECASE
)
PCT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent(?:age)?)\b", re.IGNORECASE
)
TIMEFRAME_PATTERN = re.compile(
    r"\bQ[1-4]\s*20\d{2}\b"
    r"|\bby\s+20\d{2}\b"
    r"|\bwithin\s+\d+\s+(?:months?|years?|quarters?)\b"
    r"|\bnext\s+\d+\s+(?:months?|years?|quarters?)\b"
    r"|\bend\s+of\s+(?:fiscal\s+)?20\d{2}\b"
    r"|\bH[12]\s+20\d{2}\b",
    re.IGNORECASE
)
NAMED_ENTITY_PATTERN = re.compile(
    r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3}\b"
)


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions
# ─────────────────────────────────────────────────────────────────────────────

def sentence_tokenize(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def score_specificity(sentence: str) -> int:
    score  = len(DOLLAR_PATTERN.findall(sentence))
    score += len(PCT_PATTERN.findall(sentence))
    score += len(TIMEFRAME_PATTERN.findall(sentence))
    score += min(len(NAMED_ENTITY_PATTERN.findall(sentence)), 3)
    return score


def compute_ratio(n_spec: int, n_oper: int) -> float:
    """
    spec_vs_oper_ratio robuste — proportion lissée.
    (n_spec + 1) / (n_spec + n_oper + 2)

    Interprétation :
      0.5  = autant de spec que d'oper (neutre)
      > 0.5 = plus de promesses que de réalisations
      < 0.5 = plus de réalisations que de promesses

    Jamais 0, jamais infini, toujours dans [0, 1].
    """
    return round((n_spec + 1) / (n_spec + n_oper + 2), 6)


def classify_text(text: str, word_count_override: int = None) -> dict:
    """Analyse NLP complète d'un texte MD&A."""
    if not text or not isinstance(text, str) or len(text) < 50:
        return {
            "n_ai_sentences"    : 0,
            "n_speculative"     : 0,
            "n_operational"     : 0,
            "speculative_score" : 0.0,
            "operational_score" : 0.0,
            "spec_vs_oper_ratio": compute_ratio(0, 0),
            "specificity_sum"   : 0,
            "specificity_mean"  : 0.0,
            "specificity_score" : 0.0,
        }

    sentences  = sentence_tokenize(text)
    word_count = word_count_override or max(len(text.split()), 1)

    ai_sents     = [s for s in sentences if AI_PATTERN.search(s)]
    future_sents = [s for s in ai_sents if FUTURE_PATTERN.search(s)]
    past_sents   = [s for s in ai_sents if not FUTURE_PATTERN.search(s)]

    n_spec = len(future_sents)
    n_oper = len(past_sents)

    spec_scores = [score_specificity(s) for s in future_sents]
    spec_sum    = sum(spec_scores)
    spec_mean   = float(np.mean(spec_scores)) if spec_scores else 0.0

    return {
        "n_ai_sentences"    : len(ai_sents),
        "n_speculative"     : n_spec,
        "n_operational"     : n_oper,
        "speculative_score" : round((n_spec / word_count) * 1000, 6),
        "operational_score" : round((n_oper / word_count) * 1000, 6),
        "spec_vs_oper_ratio": compute_ratio(n_spec, n_oper),
        "specificity_sum"   : spec_sum,
        "specificity_mean"  : round(spec_mean, 4),
        "specificity_score" : round((spec_sum / word_count) * 1000, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("NIVEAU 1 & 2 — Tense Detection + Specificity Scoring")
    print("=" * 65)

    df = pd.read_csv(INPUT_FILE)
    print(f"Input  : {INPUT_FILE.name}  ({df.shape[0]} rows)")

    # ── Choisir le mode ───────────────────────────────────────────────────────
    has_txt  = MDA_DIR.exists() and any(MDA_DIR.glob("*.txt"))
    has_text_col = "mda_text" in df.columns and df["mda_text"].notna().sum() > 0

    if has_txt:
        mode = "A_files"
        print(f"Mode A (fichiers .txt) — {len(list(MDA_DIR.glob('*.txt')))} fichiers trouvés")
    elif has_text_col:
        mode = "A_column"
        n_texts = df["mda_text"].notna().sum()
        print(f"Mode A (colonne mda_text) — {n_texts} textes disponibles dans le CSV")
    else:
        mode = "B"
        print("Mode B — Aucun texte disponible")
        print("  ⚠  Lancer extract_mda_texts.py d'abord pour de meilleurs résultats")

    # ── Calculer les scores ───────────────────────────────────────────────────
    results = []
    found, missing = 0, 0

    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        year   = str(int(row.get("year", row.get("fiscal_year", 0))))
        text   = None

        if mode == "A_files":
            txt_path = MDA_DIR / f"{ticker}_{year}.txt"
            if txt_path.exists():
                text  = txt_path.read_text(encoding="utf-8", errors="ignore")
                found += 1
            else:
                missing += 1

        elif mode == "A_column":
            val = row.get("mda_text")
            if pd.notna(val) and str(val).strip():
                text  = str(val)
                found += 1
            else:
                missing += 1

        scores = classify_text(text)
        scores["ticker"] = ticker
        scores["year"]   = int(year)
        results.append(scores)

    if mode != "B":
        print(f"  Textes traités : {found}  |  Manquants : {missing}")

    # ── Merger avec le panel ──────────────────────────────────────────────────
    NLP_COLS = ["n_ai_sentences", "n_speculative", "n_operational",
                "speculative_score", "operational_score", "spec_vs_oper_ratio",
                "specificity_sum", "specificity_mean", "specificity_score"]

    if mode == "B":
        # Pas de texte : recalculer spec_vs_oper_ratio depuis les hits existants
        print("\nMode B — Recalcul spec_vs_oper_ratio depuis les colonnes existantes")
        if "speculative_hits" in df.columns and "operational_hits" in df.columns:
            df["spec_vs_oper_ratio"] = df.apply(
                lambda r: compute_ratio(
                    int(r["speculative_hits"] or 0),
                    int(r["operational_hits"] or 0)
                ), axis=1
            )
            print(f"  spec_vs_oper_ratio recalculé ✓")
            print(f"  Avant : max = infini")
            print(f"  Après : max = {df['spec_vs_oper_ratio'].max():.4f}  "
                  f"(borné dans [0,1])")
        # Ajouter colonnes vides
        for col in ["n_ai_sentences", "n_speculative", "n_operational",
                    "specificity_score", "specificity_mean"]:
            if col not in df.columns:
                df[col] = np.nan
    else:
        # Mode A : merger les scores calculés
        scores_df = pd.DataFrame(results)

        # Supprimer les anciennes colonnes NLP pour les remplacer
        cols_to_drop = [c for c in NLP_COLS if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        df = df.merge(
            scores_df[["ticker", "year"] + NLP_COLS],
            on=["ticker", "year"], how="left"
        )

    # ── Stats finales ─────────────────────────────────────────────────────────
    print(f"\nStats spec_vs_oper_ratio :")
    print(df["spec_vs_oper_ratio"].describe().round(4).to_string())

    n_above_half = (df["spec_vs_oper_ratio"] > 0.5).sum()
    print(f"\nFirmes-années avec ratio > 0.5 (plus de promesses que réalisations) :"
          f" {n_above_half} / {df['spec_vs_oper_ratio'].notna().sum()}")

    if mode != "B":
        print(f"\nStats specificity_score :")
        print(df["specificity_score"].describe().round(6).to_string())

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Sauvegardé : {OUTPUT_FILE}")
    print(f"  Shape : {df.shape}")
    print(f"\nProchaine étape : python src/topic_classification.py")


if __name__ == "__main__":
    main()