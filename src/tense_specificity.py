import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

DATA_DIR = Path(config.DATA_DIR)
RAW_DIR = Path(config.RAW_DIR)
MDA_DIR = RAW_DIR / "mda_texts"

for fname in ["merged_panel_v2.csv", "merged_panel.csv"]:
    INPUT_FILE = DATA_DIR / fname
    if INPUT_FILE.exists():
        break

OUTPUT_FILE = DATA_DIR / "panel_with_specificity.csv"

# AI detection patterns

AI_KEYWORDS = [
    r"artificial intelligence", r"machine learning", r"deep learning",
    r"neural network", r"large language model", r"generative ai",
    r"natural language processing", r"computer vision",
    r"ai[- ]powered", r"ai[- ]driven", r"ai[- ]enabled",
    r"\bai\b",
]
AI_PATTERN = re.compile("|".join(AI_KEYWORDS), re.IGNORECASE)

# Level 1: forward-looking markers
FUTURE_MARKERS = [
    r"\bwill\b", r"\bplan(?:s|ned)?\s+to\b", r"\bintend(?:s|ed)?\s+to\b",
    r"\bexpect(?:s|ed)?\s+to\b", r"\baim(?:s|ed)?\s+to\b",
    r"\bgoing\s+to\b", r"\bseek(?:s|ing)?\s+to\b",
    r"\bcommit(?:ted|s)?\s+to\b", r"\bprioritiz(?:e|ing)\b",
    r"\bwould\b", r"\bcould\b", r"\bmay\b", r"\bmight\b",
    r"\bfuture\b", r"\bupcoming\b", r"\bnext\s+(?:year|quarter|fiscal)\b",
    r"\bforecast\b", r"\bprojected?\b", r"\bprospect(?:s|ive)?\b",
]
FUTURE_PAT = re.compile("|".join(FUTURE_MARKERS), re.IGNORECASE)

# Level 1: past/present markers
PAST_MARKERS = [
    r"\bdeployed?\b", r"\bintegrated?\b", r"\blaunched?\b",
    r"\bimplemented?\b", r"\brolled?\s+out\b", r"\bin\s+production\b",
    r"\bwe\s+use\b", r"\bwe\s+have\b", r"\bwe\s+developed\b",
    r"\bwe\s+built\b", r"\bwe\s+invested\b", r"\bwe\s+achieved\b",
    r"\bcurrently\b", r"\balready\b", r"\btoday\b",
    r"\bhas\s+been\b", r"\bhave\s+been\b",
    r"\bwas\s+(?:deployed|launched|implemented)\b",
]
PAST_PAT = re.compile("|".join(PAST_MARKERS), re.IGNORECASE)

# Level 2: specificity 
DOLLAR_PAT = re.compile(
    r"\$[\d,\.]+\s*(?:million|billion|M|B|bn|trillion)?\b"
    r"|\b\d[\d,\.]*\s*(?:million|billion|trillion)\s*dollars?\b",
    re.IGNORECASE)

PCT_PAT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent(?:age)?)\b", re.IGNORECASE)

TIMEFRAME_PAT = re.compile(
    r"\bQ[1-4]\s*20\d{2}\b"
    r"|\bby\s+20\d{2}\b"
    r"|\bwithin\s+\d+\s+(?:months?|years?|quarters?)\b"
    r"|\bnext\s+\d+\s+(?:months?|years?|quarters?)\b"
    r"|\bend\s+of\s+(?:fiscal\s+)?20\d{2}\b"
    r"|\bH[12]\s+20\d{2}\b",
    re.IGNORECASE)

NAMED_ENT_PAT = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3}\b")


def sentence_tokenize(text):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 20]


def score_specificity(sent):
    s = len(DOLLAR_PAT.findall(sent))
    s += len(PCT_PAT.findall(sent))
    s += len(TIMEFRAME_PAT.findall(sent))
    s += min(len(NAMED_ENT_PAT.findall(sent)), 3)
    return s


def compute_ratio(n_spec, n_oper):
    return round((n_spec + 1) / (n_spec + n_oper + 2), 6)


def classify_text(text, word_count_override=None):
    empty = {
        "n_ai_sentences": 0, "n_speculative": 0, "n_operational": 0,
        "speculative_score": 0.0, "operational_score": 0.0,
        "spec_vs_oper_ratio": compute_ratio(0, 0),
        "specificity_sum": 0, "specificity_mean": 0.0, "specificity_score": 0.0,
    }
    if not text or not isinstance(text, str) or len(text) < 50:
        return empty

    sents = sentence_tokenize(text)
    wc = word_count_override or max(len(text.split()), 1)

    ai_sents = [s for s in sents if AI_PATTERN.search(s)]
    future_sents = [s for s in ai_sents if FUTURE_PAT.search(s)]
    past_sents = [s for s in ai_sents if not FUTURE_PAT.search(s)]

    n_spec = len(future_sents)
    n_oper = len(past_sents)

    spec_scores = [score_specificity(s) for s in future_sents]
    ssum = sum(spec_scores)
    smean = float(np.mean(spec_scores)) if spec_scores else 0.0

    return {
        "n_ai_sentences": len(ai_sents),
        "n_speculative": n_spec,
        "n_operational": n_oper,
        "speculative_score": round((n_spec / wc) * 1000, 6),
        "operational_score": round((n_oper / wc) * 1000, 6),
        "spec_vs_oper_ratio": compute_ratio(n_spec, n_oper),
        "specificity_sum": ssum,
        "specificity_mean": round(smean, 4),
        "specificity_score": round((ssum / wc) * 1000, 6),
    }


def main():
    print("=" * 60)
    print("LEVEL 1 & 2 : Tense Detection + Specificity Scoring")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)
    print(f"Input: {INPUT_FILE.name}  ({df.shape[0]} rows)")

    # figuring out where the mda texts are
    has_txt = MDA_DIR.exists() and any(MDA_DIR.glob("*.txt"))
    has_text_col = "mda_text" in df.columns and df["mda_text"].notna().sum() > 0

    if has_txt:
        mode = "files"
        print(f"Using .txt files: {len(list(MDA_DIR.glob('*.txt')))} found")
    elif has_text_col:
        mode = "column"
        print(f"Using mda_text column — {df['mda_text'].notna().sum()} texts")
    else:
        mode = "fallback"
        print("No MDA texts available, will recalculate from existing columns")

    # compute scores
    results = []
    found = 0
    missing = 0

    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        year = str(int(row.get("year", row.get("fiscal_year", 0))))
        text = None

        if mode == "files":
            p = MDA_DIR / f"{ticker}_{year}.txt"
            if p.exists():
                text = p.read_text(encoding="utf-8", errors="ignore")
                found += 1
            else:
                missing += 1
        elif mode == "column":
            val = row.get("mda_text")
            if pd.notna(val) and str(val).strip():
                text = str(val)
                found += 1
            else:
                missing += 1

        scores = classify_text(text)
        scores["ticker"] = ticker
        scores["year"] = int(year)
        results.append(scores)

    if mode != "fallback":
        print(f"  Processed: {found}  |  Missing: {missing}")

    # merging back into the panel
    NLP_COLS = [
        "n_ai_sentences", "n_speculative", "n_operational",
        "speculative_score", "operational_score", "spec_vs_oper_ratio",
        "specificity_sum", "specificity_mean", "specificity_score",
    ]

    if mode == "fallback":
        # no raw text: just fix the ratio from existing hit counts
        print("\nFallback — recalculating spec_vs_oper_ratio from existing columns")
        if "speculative_hits" in df.columns and "operational_hits" in df.columns:
            df["spec_vs_oper_ratio"] = df.apply(
                lambda r: compute_ratio(int(r["speculative_hits"] or 0),
                                        int(r["operational_hits"] or 0)),
                axis=1)
            print(f"  Done, max = {df['spec_vs_oper_ratio'].max():.4f} (bounded [0,1])")
        for c in ["n_ai_sentences", "n_speculative", "n_operational",
                   "specificity_score", "specificity_mean"]:
            if c not in df.columns:
                df[c] = np.nan
    else:
        scores_df = pd.DataFrame(results)
        # drop old NLP cols and replace
        old_cols = [c for c in NLP_COLS if c in df.columns]
        df = df.drop(columns=old_cols)
        df = df.merge(scores_df[["ticker", "year"] + NLP_COLS],
                      on=["ticker", "year"], how="left")

    print(f"\nspec_vs_oper_ratio stats:")
    print(df["spec_vs_oper_ratio"].describe().round(4).to_string())

    above = (df["spec_vs_oper_ratio"] > 0.5).sum()
    total = df["spec_vs_oper_ratio"].notna().sum()
    print(f"\nFirm-years with ratio > 0.5: {above}/{total}")

    if mode != "fallback":
        print(f"\nspecificity_score stats:")
        print(df["specificity_score"].describe().round(6).to_string())

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}  ({df.shape})")
    print(f"Next: python src/topic_classification.py")


if __name__ == "__main__":
    main()