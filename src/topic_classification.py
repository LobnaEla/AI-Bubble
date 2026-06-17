import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

DATA_DIR = Path(config.DATA_DIR)
RAW_DIR = Path(config.RAW_DIR)
MDA_DIR = RAW_DIR / "mda_texts"

for fname in ["panel_with_specificity.csv", "merged_panel_v2.csv", "merged_panel.csv"]:
    INPUT_FILE = DATA_DIR / fname
    if INPUT_FILE.exists():
        break

OUTPUT_FILE = DATA_DIR / "panel_with_topics.csv"

# AI sentence detection
AI_KEYWORDS = [
    r"artificial intelligence", r"machine learning", r"deep learning",
    r"neural network", r"large language model", r"generative ai",
    r"natural language processing", r"computer vision",
    r"ai[- ]powered", r"ai[- ]driven", r"ai[- ]enabled",
    r"\bai\b",
]
AI_PATTERN = re.compile("|".join(AI_KEYWORDS), re.IGNORECASE)

# Topics keywords
TOPIC_KEYWORDS = {
    "opportunity": [
        r"\bmarket\s+(?:opportunity|share|leadership|position)\b",
        r"\bcompetitive\s+(?:advantage|position|differentiat)\b",
        r"\bnew\s+(?:revenue|market|opportunit)\b",
        r"\brevenue\s+(?:stream|growth|opportunit)\b",
        r"\bgrowth\s+(?:opportunit|driver|potential)\b",
        r"\btransform(?:ation|ative|ing|s)?\b",
        r"\bdisrupt(?:ion|ive|ing|s)?\b",
        r"\bvalue\s+(?:creation|proposition|driver)\b",
        r"\bstrategic\s+(?:advantage|priority|initiative|differentiator)\b",
        r"\bnext[- ]gen(?:eration)?\b",
        r"\binnovat(?:ion|ive|ing)\b",
        r"\bunlock(?:ing)?\s+(?:value|growth|potential)\b",
        r"\bscal(?:e|able|ing|ability)\b",
        r"\bai[- ](?:first|native|driven\s+growth)\b",
        r"\blong[- ]term\s+(?:value|growth|vision)\b",
        r"\bexpand(?:ing)?\s+(?:market|opportunit|capabilit)\b",
        r"\bpotential\s+(?:of|for|to)\b",
        r"\bopportunity\b",
        r"\badvantage\b",
    ],
    "adoption": [
        r"\bdeploy(?:ed|ing|ment)?\b",
        r"\bintegrat(?:ed|ing|ion)\b",
        r"\blaunch(?:ed|ing)?\b",
        r"\bimplement(?:ed|ing|ation)\b",
        r"\broll(?:ed|ing)?\s+out\b",
        r"\bin\s+production\b",
        r"\bin\s+use\b",
        r"\buse\s+(?:of\s+)?(?:ai|machine\s+learning|ml)\b",
        r"\busing\s+(?:ai|ml|machine\s+learning)\b",
        r"\bpowered\s+by\b",
        r"\bai[- ](?:powered|enabled|driven)\b",
        r"\b(?:ai|ml)\s+(?:solution|platform|tool|system|product|feature|model|agent)\b",
        r"\b(?:solution|platform|tool|system|product|feature)\s+(?:that\s+uses?\s+)?(?:ai|ml)\b",
        r"\badopt(?:ed|ion|ing)\b",
        r"\bpilot(?:ed|ing|s)?\b",
        r"\bembedd(?:ed|ing)\b",
        r"\bcustomer[- ]facing\b",
        r"\bproduction\s+(?:system|environment|deployment)\b",
        r"\blive\b",
        r"\boperational\b",
    ],
    "laborsaving": [
        r"\bautomat(?:e|ed|ing|ion|es)\b",
        r"\bproductivity\b",
        r"\befficienc(?:y|ies)\b",
        r"\bcost\s+(?:saving|reduction|cutting|optimization)\b",
        r"\breduc(?:e|ing|tion)\s+(?:cost|headcount|labor|workforce|manual)\b",
        r"\bheadcount\b",
        r"\bworkforce\s+(?:reduction|optimization|transformation)\b",
        r"\bjob\s+(?:cut|eliminat|reduc)\b",
        r"\bstreamlin(?:e|ed|ing)\b",
        r"\boperational\s+(?:efficienc|improvement|saving)\b",
        r"\blabor\s+(?:cost|saving|productiv)\b",
        r"\brepetitive\s+(?:task|work|process)\b",
        r"\bmanual\s+(?:process|task|work)\b",
        r"\bhuman[- ]in[- ]the[- ]loop\b",
        r"\breplac(?:e|ing|ement)\s+(?:human|worker|staff|employee)\b",
        r"\bFTE\b",
        r"\bresource\s+(?:optim|saving|efficienc)\b",
    ],
    "rd_investment": [
        r"\binvest(?:ing|ment|ed)\b",
        r"\bcapex\b",
        r"\bcapital\s+(?:expenditure|allocation|investment)\b",
        r"\bdata\s+cent(?:er|re)\b",
        r"\bGPU\b", r"\bTPU\b", r"\bcompute\b",
        r"\bcloud\s+(?:infrastructure|computing|platform|service)\b",
        r"\binfrastructure\b",
        r"\bresearch\s+(?:and\s+development|investment|spending)\b",
        r"\bR&D\b",
        r"\bfoundation\s+model\b",
        r"\bpre[- ]?train(?:ed|ing)?\b",
        r"\bfine[- ]?tun(?:e|ed|ing)\b",
        r"\bmodel\s+(?:train|develop|build)\b",
        r"\bai\s+(?:chip|hardware|server|cluster)\b",
        r"\bsemiconductor\b",
        r"\bpartnered?\s+with\s+(?:openai|google|microsoft|nvidia|anthropic|amazon|meta)\b",
        r"\blicens(?:e|ed|ing)\s+(?:from|with)\b",
        r"\bspend(?:ing)?\s+(?:on\s+)?(?:ai|technology|tech|compute)\b",
        r"\bbuild(?:ing)?\s+(?:our\s+)?(?:ai|ml|platform|model|system)\b",
        r"\bdevelop(?:ing|ment)\s+(?:of\s+)?(?:ai|ml|model)\b",
    ],
}

TOPIC_PATTERNS = {
    t: re.compile("|".join(kws), re.IGNORECASE)
    for t, kws in TOPIC_KEYWORDS.items()
}


def sentence_tokenize(text):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 20]


def classify_topics(text):
    if not text or not isinstance(text, str) or len(text) < 50:
        return {f"topic_{t}": 0.0 for t in TOPIC_KEYWORDS}

    sents = sentence_tokenize(text)
    wc = max(len(text.split()), 1)
    counts = defaultdict(int)

    for i, sent in enumerate(sents):
        if not AI_PATTERN.search(sent):
            continue

        # building context window : prev + current + next
        window_parts = [sent]
        if i > 0:
            window_parts.append(sents[i-1])
        if i < len(sents) - 1:
            window_parts.append(sents[i+1])
        window = " ".join(window_parts)

        for topic, pat in TOPIC_PATTERNS.items():
            if pat.search(window):
                counts[topic] += 1

    return {
        f"topic_{t}": round((counts[t] / wc) * 1000, 6)
        for t in TOPIC_KEYWORDS
    }


def add_dominant_topic(df):
    tcols = [f"topic_{t}" for t in TOPIC_KEYWORDS]
    totals = df[tcols].fillna(0).sum(axis=1)
    df["topic_dominant"] = df[tcols].fillna(0).idxmax(axis=1).str.replace("topic_", "")
    df.loc[totals == 0, "topic_dominant"] = "none"
    return df


def main():
    print("=" * 60)
    print("LEVEL 3 — 4-topic Classification (Ca'Zorzi et al. strategy)")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)
    print(f"Input: {INPUT_FILE.name}  ({df.shape[0]} rows, {df.shape[1]} cols)")

    has_mda = MDA_DIR.exists() and any(MDA_DIR.glob("*.txt"))
    has_col = "mda_text" in df.columns and df["mda_text"].notna().sum() > 0

    if has_mda:
        print(f"Using .txt files — {len(list(MDA_DIR.glob('*.txt')))} found")
    elif has_col:
        print(f"Using mda_text column — {df['mda_text'].notna().sum()} texts")
    else:
        print("No text available — can't classify topics")
        return

    results = []
    found, missing = 0, 0

    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        year = str(int(row.get("year", row.get("fiscal_year", 0))))
        text = None

        if has_mda:
            p = MDA_DIR / f"{ticker}_{year}.txt"
            if p.exists():
                text = p.read_text(encoding="utf-8", errors="ignore")
                found += 1
            else:
                missing += 1
        elif has_col:
            val = row.get("mda_text")
            if pd.notna(val) and str(val).strip():
                text = str(val)
                found += 1
            else:
                missing += 1

        scores = classify_topics(text or "")
        scores["ticker"] = ticker
        scores["year"] = int(year)
        results.append(scores)

    print(f"  Processed: {found}  |  Missing: {missing}")

    # replacing old topic columns if they exist
    tcols = [f"topic_{t}" for t in TOPIC_KEYWORDS]
    old = [c for c in tcols + ["topic_dominant"] if c in df.columns]
    if old:
        df = df.drop(columns=old)

    scores_df = pd.DataFrame(results)
    df = df.merge(scores_df[["ticker", "year"] + tcols],
                  on=["ticker", "year"], how="left")
    df = add_dominant_topic(df)

    # stats
    print(f"\nMean score per topic:")
    for col in tcols:
        nz = (df[col] > 0).sum()
        print(f"  {col:<28s}  mean={df[col].mean():.4f}  nonzero={nz}/{len(df)}")

    print(f"\nDominant topic distribution:")
    print(df["topic_dominant"].value_counts().to_string())

    print(f"\nDominant topic by year:")
    print(pd.crosstab(df["year"], df["topic_dominant"]))

    # pre vs post ChatGPT comparison
    if "post_2022" in df.columns:
        print(f"\nPre vs post ChatGPT:")
        comp = df.groupby("post_2022")[tcols].mean().round(4)
        comp.index = comp.index.map({0: "pre-2023", 1: "post-2022"})
        print(comp.to_string())

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}  ({df.shape})")


if __name__ == "__main__":
    main()