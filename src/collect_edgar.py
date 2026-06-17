import bisect
import json
import os
import re
import time
import warnings
import pandas as pd
from edgar import set_identity, Company

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# 1 ── LOAD CONFIG + FIRM LIST
# ══════════════════════════════════════════════════════════════════

print("=" * 65)
print("1 ── Loading config and firm list")
print("=" * 65)

with open("config.json") as f:
    cfg = json.load(f)

YEAR_START    = cfg["year_start"]
YEAR_END      = cfg["year_end"]
EDGAR_ID      = cfg["edgar_identity"]
CONTEXT_WIN   = cfg["nlp"]["context_window_words"]
AI_KW         = [k.lower() for k in cfg["nlp"]["ai_keywords"]]
SPEC_MK       = [m.lower() for m in cfg["nlp"]["speculative_markers"]]
OPER_MK       = [m.lower() for m in cfg["nlp"]["operational_markers"]]
EPSILON       = 1e-6
YEARS         = list(range(YEAR_START, YEAR_END + 1))

# Pre-split keywords into single-word (fast index lookup)
# and multi-word (text search + bisect)
SINGLE_KW = {kw for kw in AI_KW if len(kw.split()) == 1}
MULTI_KW  = [kw for kw in AI_KW if len(kw.split()) > 1]

firms   = pd.read_csv("data/firm_groups_all_scored.csv")
TICKERS = firms["ticker"].tolist()

print(f"  Input file     : data/firm_groups_all_scored.csv")
print(f"  Firms          : {len(TICKERS)}")
print(f"  Years          : {YEAR_START}–{YEAR_END}  ({len(YEARS)} years)")
print(f"  Total filings  : {len(TICKERS) * len(YEARS)}")
print(f"  AI keywords    : {len(AI_KW)}  "
      f"({len(SINGLE_KW)} single-word, {len(MULTI_KW)} multi-word)")
print(f"  Spec markers   : {len(SPEC_MK)}")
print(f"  Oper markers   : {len(OPER_MK)}")
print(f"  Context window : ±{CONTEXT_WIN} words")

set_identity(EDGAR_ID)

# ══════════════════════════════════════════════════════════════════
# 2 ── HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def clean_text(raw: str) -> str:
    """Remove HTML/XML tags and entities, collapse whitespace, lowercase."""
    text = re.sub(r"<[^>]+>",       " ", raw or "")
    text = re.sub(r"&[a-z0-9#]+;",  " ", text)
    text = re.sub(r"\s+",           " ", text).strip()
    return text.lower()


def tokenize(text: str) -> list:
    """Lowercase word tokens (letters + digits + hyphens)."""
    return re.findall(r"[a-z][a-z0-9\-]*", text)


def find_10k(company, fiscal_year: int):
    """
    Find the 10-K filed for fiscal_year.
    A FY20XX report is filed between Jul 20XX and Jun 20XX+1.
    Returns (filing, status_string).
    """
    try:
        filings = company.get_filings(form="10-K")
        if not filings:
            return None, "no_filings"
        for f in filings:
            try:
                ds = str(f.filing_date)
                yr = int(ds[:4])
                mo = int(ds[5:7])
                if (yr == fiscal_year     and mo >= 7) or \
                   (yr == fiscal_year + 1 and mo <= 6):
                    return f, "found"
            except Exception:
                continue
        return None, "year_not_found"
    except Exception as e:
        return None, f"error: {e}"


def extract_mda(filing) -> tuple:
    """
    Extract Item 7 (MD&A) text using edgartools 5.x correctly.

    Strategy 1 — TenK.management_discussion (HIGH quality)
      Uses the official edgartools 5.x property which internally chains:
      HTMLParser → ChunkedDocument → CrossReferenceIndex

    Strategy 2 — Regex on raw HTML (MEDIUM quality)
      Finds Item 7 / Item 8 boundaries in the raw HTML text.

    Strategy 3 — Full HTML text (LOW quality)
      Last resort. Scores will be computed on the full document,
      not just MD&A. Still usable if consistent across firms.

    Returns (cleaned_text, method_label).
    """

    # ── Strategy 1: edgartools TenK.management_discussion ─────────
    try:
        tenk = filing.obj()
        if tenk is not None:
            mda = tenk.management_discussion   # official Item 7 property
            if mda is not None:
                mda_str = str(mda).strip()
                if len(mda_str) > 500:
                    return clean_text(mda_str), "tenk_management_discussion"
    except Exception:
        pass

    # ── Strategy 2: regex boundary detection on raw HTML ──────────
    try:
        html = filing.html() or ""
        hl   = html.lower()

        start = -1
        for pat in [
            r"item\s+7[\.\s\-—]+management.{0,40}discussion",
            r"item\s+7[\.\s\-—]+md&a",
            r"item\s*7\b",
        ]:
            m = re.search(pat, hl)
            if m:
                start = m.start()
                break

        if start > 0:
            end = len(html)
            for pat in [
                r"item\s+7a[\.\s\-—]+quantitative",
                r"item\s+8[\.\s\-—]+financial\s+statements",
                r"item\s+8\b",
            ]:
                m = re.search(pat, hl[start + 200:])
                if m:
                    end = start + 200 + m.start()
                    break

            chunk   = clean_text(html[start:end])
            n_words = len(chunk.split())
            if n_words > 200:
                return chunk, "regex_html"
    except Exception:
        pass

    # ── Strategy 3: full HTML text (last resort) ──────────────────
    try:
        html    = filing.html() or ""
        cleaned = clean_text(html)
        if len(cleaned.split()) > 200:
            return cleaned, "full_text_fallback"
    except Exception:
        pass

    return "", "failed"


def score_mda(text: str, words: list) -> dict:
    """
    Compute NLP scores from the MD&A text.

    Uses two optimisations vs the naive approach:
      - Single-word keywords: O(1) lookup via pre-built word index
      - Multi-word keywords:  text.find() + bisect for word position
        (avoids re.findall on text[:idx] which was the bottleneck)

    Context check: for each AI keyword hit, look at ±CONTEXT_WIN words
    and check for speculative/operational markers.
    """
    n = len(words)
    if n == 0:
        return dict(
            mda_word_count=0, ai_keyword_hits=0, ai_density_per1k=0.0,
            speculative_hits=0, speculative_score=0.0,
            operational_hits=0, operational_score=0.0,
            spec_vs_oper_ratio=0.0,
        )

    # Build word → positions index (single pass, O(n))
    word_index = {}
    for i, w in enumerate(words):
        word_index.setdefault(w, []).append(i)

    # Build char offset → word index map for multi-word phrases (O(n))
    char_offsets = [m.start() for m in re.finditer(r"[a-z][a-z0-9\-]*", text)]

    ai_positions = []

    # Single-word keywords: direct index lookup
    for kw in SINGLE_KW:
        if kw in word_index:
            ai_positions.extend(word_index[kw])

    # Multi-word keywords: text search + bisect
    for kw in MULTI_KW:
        pos = 0
        while True:
            idx = text.find(kw, pos)
            if idx == -1:
                break
            word_idx = bisect.bisect_right(char_offsets, idx) - 1
            if word_idx >= 0:
                ai_positions.append(word_idx)
            pos = idx + len(kw)

    ai_hits   = len(ai_positions)
    spec_hits = 0
    oper_hits = 0

    for wi in ai_positions:
        lo      = max(0, wi - CONTEXT_WIN)
        hi      = min(n, wi + CONTEXT_WIN + 1)
        context = " ".join(words[lo:hi])
        if any(m in context for m in SPEC_MK):
            spec_hits += 1
        if any(m in context for m in OPER_MK):
            oper_hits += 1

    return dict(
        mda_word_count     = n,
        ai_keyword_hits    = ai_hits,
        ai_density_per1k   = round(ai_hits   / n * 1000, 4),
        speculative_hits   = spec_hits,
        speculative_score  = round(spec_hits / n * 1000, 4),
        operational_hits   = oper_hits,
        operational_score  = round(oper_hits / n * 1000, 4),
        spec_vs_oper_ratio = round(
            (spec_hits / n * 1000) / (oper_hits / n * 1000 + EPSILON), 4
        ),
    )


# ══════════════════════════════════════════════════════════════════
# 3 ── MAIN COLLECTION LOOP
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("2 ── Collecting and scoring 10-K filings from EDGAR")
print(f"     {len(TICKERS)} firms × {len(YEARS)} years = "
      f"{len(TICKERS) * len(YEARS)} filings")
print("=" * 65)

METHOD_QUALITY = {
    "tenk_management_discussion": "HIGH",
    "regex_html":                 "MEDIUM",
    "full_text_fallback":         "LOW",
    "failed":                     "FAILED",
}

records = []

for ticker in TICKERS:
    print(f"\n  [{ticker}]")

    try:
        company = Company(ticker)
    except Exception as e:
        print(f"    ✗ Company load error: {e}")
        for yr in YEARS:
            records.append({
                "ticker": ticker, "fiscal_year": yr,
                "filing_date": None, "accession_number": None,
                "extraction_method": "company_load_error",
                "data_quality": "FAILED",
                "mda_word_count": None, "ai_keyword_hits": None,
                "ai_density_per1k": None, "speculative_hits": None,
                "speculative_score": None, "operational_hits": None,
                "operational_score": None, "spec_vs_oper_ratio": None,
                "status": f"company_load_error: {e}",
            })
        continue

    for fiscal_year in YEARS:

        filing, fstatus = find_10k(company, fiscal_year)

        if filing is None:
            print(f"    {fiscal_year}  ✗  {fstatus}")
            records.append({
                "ticker": ticker, "fiscal_year": fiscal_year,
                "filing_date": None, "accession_number": None,
                "extraction_method": fstatus, "data_quality": "FAILED",
                "mda_word_count": None, "ai_keyword_hits": None,
                "ai_density_per1k": None, "speculative_hits": None,
                "speculative_score": None, "operational_hits": None,
                "operational_score": None, "spec_vs_oper_ratio": None,
                "status": "not_found",
            })
            time.sleep(0.3)
            continue

        mda_text, method = extract_mda(filing)
        quality          = METHOD_QUALITY.get(method, "UNKNOWN")

        if not mda_text:
            print(f"    {fiscal_year}  ✗  extraction failed")
            records.append({
                "ticker": ticker, "fiscal_year": fiscal_year,
                "filing_date":      str(filing.filing_date),
                "accession_number": str(filing.accession_no),
                "extraction_method": method, "data_quality": "FAILED",
                "mda_word_count": 0, "ai_keyword_hits": 0,
                "ai_density_per1k": None, "speculative_hits": 0,
                "speculative_score": None, "operational_hits": 0,
                "operational_score": None, "spec_vs_oper_ratio": None,
                "status": "extraction_failed",
            })
            time.sleep(0.5)
            continue

        words  = tokenize(mda_text)
        scores = score_mda(mda_text, words)

        print(f"    {fiscal_year}  ✓  "
              f"filed={filing.filing_date}  "
              f"words={scores['mda_word_count']:>7,}  "
              f"ai_hits={scores['ai_keyword_hits']:>4}  "
              f"density={scores['ai_density_per1k']:.3f}/1k  "
              f"spec={scores['speculative_score']:.3f}  "
              f"oper={scores['operational_score']:.3f}  "
              f"[{quality}]")

        records.append({
            "ticker":             ticker,
            "fiscal_year":        fiscal_year,
            "filing_date":        str(filing.filing_date),
            "accession_number":   str(filing.accession_no),
            "extraction_method":  method,
            "data_quality":       quality,
            **scores,
            "mda_text":           mda_text,
            "status": "ok",
        })

        time.sleep(0.5)

    time.sleep(0.5)

# ══════════════════════════════════════════════════════════════════
# 4 ── BUILD PANEL + MERGE FIRM METADATA
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("3 ── Building panel and merging firm metadata")
print("=" * 65)

panel = pd.DataFrame(records)

# Merge firm metadata from the input file
meta_cols = ["ticker", "name", "sector", "sub_industry",
             "group", "ai_density_2023"]
panel = panel.merge(firms[meta_cols], on="ticker", how="left")

col_order = [
    "ticker", "name", "sector", "sub_industry", "group",
    "fiscal_year", "filing_date", "accession_number",
    "extraction_method", "data_quality",
    "mda_word_count",
    "ai_keyword_hits",   "ai_density_per1k",
    "speculative_hits",  "speculative_score",
    "operational_hits",  "operational_score",
    "spec_vs_oper_ratio",
    "ai_density_2023",
    "mda_text",
    "status",
]
panel = panel[col_order]

# ══════════════════════════════════════════════════════════════════
# 5 ── SAVE + SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════

os.makedirs("data", exist_ok=True)
out = "data/edgar_panel.csv"
panel.to_csv(out, index=False)

ok     = panel[panel["status"] == "ok"]
failed = panel[panel["status"] != "ok"]

print(f"\n  Saved → {out}")
print(f"\n{'=' * 65}")
print("SUMMARY")
print(f"{'=' * 65}")
print(f"  Total observations   : {len(panel)}")
print(f"  Successful           : {len(ok)}  ({len(ok)/len(panel)*100:.1f}%)")
print(f"  Failed / not found   : {len(failed)}")

if not ok.empty:
    print(f"\n  Extraction quality breakdown:")
    print(ok["data_quality"].value_counts().to_string())

    print(f"\n  Mean ai_density_per1k by group × year:")
    try:
        pivot = ok.groupby(["group", "fiscal_year"])["ai_density_per1k"] \
                  .mean().round(3).unstack("fiscal_year")
        print(pivot.to_string())
    except Exception:
        pass

    print(f"\n  Mean spec_vs_oper_ratio by group × year:")
    try:
        pivot2 = ok.groupby(["group", "fiscal_year"])["spec_vs_oper_ratio"] \
                   .mean().round(3).unstack("fiscal_year")
        print(pivot2.to_string())
    except Exception:
        pass

if not failed.empty:
    print(f"\n  Failed filings by reason:")
    print(failed.groupby("status")["ticker"].count().to_string())
