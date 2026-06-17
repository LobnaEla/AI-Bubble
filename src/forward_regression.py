import os
import sys
import pandas as pd

try:
    from linearmodels.panel import PanelOLS
except ImportError:
    sys.exit("linearmodels manquant -> pip install linearmodels")

# ---------- 1. Localiser et charger le panel ----------
def find_panel():
    for c in ["data/processed/panel_final.csv",
              "../data/processed/panel_final.csv",
              "panel_final.csv"]:
        if os.path.exists(c):
            return c
    sys.exit("panel_final.csv introuvable (cherche dans data/processed/).")

path = sys.argv[1] if len(sys.argv) > 1 else find_panel()
df = pd.read_csv(path)
print(f"Charge : {path}  ->  {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ---------- 2. Detecter colonnes firme / annee ----------
def pick(cols, candidates, label):
    for c in candidates:
        if c in cols:
            return c
    sys.exit(f"Colonne {label} introuvable. Colonnes dispo : {list(cols)}")

firm_col = pick(df.columns, ["ticker", "Ticker", "firm", "symbol", "company", "cik"], "firme")
year_col = pick(df.columns, ["year", "Year", "fiscal_year", "fy"], "annee")
print(f"Firme = '{firm_col}'   |   Annee = '{year_col}'")

# ---------- 3. Variables ----------
Y = "pe_premium"
CONTROLS = ["log_revenue", "log_mktcap", "op_margin", "rd_intensity", "rev_growth"]

def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

def run(name, regressors):
    cols = [Y] + regressors + [firm_col, year_col]
    d = df[cols].dropna().copy().set_index([firm_col, year_col])
    res = PanelOLS(d[Y], d[regressors],
                   entity_effects=True, time_effects=True,
                   drop_absorbed=True).fit(cov_type="robust")
    print("\n" + "=" * 66)
    print(f"{name}    (N={int(res.nobs)}, R2={res.rsquared:.4f})")
    print("=" * 66)
    rows = []
    for v in regressors:
        b, se, p = res.params[v], res.std_errors[v], res.pvalues[v]
        tag = "  <-- variable cle" if v in ("speculative_score", "spec_vs_oper_ratio") else ""
        print(f"  {v:<20} beta={b:>10.3f}   SE={se:>9.3f}   p={p:6.3f} {stars(p):<3}{tag}")
        rows.append({"spec": name, "var": v, "beta": round(b, 3),
                     "se": round(se, 3), "pval": round(p, 4), "sig": stars(p)})
    return rows

out = []
out += run("A. Forward-looking density (speculative_score seul)",
           ["speculative_score"] + CONTROLS)
out += run("B. Forward vs Operational density",
           ["speculative_score", "operational_score"] + CONTROLS)
out += run("C. Ratio (rappel = ton M1)",
           ["spec_vs_oper_ratio"] + CONTROLS)

# ---------- 4. Export ----------
os.makedirs("data/outputs", exist_ok=True)
dest = "data/outputs/forward_looking_test.csv"
pd.DataFrame(out).to_csv(dest, index=False, encoding="utf-8-sig")
print(f"\n[OK] Resultats exportes -> {dest}")