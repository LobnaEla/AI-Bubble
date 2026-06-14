"""
regression_final.py — Panel regression complète + robustesse + stats descriptives
Projet AI Bubble (Youssef Chebil & Lobna Elabed)
"""

import os
import warnings
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

# ── Chemins ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

INPUT_FILE = os.path.join(PARENT_DIR, "data", "processed", "panel_final.csv")
OUTPUT_DIR = os.path.join(PARENT_DIR, "data", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(INPUT_FILE):
    INPUT_FILE = os.path.join(BASE_DIR, "panel_final.csv")

# ── Chargement & préparation ───────────────────────────────────────────────────
print("=" * 65)
print("RÉGRESSION PANEL — AI Narrative & P/E Premium")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)
print(f"\nInput  : {INPUT_FILE}")
print(f"Lignes : {len(df)} | Firmes : {df['ticker'].nunique()} | Années : {sorted(df['year'].unique())}")

# Calculer pe_premium_avg (robustesse) = pe_avg - médiane sectorielle x année
df["sector_pe_avg"]  = df.groupby(["sector", "year"])["pe_avg"].transform("median")
df["pe_premium_avg"] = df["pe_avg"] - df["sector_pe_avg"]

# ── Variables ──────────────────────────────────────────────────────────────────
Y        = "pe_premium"
Y_ROB    = "pe_premium_avg"
MAIN_VAR = "spec_vs_oper_ratio"
INTER    = "spec_ratio_x_post"
TOPICS   = ["topic_opportunity", "topic_adoption", "topic_laborsaving", "topic_rd_investment"]
TOPICS_X = ["topic_opportunity_x_post", "topic_adoption_x_post",
            "topic_laborsaving_x_post", "topic_rd_investment_x_post"]
CONTROLS = ["log_revenue", "log_mktcap", "op_margin", "rd_intensity", "rev_growth"]
FE       = "EntityEffects + TimeEffects"

LABEL = {
    "spec_vs_oper_ratio":          "Spec/Oper ratio (β₁)",
    "spec_ratio_x_post":           "Spec/Oper × post2022 (β₆)",
    "topic_opportunity":           "Opportunity",
    "topic_adoption":              "Adoption",
    "topic_laborsaving":           "Labor-saving",
    "topic_rd_investment":         "R&D/Investment",
    "topic_opportunity_x_post":    "Opportunity × post2022",
    "topic_adoption_x_post":       "Adoption × post2022",
    "topic_laborsaving_x_post":    "Labor-saving × post2022",
    "topic_rd_investment_x_post":  "R&D/Investment × post2022",
    "log_revenue":                 "Log(Revenue)",
    "log_mktcap":                  "Log(Market Cap)",
    "op_margin":                   "Operating Margin (%)",
    "rd_intensity":                "R&D Intensity (%)",
    "rev_growth":                  "Revenue Growth (%)",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_coef(res, var):
    try:
        p  = res.params[var]
        se = res.std_errors[var]
        pv = res.pvalues[var]
        stars = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.1 else ""
        return f"{p:.3f}{stars}", f"({se:.3f})", pv
    except KeyError:
        return "—", "", None

def run_model(name, formula, data, y_var):
    d = data.dropna(subset=[y_var] + [c for c in CONTROLS if c in formula])
    d = d.set_index(["ticker", "year"])
    mod = PanelOLS.from_formula(formula, data=d)
    res = mod.fit(cov_type="robust")
    print(f"\n{'─'*65}")
    print(f"  {name}")
    print(f"{'─'*65}")
    print(f"  N={res.nobs} | Firmes={res.entity_info['total']} | "
          f"R²={res.rsquared:.4f} | R²-within={res.rsquared_within:.4f}")
    print(res.summary.tables[1])
    return res


# ══════════════════════════════════════════════════════════════════════════════
# 1. STATISTIQUES DESCRIPTIVES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STATISTIQUES DESCRIPTIVES")
print("=" * 65)

desc_vars = {
    "P/E premium (Y principal)":     "pe_premium",
    "P/E premium avg (robustesse)":  "pe_premium_avg",
    "Spec/Oper ratio":               "spec_vs_oper_ratio",
    "N phrases AI":                  "n_ai_sentences",
    "AI density (/1000 mots)":       "ai_density_per1k",
    "Specificity score":             "specificity_score",
    "Topic: Opportunity":            "topic_opportunity",
    "Topic: Adoption":               "topic_adoption",
    "Topic: Labor-saving":           "topic_laborsaving",
    "Topic: R&D/Investment":         "topic_rd_investment",
    "Log(Revenue)":                  "log_revenue",
    "Log(Market Cap)":               "log_mktcap",
    "Operating Margin (%)":          "op_margin",
    "R&D Intensity (%)":             "rd_intensity",
    "Revenue Growth (%)":            "rev_growth",
}

rows_desc = []
for label, col in desc_vars.items():
    s = df[col].dropna()
    rows_desc.append({
        "Variable":  label,
        "N":         int(s.count()),
        "Mean":      round(s.mean(), 4),
        "Std":       round(s.std(), 4),
        "Min":       round(s.min(), 4),
        "P25":       round(s.quantile(0.25), 4),
        "Median":    round(s.median(), 4),
        "P75":       round(s.quantile(0.75), 4),
        "Max":       round(s.max(), 4),
    })

desc_df = pd.DataFrame(rows_desc)
print(desc_df.to_string(index=False))

desc_path = os.path.join(OUTPUT_DIR, "descriptive_stats.csv")
desc_df.to_csv(desc_path, index=False, encoding="utf-8")
print(f"\n=> Exporte : {desc_path}")

# Stats pre/post ChatGPT
print("\n--- Evolution pre vs post ChatGPT (2023) ---")
pre  = df[df["year"] <= 2022]
post = df[df["year"] >  2022]
for col, label in [("n_ai_sentences", "N phrases AI (mean)"),
                   ("spec_vs_oper_ratio", "Spec/Oper ratio (mean)"),
                   ("topic_opportunity", "Topic Opportunity (mean)"),
                   ("topic_adoption", "Topic Adoption (mean)"),
                   ("topic_rd_investment", "Topic R&D/Inv (mean)"),
                   ("pe_premium", "P/E premium (mean)")]:
    print(f"  {label:35s} | Pre: {pre[col].mean():.4f} | Post: {post[col].mean():.4f} | "
          f"x{post[col].mean()/pre[col].mean():.2f}" if pre[col].mean() != 0 else "  N/A")


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODÈLES PRINCIPAUX (pe_premium)
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("MODÈLES PRINCIPAUX — Y = pe_premium (roic.ai)")
print("=" * 65)

base = df.copy()

f_m1 = f"{Y} ~ {MAIN_VAR} + " + " + ".join(CONTROLS) + f" + {FE}"
f_m2 = f"{Y} ~ {MAIN_VAR} + {INTER} + " + " + ".join(CONTROLS) + f" + {FE}"
f_m3 = f"{Y} ~ " + " + ".join(TOPICS) + " + " + " + ".join(CONTROLS) + f" + {FE}"
f_m4 = f"{Y} ~ " + " + ".join(TOPICS) + " + " + " + ".join(TOPICS_X) + " + " + " + ".join(CONTROLS) + f" + {FE}"

res_m1 = run_model("M1 — Narratif AI → P/E (base)", f_m1, base, Y)
res_m2 = run_model("M2 — Signal bulle post-ChatGPT (β6)", f_m2, base, Y)
res_m3 = run_model("M3 — 4 Topics Ca'Zorzi", f_m3, base, Y)
res_m4 = run_model("M4 — Topics × post_2022", f_m4, base, Y)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROBUSTESSE — Y = pe_premium_avg (prix moyen au lieu de prix fin d'année)
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("ROBUSTESSE — Y = pe_premium_avg (prix moyen annuel)")
print("=" * 65)

f_r1 = f"{Y_ROB} ~ {MAIN_VAR} + " + " + ".join(CONTROLS) + f" + {FE}"
f_r2 = f"{Y_ROB} ~ {MAIN_VAR} + {INTER} + " + " + ".join(CONTROLS) + f" + {FE}"
f_r3 = f"{Y_ROB} ~ " + " + ".join(TOPICS) + " + " + " + ".join(CONTROLS) + f" + {FE}"
f_r4 = f"{Y_ROB} ~ " + " + ".join(TOPICS) + " + " + " + ".join(TOPICS_X) + " + " + " + ".join(CONTROLS) + f" + {FE}"

res_r1 = run_model("R1 — Robustesse M1 (pe_premium_avg)", f_r1, base, Y_ROB)
res_r2 = run_model("R2 — Robustesse M2 (pe_premium_avg)", f_r2, base, Y_ROB)
res_r3 = run_model("R3 — Robustesse M3 (pe_premium_avg)", f_r3, base, Y_ROB)
res_r4 = run_model("R4 — Robustesse M4 (pe_premium_avg)", f_r4, base, Y_ROB)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TABLEAU RÉCAPITULATIF COMPLET
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("TABLEAU RÉCAPITULATIF — Principaux + Robustesse")
print("=" * 65)

all_models = {
    "M1": res_m1, "M2": res_m2, "M3": res_m3, "M4": res_m4,
    "R1": res_r1, "R2": res_r2, "R3": res_r3, "R4": res_r4,
}

vars_order = [MAIN_VAR, INTER] + TOPICS + TOPICS_X + CONTROLS

rows_tab = []
for v in vars_order:
    row = {"Variable": LABEL.get(v, v)}
    for mname, res in all_models.items():
        coef, se, pv = get_coef(res, v)
        row[mname] = f"{coef} {se}".strip()
    rows_tab.append(row)

# Stats
for stat, vals in [
    ("Y",        [Y, Y, Y, Y, Y_ROB, Y_ROB, Y_ROB, Y_ROB]),
    ("N",        [r.nobs for r in all_models.values()]),
    ("Firmes",   [r.entity_info["total"] for r in all_models.values()]),
    ("R²",       [f"{r.rsquared:.4f}" for r in all_models.values()]),
    ("R²-within",[f"{r.rsquared_within:.4f}" for r in all_models.values()]),
    ("Firm FE",  ["Yes"] * 8),
    ("Year FE",  ["Yes"] * 8),
    ("SE",       ["Robust HC"] * 8),
]:
    row = {"Variable": stat}
    for mname, val in zip(all_models.keys(), vals):
        row[mname] = val
    rows_tab.append(row)

tab_df = pd.DataFrame(rows_tab)
print(tab_df.to_string(index=False))

# Export CSV
tab_path = os.path.join(OUTPUT_DIR, "regression_table_full.csv")
tab_df.to_csv(tab_path, index=False, encoding="utf-8")
print(f"\n=> Exporte : {tab_path}")

# Export LaTeX
tex_path = os.path.join(OUTPUT_DIR, "regression_table_full.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write("\\caption{AI Narrative and P/E Premium — Panel OLS Results}\n")
    f.write("\\label{tab:regression_full}\n\\small\n")
    f.write("\\begin{tabular}{lcccc|cccc}\n")
    f.write("\\hline\\hline\n")
    f.write(" & \\multicolumn{4}{c|}{Main results ($Y = pe\\_premium$)} & "
            "\\multicolumn{4}{c}{Robustness ($Y = pe\\_premium\\_avg$)} \\\\\n")
    f.write(" & M1 & M2 & M3 & M4 & R1 & R2 & R3 & R4 \\\\\n")
    f.write("\\hline\n")

    sections = [
        ("Main variable",       [MAIN_VAR, INTER]),
        ("4 Topics (Ca'Zorzi)", TOPICS),
        ("Topics × post-2022",  TOPICS_X),
        ("Controls",            CONTROLS),
    ]
    tex_label = {k: v.replace("&", "\\&").replace("%", "\\%").replace("β₁", "$\\beta_1$").replace("β₆", "$\\beta_6$")
                 for k, v in LABEL.items()}

    for section, svars in sections:
        f.write(f"\\multicolumn{{9}}{{l}}{{\\textit{{{section}}}}}\\\\\n")
        for v in svars:
            label = tex_label.get(v, v)
            coefs = []
            ses   = []
            for res in all_models.values():
                c, s, _ = get_coef(res, v)
                coefs.append(c)
                ses.append(s)
            coef_row = " & ".join(coefs)
            se_row   = " & ".join(ses)
            f.write(f"\\quad {label} & {coef_row} \\\\\n")
            f.write(f"  & {se_row} \\\\\n")

    f.write("\\hline\n")
    f.write("$Y$ & " + " & ".join(["$pe\\_prem.$"] * 4 + ["$pe\\_prem.\\_avg$"] * 4) + " \\\\\n")
    f.write("N & " + " & ".join(str(r.nobs) for r in all_models.values()) + " \\\\\n")
    f.write("$R^2$ & " + " & ".join(f"{r.rsquared:.4f}" for r in all_models.values()) + " \\\\\n")
    f.write("Firm FE & " + " & ".join(["Yes"] * 8) + " \\\\\n")
    f.write("Year FE & " + " & ".join(["Yes"] * 8) + " \\\\\n")
    f.write("SE & \\multicolumn{8}{c}{Robust HC} \\\\\n")
    f.write("\\hline\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\begin{tablenotes}\\small\n")
    f.write("\\item *** p$<$0.01, ** p$<$0.05, * p$<$0.10. Standard errors in parentheses.\n")
    f.write("\\item $pe\\_premium$ = firm P/E minus sector-year median P/E (roic.ai).\n")
    f.write("\\item $pe\\_premium\\_avg$ uses annual average price instead of year-end price.\n")
    f.write("\\item All models include firm and year fixed effects with robust HC standard errors.\n")
    f.write("\\end{tablenotes}\n\\end{table}\n")

print(f"=> Exporte : {tex_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. INTERPRÉTATION COMPLÈTE
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("INTERPRÉTATION — RÉSULTATS PRINCIPAUX")
print("=" * 65)

def interpret(res, var, label):
    try:
        p  = res.params[var]
        pv = res.pvalues[var]
        se = res.std_errors[var]
        if pv < 0.01:   sig = "*** TRES significatif (p<0.01)"
        elif pv < 0.05: sig = "**  Significatif (p<0.05)"
        elif pv < 0.10: sig = "*   Marginalement sig. (p<0.10)"
        else:           sig = "    NON significatif"
        direction = "POSITIF" if p > 0 else "negatif"
        print(f"  {label}")
        print(f"    beta={p:.3f} | SE={se:.3f} | p={pv:.4f} | {direction} | {sig}")
    except KeyError:
        pass

print("\n[M1] Narratif AI global gonfle-t-il le P/E ?")
interpret(res_m1, MAIN_VAR, "spec_vs_oper_ratio")

print("\n[M2] L'effet s'est-il amplifie apres ChatGPT ?")
interpret(res_m2, INTER, "spec_ratio_x_post")

print("\n[M3] Quel type de discours drive le P/E premium ?")
for t in TOPICS:
    interpret(res_m3, t, t)

print("\n[M4] Lequel s'amplifie post-ChatGPT ?")
for t in TOPICS_X:
    interpret(res_m4, t, t)

print("\n[Robustesse R3 — confirmation M3 avec pe_premium_avg ?]")
for t in TOPICS:
    interpret(res_r3, t, t)

print("\n" + "=" * 65)
print("FICHIERS PRODUITS")
print("=" * 65)
print(f"  descriptive_stats.csv        -> stats descriptives")
print(f"  regression_table_full.csv    -> tableau complet (principaux + robustesse)")
print(f"  regression_table_full.tex    -> version LaTeX pour rapport")
print("=" * 65)
