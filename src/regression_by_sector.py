import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as mticker
from linearmodels.panel import PanelOLS
from linearmodels.panel import PooledOLS

warnings.filterwarnings("ignore")

# ── Chemins ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

INPUT_FILE = os.path.join(PARENT_DIR, "data", "processed", "panel_final.csv")
OUTPUT_DIR = os.path.join(PARENT_DIR, "data", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(INPUT_FILE):
    INPUT_FILE = os.path.join(BASE_DIR, "panel_final.csv")

# ── Données ────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=['pe_premium'])

TOPICS   = ['topic_opportunity','topic_adoption','topic_laborsaving','topic_rd_investment']
CONTROLS = ['log_revenue','log_mktcap','op_margin','rd_intensity','rev_growth']
TOPICS_X = ['topic_opportunity_x_post','topic_adoption_x_post',
            'topic_laborsaving_x_post','topic_rd_investment_x_post']

TOPIC_LABELS = {
    'topic_opportunity':   'Opportunity',
    'topic_adoption':      'Adoption',
    'topic_laborsaving':   'Labor-saving',
    'topic_rd_investment': 'R&D/Invest.',
}

SECTOR_SHORT = {
    'Information Technology':  'IT',
    'Health Care':             'Health',
    'Financials':              'Financials',
    'Communication Services':  'Comm.',
    'Industrials':             'Industrials',
    'Energy':                  'Energy',
}

SECTOR_COLORS = {
    'Information Technology':  '#2e6da4',
    'Health Care':             '#1a7a4a',
    'Financials':              '#d35400',
    'Communication Services':  '#8e44ad',
    'Industrials':             '#c0392b',
    'Energy':                  '#f39c12',
}

def stars(pv):
    if pv is None: return ''
    return '***' if pv < 0.01 else '**' if pv < 0.05 else '*' if pv < 0.10 else ''

def get_coef(res, var):
    try:
        return res.params[var], res.pvalues[var], res.std_errors[var]
    except:
        return np.nan, np.nan, np.nan

# ══════════════════════════════════════════════════════════════════════════════
# 1. M3 PAR SECTEUR
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("RÉGRESSION M3 PAR SECTEUR")
print("=" * 65)

results = {}  # stocker tous les résultats

for sector in df['sector'].unique():
    d = df[df['sector'] == sector].dropna(subset=CONTROLS).copy()
    n_firms = d['ticker'].nunique()
    n_obs   = len(d)

    print(f"\n{'─'*60}")
    print(f"  {sector}  (N={n_obs}, firmes={n_firms})")
    print(f"{'─'*60}")

    if n_obs < 20 or n_firms < 3:
        print("  ⚠ Trop peu d'observations — résultats non fiables, affichés à titre indicatif")

    try:
        d_idx = d.set_index(['ticker','year'])

        # Si pas assez de variation temporelle → PooledOLS
        if n_firms < 4:
            formula = f"pe_premium ~ {' + '.join(TOPICS+CONTROLS)} + 1"
            mod = PooledOLS.from_formula(formula, data=d_idx)
            method = "PooledOLS (peu de firmes)"
        else:
            formula = f"pe_premium ~ {' + '.join(TOPICS+CONTROLS)} + EntityEffects + TimeEffects"
            mod = PanelOLS.from_formula(formula, data=d_idx)
            method = "PanelOLS + FE"

        res = mod.fit(cov_type='robust')
        results[sector] = res

        print(f"  Methode : {method}")
        print(f"  R² = {res.rsquared:.4f}")
        print(f"\n  {'Topic':<25} {'Beta':>8} {'SE':>8} {'p-value':>8} {'Sig':>5}")
        print(f"  {'-'*56}")
        for t in TOPICS:
            b, p, se = get_coef(res, t)
            flag = '  ← SIGNIFICATIF' if not np.isnan(p) and p < 0.10 else ''
            print(f"  {TOPIC_LABELS[t]:<25} {b:>8.2f} {se:>8.2f} {p:>8.4f} {stars(p):>5}{flag}")

    except Exception as e:
        print(f"  ERREUR : {e}")
        results[sector] = None

# ══════════════════════════════════════════════════════════════════════════════
# 2. PRE vs POST ChatGPT PAR SECTEUR (M3 sur chaque periode)
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("PRE vs POST ChatGPT PAR SECTEUR")
print("=" * 65)

results_pre  = {}
results_post = {}

for sector in df['sector'].unique():
    for period, label, res_dict in [
        (df['year'] <= 2022, 'Pre-2023',  results_pre),
        (df['year'] >  2022, 'Post-2022', results_post),
    ]:
        d = df[(df['sector'] == sector) & period].dropna(subset=CONTROLS).copy()
        if len(d) < 15 or d['ticker'].nunique() < 3:
            res_dict[sector] = None
            continue
        try:
            d_idx = d.set_index(['ticker','year'])
            formula = f"pe_premium ~ {' + '.join(TOPICS+CONTROLS)} + EntityEffects + TimeEffects"
            res = PanelOLS.from_formula(formula, data=d_idx).fit(cov_type='robust')
            res_dict[sector] = res
        except:
            res_dict[sector] = None

for sector in df['sector'].unique():
    print(f"\n  {sector}")
    for t in TOPICS:
        pre  = results_pre.get(sector)
        post = results_post.get(sector)
        b_pre,  p_pre,  _ = get_coef(pre,  t) if pre  else (np.nan, np.nan, np.nan)
        b_post, p_post, _ = get_coef(post, t) if post else (np.nan, np.nan, np.nan)
        if not np.isnan(p_pre) and p_pre < 0.10 or not np.isnan(p_post) and p_post < 0.10:
            print(f"    {TOPIC_LABELS[t]:<20} Pre: {b_pre:>7.2f}{stars(p_pre):<3}  Post: {b_post:>7.2f}{stars(p_post):<3}  ← signal")

# ══════════════════════════════════════════════════════════════════════════════
# 3. TABLEAU RÉCAPITULATIF
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("TABLEAU RÉCAPITULATIF — Beta par secteur et topic")
print("=" * 65)

sectors_valid = [s for s in df['sector'].unique() if results.get(s) is not None]

header = f"{'Topic':<22}" + "".join(f"{SECTOR_SHORT[s]:>14}" for s in sectors_valid)
print(header)
print("─" * (22 + 14 * len(sectors_valid)))

for t in TOPICS:
    row = f"{TOPIC_LABELS[t]:<22}"
    for s in sectors_valid:
        res = results[s]
        b, p, _ = get_coef(res, t)
        if np.isnan(b):
            row += f"{'n/a':>14}"
        else:
            cell = f"{b:.1f}{stars(p)}"
            row += f"{cell:>14}"
    print(row)

print("─" * (22 + 14 * len(sectors_valid)))
print("Significativité : *** p<0.01  ** p<0.05  * p<0.10")

# ══════════════════════════════════════════════════════════════════════════════
# 4. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
})

# ── Graphique 1 : Coefficients par secteur (barres groupées) ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("M3 — Coefficients des 4 Topics par Secteur\n(régression panel OLS avec Firm + Year FE, erreurs robustes HC)",
             fontsize=13, fontweight='bold', color='#1a3a5c', y=1.01)

topic_colors = {
    'topic_opportunity':   '#f39c12',
    'topic_adoption':      '#1a7a4a',
    'topic_laborsaving':   '#8e44ad',
    'topic_rd_investment': '#2e6da4',
}

for ax, topic in zip(axes.flatten(), TOPICS):
    betas, errs, colors_bar, xlabels = [], [], [], []

    for sector in sectors_valid:
        res = results[sector]
        b, p, se = get_coef(res, topic)
        if np.isnan(b):
            continue
        betas.append(b)
        errs.append(se)
        colors_bar.append('#2ecc71' if (not np.isnan(p) and p < 0.10) else '#bdc3c7')
        xlabels.append(SECTOR_SHORT[sector])

    bars = ax.bar(xlabels, betas, color=colors_bar, edgecolor='white',
                  linewidth=0.8, zorder=3, width=0.6)
    ax.errorbar(xlabels, betas, yerr=errs, fmt='none',
                color='#333333', capsize=4, linewidth=1.2, zorder=4)
    ax.axhline(0, color='#333333', linewidth=0.8, zorder=2)

    # Étoiles sur les barres significatives
    for i, (sector, b, p) in enumerate(zip(
            [s for s in sectors_valid], betas,
            [get_coef(results[s], topic)[1] for s in sectors_valid if results.get(s)])):
        s = stars(p)
        if s:
            y_pos = b + errs[i] + abs(max(betas, default=1)) * 0.05
            ax.text(i, y_pos, s, ha='center', va='bottom',
                    fontsize=11, color='#c0392b', fontweight='bold')

    # Ligne globale M3
    res_global = None
    try:
        d_all = df.dropna(subset=CONTROLS).set_index(['ticker','year'])
        f = f"pe_premium ~ {' + '.join(TOPICS+CONTROLS)} + EntityEffects + TimeEffects"
        res_global_obj = PanelOLS.from_formula(f, data=d_all).fit(cov_type='robust')
        b_global, p_global, _ = get_coef(res_global_obj, topic)
        ax.axhline(b_global, color=topic_colors[topic], linewidth=1.8,
                   linestyle=':', alpha=0.8, label=f'Global: {b_global:.1f}{stars(p_global)}')
        ax.legend(fontsize=8.5, framealpha=0.8)
    except:
        pass

    ax.set_title(f"{TOPIC_LABELS[topic]}", fontsize=12, fontweight='bold',
                 color=topic_colors[topic])
    ax.set_ylabel('Coefficient β', fontsize=9)
    ax.tick_params(axis='x', labelsize=8.5)

    # Légende couleur
    patch_sig = mpatches.Patch(color='#2ecc71', label='Significatif (p<0.10)')
    patch_ns  = mpatches.Patch(color='#bdc3c7', label='Non significatif')

plt.tight_layout()
path_bars = os.path.join(OUTPUT_DIR, 'sector_1_betas_par_secteur.png')
plt.savefig(path_bars, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✓ Graphique 1 sauvegardé : {path_bars}")

# ── Graphique 2 : Heatmap des coefficients ─────────────────────────────────────
coef_matrix = pd.DataFrame(index=[TOPIC_LABELS[t] for t in TOPICS],
                            columns=[SECTOR_SHORT[s] for s in sectors_valid])
pval_matrix = pd.DataFrame(index=[TOPIC_LABELS[t] for t in TOPICS],
                            columns=[SECTOR_SHORT[s] for s in sectors_valid])

for t in TOPICS:
    for s in sectors_valid:
        res = results[s]
        b, p, _ = get_coef(res, t)
        coef_matrix.loc[TOPIC_LABELS[t], SECTOR_SHORT[s]] = b
        pval_matrix.loc[TOPIC_LABELS[t], SECTOR_SHORT[s]] = p

coef_matrix = coef_matrix.astype(float)
pval_matrix = pval_matrix.astype(float)

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor('white')

vmax = coef_matrix.abs().max().max()
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(coef_matrix.values, cmap='RdYlGn', norm=norm, aspect='auto')

ax.set_xticks(range(len(sectors_valid)))
ax.set_xticklabels([SECTOR_SHORT[s] for s in sectors_valid], fontsize=11)
ax.set_yticks(range(len(TOPICS)))
ax.set_yticklabels([TOPIC_LABELS[t] for t in TOPICS], fontsize=11)

for i, t in enumerate(TOPICS):
    for j, s in enumerate(sectors_valid):
        b = coef_matrix.iloc[i, j]
        p = pval_matrix.iloc[i, j]
        if np.isnan(b):
            continue
        s_stars = stars(p)
        text = f"{b:.0f}\n{s_stars}" if s_stars else f"{b:.0f}"
        color = 'white' if abs(b) > vmax * 0.5 else '#1a1a1a'
        ax.text(j, i, text, ha='center', va='center',
                fontsize=10, color=color, fontweight='bold' if s_stars else 'normal')

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Coefficient β', fontsize=10)

ax.set_title("Heatmap des Coefficients M3 par Secteur\n(vert = positif, rouge = négatif | *** p<0.01  ** p<0.05  * p<0.10)",
             fontsize=12, fontweight='bold', color='#1a3a5c', pad=12)

plt.tight_layout()
path_heat = os.path.join(OUTPUT_DIR, 'sector_2_heatmap_coefficients.png')
plt.savefig(path_heat, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Graphique 2 (heatmap) sauvegardé : {path_heat}")

# ── Graphique 3 : Pre vs Post ChatGPT pour IT et Health Care ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Evolution Pre vs Post ChatGPT par Secteur\n(topic_rd_investment & topic_adoption)",
             fontsize=12, fontweight='bold', color='#1a3a5c')

focus_sectors = ['Information Technology', 'Health Care', 'Financials', 'Industrials']
focus_sectors = [s for s in focus_sectors if results_pre.get(s) and results_post.get(s)]

for ax, topic in zip(axes, ['topic_rd_investment', 'topic_adoption']):
    x      = np.arange(len(focus_sectors))
    width  = 0.35
    betas_pre, betas_post = [], []
    errs_pre, errs_post   = [], []
    pvals_pre, pvals_post = [], []

    for s in focus_sectors:
        b_pre,  p_pre,  se_pre  = get_coef(results_pre.get(s),  topic)
        b_post, p_post, se_post = get_coef(results_post.get(s), topic)
        betas_pre.append(b_pre   if not np.isnan(b_pre)  else 0)
        betas_post.append(b_post if not np.isnan(b_post) else 0)
        errs_pre.append(se_pre   if not np.isnan(se_pre)  else 0)
        errs_post.append(se_post if not np.isnan(se_post) else 0)
        pvals_pre.append(p_pre)
        pvals_post.append(p_post)

    bars1 = ax.bar(x - width/2, betas_pre, width, label='Pre-2023',
                   color='#7fb3d3', edgecolor='white', zorder=3)
    bars2 = ax.bar(x + width/2, betas_post, width, label='Post-2022',
                   color='#2e6da4', edgecolor='white', zorder=3)
    ax.errorbar(x - width/2, betas_pre,  yerr=errs_pre,  fmt='none',
                color='#333', capsize=3, linewidth=1, zorder=4)
    ax.errorbar(x + width/2, betas_post, yerr=errs_post, fmt='none',
                color='#333', capsize=3, linewidth=1, zorder=4)

    # Étoiles
    for i, (p1, p2, b1, b2, e1, e2) in enumerate(
            zip(pvals_pre, pvals_post, betas_pre, betas_post, errs_pre, errs_post)):
        if stars(p1): ax.text(i-width/2, b1+e1+5, stars(p1), ha='center', fontsize=10, color='#c0392b', fontweight='bold')
        if stars(p2): ax.text(i+width/2, b2+e2+5, stars(p2), ha='center', fontsize=10, color='#c0392b', fontweight='bold')

    ax.axhline(0, color='#333333', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SECTOR_SHORT[s] for s in focus_sectors], fontsize=10)
    ax.set_ylabel('Coefficient β', fontsize=10)
    ax.set_title(f"{TOPIC_LABELS[topic]}", fontsize=12, fontweight='bold',
                 color=topic_colors[topic])
    ax.legend(fontsize=9, framealpha=0.8)

plt.tight_layout()
path_prepost = os.path.join(OUTPUT_DIR, 'sector_3_pre_vs_post_chatgpt.png')
plt.savefig(path_prepost, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Graphique 3 (pre/post) sauvegardé : {path_prepost}")

# ── Export CSV récapitulatif ───────────────────────────────────────────────────
rows = []
for s in sectors_valid:
    res = results[s]
    n_obs   = len(df[df['sector']==s].dropna(subset=CONTROLS))
    n_firms = df[df['sector']==s]['ticker'].nunique()
    row = {
        'Secteur': s,
        'N': n_obs,
        'Firmes': n_firms,
        'R2': round(res.rsquared, 4),
    }
    for t in TOPICS:
        b, p, se = get_coef(res, t)
        row[f'{TOPIC_LABELS[t]}_beta'] = round(b, 2) if not np.isnan(b) else ''
        row[f'{TOPIC_LABELS[t]}_pval'] = round(p, 4) if not np.isnan(p) else ''
        row[f'{TOPIC_LABELS[t]}_sig']  = stars(p)
    rows.append(row)

csv_df = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR, 'sector_regression_results.csv')
csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✓ CSV exporté : {csv_path}")