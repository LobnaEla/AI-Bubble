"""
scatter_plots.py — Scatter plots pour la présentation AI Bubble
Youssef Chebil & Lobna Elabed — Telecom Paris 2A

Génère 3 scatter plots :
  1. spec_vs_oper_ratio vs pe_premium
  2. topic_rd_investment vs pe_premium
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Chemins ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

INPUT_FILE = os.path.join(PARENT_DIR, "data", "processed", "panel_final.csv")
OUTPUT_DIR = os.path.join(PARENT_DIR, "data", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(INPUT_FILE):
    INPUT_FILE = os.path.join(BASE_DIR, "panel_final.csv")

# ── Chargement ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=['pe_premium'])

# Winsoriser pe_premium à 99% pour la visualisation (enlever les outliers extrêmes)
p99 = df['pe_premium'].quantile(0.99)
p01 = df['pe_premium'].quantile(0.01)
df_plot = df[(df['pe_premium'] <= p99) & (df['pe_premium'] >= p01)].copy()

# ── Palette couleurs par secteur ───────────────────────────────────────────────
SECTOR_COLORS = {
    'Information Technology':  '#2e6da4',
    'Health Care':             '#1a7a4a',
    'Financials':              '#d35400',
    'Communication Services':  '#8e44ad',
    'Industrials':             '#c0392b',
    'Energy':                  '#f39c12',
}

def get_colors(series):
    return [SECTOR_COLORS.get(s, '#888888') for s in series]

# ── Style général ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
})

def add_regression_line(ax, x, y, color='#1a3a5c'):
    """Ajoute une droite de régression et affiche R² et p-value."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]
    if len(x_clean) < 10:
        return
    slope, intercept, r, p, se = stats.linregress(x_clean, y_clean)
    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=color, linewidth=2.5, linestyle='--', alpha=0.9, zorder=5)
    stars = '***' if p < 0.001 else '**' if p < 0.05 else '*' if p < 0.10 else 'n.s.'
    ax.text(0.97, 0.05, f'R² = {r**2:.3f}\np = {p:.3f} {stars}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, color=color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=color, alpha=0.8))

def add_legend(ax):
    patches = [mpatches.Patch(color=c, label=s) for s, c in SECTOR_COLORS.items()]
    ax.legend(handles=patches, title='Secteur', loc='upper left',
              fontsize=8.5, title_fontsize=9, framealpha=0.9,
              edgecolor='#dde3ea')

# ══════════════════════════════════════════════════════════════════════════════
# SCATTER 1 — spec_vs_oper_ratio vs pe_premium
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

colors = get_colors(df_plot['sector'])
ax.scatter(df_plot['spec_vs_oper_ratio'], df_plot['pe_premium'],
           c=colors, alpha=0.55, s=45, edgecolors='white', linewidths=0.4, zorder=3)

add_regression_line(ax, df_plot['spec_vs_oper_ratio'].values, df_plot['pe_premium'].values)

ax.axvline(x=0.5, color='#888888', linewidth=1, linestyle=':', alpha=0.7)
ax.text(0.51, ax.get_ylim()[1]*0.92, 'Neutre (0.5)', color='#888888', fontsize=9, alpha=0.8)

ax.set_xlabel('spec_vs_oper_ratio\n(0 = 100% opérationnel  |  0.5 = neutre  |  1 = 100% spéculatif)',
              fontsize=11, labelpad=8)
ax.set_ylabel('P/E Premium\n(P/E firme − médiane sectorielle)', fontsize=11, labelpad=8)
ax.set_title('Narratif AI global vs P/E Premium\n(M1 — spec_vs_oper_ratio)',
             fontsize=14, fontweight='bold', color='#1a3a5c', pad=12)

add_legend(ax)
plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, 'scatter_1_ratio_vs_pepremium.png')
plt.savefig(path1, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'✓ Scatter 1 sauvegardé : {path1}')

# ══════════════════════════════════════════════════════════════════════════════
# SCATTER 2 — topic_rd_investment vs pe_premium
# ══════════════════════════════════════════════════════════════════════════════
df_rd = df_plot[df_plot['topic_rd_investment'] > 0].copy()  # garder seulement firmes avec signal

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

colors = get_colors(df_rd['sector'])
ax.scatter(df_rd['topic_rd_investment'], df_rd['pe_premium'],
           c=colors, alpha=0.6, s=55, edgecolors='white', linewidths=0.4, zorder=3)

add_regression_line(ax, df_rd['topic_rd_investment'].values, df_rd['pe_premium'].values,
                    color='#2e6da4')

ax.set_xlabel('Topic R&D/Investment Score\n(densité du discours investissement IA / 1000 mots)',
              fontsize=11, labelpad=8)
ax.set_ylabel('P/E Premium\n(P/E firme − médiane sectorielle)', fontsize=11, labelpad=8)
ax.set_title('Discours R&D/Investment vs P/E Premium\n(M3 — β = +96.9, p < 0.001 ***)',
             fontsize=14, fontweight='bold', color='#1a3a5c', pad=12)

# Annotation résultat clé
ax.annotate('Parler d\'investir en IA\ngonfle le P/E de +97 pts ***',
            xy=(df_rd['topic_rd_investment'].quantile(0.85),
                df_rd['pe_premium'].quantile(0.75)),
            xytext=(df_rd['topic_rd_investment'].quantile(0.6),
                    df_rd['pe_premium'].quantile(0.88)),
            fontsize=9.5, color='#1a3a5c', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#1a3a5c', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#e3f0ff', edgecolor='#2e6da4', alpha=0.9))

add_legend(ax)
plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, 'scatter_2_rdinvestment_vs_pepremium.png')
plt.savefig(path2, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'✓ Scatter 2 sauvegardé : {path2}')

print('\n✓ Tous les scatter plots générés dans data/outputs/')
