"""
Figure for Results \\S4.2.4 "Walk-Forward Predictive Performance":
  - GW38 xP Predicted vs Actual by Player.png

Per-player bar comparison of predicted vs. actual xP at game week 38, one panel per
position, restricted to each position's top N players by actual xP that gameweek (the
full player pool per position numbers in the hundreds -- most with near-zero xP that
week -- so a full bar chart would be unreadable; the top scorers are also the players
this comparison is most informative for).

Source data: predictors/hist/{pos}_preds.csv (columns: element, name, xP_pred,
xP_actual, round), filtered to round == GW.
"""
import pathlib

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent  # .../FPL Data
HIST_DIR = ROOT / 'predictors' / 'hist'
OUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

GW = 38
TOP_N = 12
POSITIONS = {'gk': 'Goalkeeper', 'def': 'Defender', 'mid': 'Midfielder', 'fwd': 'Forward'}
COLOR_ACTUAL = '#2ca02c'
COLOR_PRED = '#1f77b4'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def top_players_at_gw(suf):
    df = pd.read_csv(HIST_DIR / f'{suf}_preds.csv')
    g = df[df['round'] == GW].copy()
    g = g.sort_values('xP_actual', ascending=False).head(TOP_N)
    return g.sort_values('xP_actual', ascending=True)  # ascending for barh top-to-bottom


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (suf, pos) in zip(axes.ravel(), POSITIONS.items()):
        g = top_players_at_gw(suf)
        y = np.arange(len(g))
        height = 0.38
        ax.barh(y + height / 2, g['xP_actual'], height=height, color=COLOR_ACTUAL, label='actual xP')
        ax.barh(y - height / 2, g['xP_pred'], height=height, color=COLOR_PRED, label='predicted xP')
        ax.set_yticks(y)
        ax.set_yticklabels(g['name'], fontsize=8)
        ax.set_xlabel('xP')
        ax.set_title(f'{pos} (top {TOP_N} by actual xP, GW{GW})', fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'GW38 xP Predicted vs Actual by Player.png', dpi=200)
    plt.close(fig)

    print(f"Top {TOP_N} players by actual xP at GW{GW}, per position:")
    for suf, pos in POSITIONS.items():
        g = top_players_at_gw(suf).sort_values('xP_actual', ascending=False)
        print(f"\n  {pos}:")
        for _, r in g.iterrows():
            print(f"    {r['name']:25s} actual={r['xP_actual']:.2f}  predicted={r['xP_pred']:.2f}")
    print(f"\nFigure written to {OUT_DIR}")


if __name__ == '__main__':
    main()
