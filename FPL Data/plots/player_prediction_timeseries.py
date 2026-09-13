"""
Figure for Results \\S4.2.4 "Walk-Forward Predictive Performance":
  - xP Predicted vs Actual Timeseries.png

Complements the pooled predicted-vs-actual scatter (xp_predictor_performance.py) with
a per-player view: for each position's single highest actual-xP scorer over the
season, plots xP_pred against xP_actual week by week, so the tracking error is visible
as a trajectory rather than only as a pooled cloud of points.

Source data: predictors/hist/{pos}_preds.csv (columns: element, name, xP_pred,
xP_actual, round). The "top player" per position is whoever has the highest sum of
xP_actual over game weeks 6-38 in that file -- not hand-picked.
"""
import pathlib

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent  # .../FPL Data
HIST_DIR = ROOT / 'predictors' / 'hist'
OUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

POSITIONS = {'gk': 'Goalkeeper', 'def': 'Defender', 'mid': 'Midfielder', 'fwd': 'Forward'}
COLOR_ACTUAL = '#2ca02c'
COLOR_PRED = '#1f77b4'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def top_player_by_position():
    chosen = {}
    for suf in POSITIONS:
        df = pd.read_csv(HIST_DIR / f'{suf}_preds.csv')
        totals = df.groupby('name')['xP_actual'].sum().sort_values(ascending=False)
        top_name = totals.index[0]
        chosen[suf] = (top_name, df[df['name'] == top_name].sort_values('round'))
    return chosen


def main():
    chosen = top_player_by_position()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    for ax, (suf, pos) in zip(axes.ravel(), POSITIONS.items()):
        name, g = chosen[suf]
        ax.plot(g['round'], g['xP_actual'], marker='o', markersize=3.5, linewidth=1.6,
                color=COLOR_ACTUAL, label='actual xP')
        ax.plot(g['round'], g['xP_pred'], marker='s', markersize=3.5, linewidth=1.6,
                color=COLOR_PRED, linestyle='--', label='predicted xP')
        mae = (g['xP_actual'] - g['xP_pred']).abs().mean()
        ax.set_title(f'{pos}: {name} (season MAE={mae:.2f})', fontsize=10)
        ax.set_xlabel('Game week')
        ax.set_ylabel('xP')
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'xP Predicted vs Actual Timeseries.png', dpi=200)
    plt.close(fig)

    print("Top scorer (by total actual xP) per position:")
    for suf, pos in POSITIONS.items():
        name, g = chosen[suf]
        mae = (g['xP_actual'] - g['xP_pred']).abs().mean()
        print(f"  {pos:12s} {name:25s} total actual xP={g['xP_actual'].sum():.1f}  season MAE={mae:.2f}")
    print(f"\nFigure written to {OUT_DIR}")


if __name__ == '__main__':
    main()
