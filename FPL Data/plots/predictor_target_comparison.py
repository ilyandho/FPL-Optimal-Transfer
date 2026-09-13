"""
Figure for Results \\S4.2.3 "Choice of Prediction Target":
  - R2 XP vs Points Model.png

Compares the walk-forward xP-targeted model against the walk-forward total_points-
targeted model, both evaluated against actual total_points, by position (Table 4.1 /
Figure 4.4 in the thesis).

Source data:
  predictors/hist/{pos}_preds.csv         (xP-targeted model; columns include xP_pred)
  predictors/hist/{pos}_preds_points.csv  (total_points-targeted model; columns include
                                            points_pred, points_actual)
"""
import pathlib

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent  # .../FPL Data
HIST_DIR = ROOT / 'predictors' / 'hist'
OUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

POSITIONS = {'gk': 'Goalkeeper', 'def': 'Defender', 'mid': 'Midfielder', 'fwd': 'Forward'}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def main():
    r2_xp, r2_points = [], []
    for suf in POSITIONS:
        xp_preds = pd.read_csv(HIST_DIR / f'{suf}_preds.csv')  # xP-targeted, has xP_pred/xP_actual
        points_preds = pd.read_csv(HIST_DIR / f'{suf}_preds_points.csv')  # total_points-targeted

        # xP model's R2 against actual total_points requires total_points; xP_preds.csv
        # doesn't carry it, so pull actual total_points from the points-targeted file's
        # own actuals (points_actual == total_points) and merge on (element, round).
        merged = xp_preds.merge(
            points_preds[['element', 'round', 'points_actual']], on=['element', 'round'], how='inner'
        )
        r2_xp.append(r2(merged['points_actual'], merged['xP_pred']))
        r2_points.append(r2(points_preds['points_actual'], points_preds['points_pred']))

    positions = list(POSITIONS.values())
    x = np.arange(len(positions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4))
    b1 = ax.bar(x - width / 2, r2_xp, width, label='xP-targeted model', color='#1f77b4')
    b2 = ax.bar(x + width / 2, r2_points, width, label='total_points-targeted model', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_ylabel('$R^2$ against actual total_points')
    ax.legend(frameon=False)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', (bar.get_x() + bar.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 3), ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'R2 XP vs Points Model.png', dpi=200)
    plt.close(fig)

    print("R2 against actual total_points:")
    for pos, rx, rp in zip(positions, r2_xp, r2_points):
        print(f"  {pos:12s} xP model={rx:.3f}   total_points model={rp:.3f}")
    print(f"\nFigure written to {OUT_DIR}")


if __name__ == '__main__':
    main()
