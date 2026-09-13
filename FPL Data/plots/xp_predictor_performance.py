"""
Figures for Results \\S4.2.2 "Walk-Forward Predictive Performance":
  - xP Predicted vs Actual.png   (predicted vs. actual xP scatter, by position)
  - xP MAE Evolution.png         (per-game-week MAE, by position)
  - xP RMSE Evolution.png        (per-game-week RMSE, by position)
  - xP R2 Evolution.png          (per-game-week R^2, by position)

Source data: predictors/hist/{pos}_preds.csv -- the walk-forward predictions of the
deployed xP model (columns: element, name, xP_pred, xP_actual, round), one row per
player per game week, for each of GK/DEF/MID/FWD. This is the same model
build_mu_sigma_delta reads throughout the optimization pipeline.

RMSE is reported alongside MAE (rather than replacing it) because the model
comparison/tuning stage (utils.py's tune_best_model) actually selects on MAE
(scoring="neg_mean_absolute_error"), so MAE is what the pipeline is optimized for;
RMSE's extra sensitivity to large individual misses is still useful as a secondary
diagnostic, since it should react more sharply to exactly the outlier weeks/players
(e.g. the blank-gameweek collapses, or a player like Salah with occasional huge misses)
that MAE alone can understate.
"""
import pathlib

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = pathlib.Path(__file__).resolve().parent.parent  # .../FPL Data
HIST_DIR = ROOT / 'predictors' / 'hist'
OUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

POSITIONS = {'gk': 'Goalkeeper', 'def': 'Defender', 'mid': 'Midfielder', 'fwd': 'Forward'}
COLORS = {'gk': '#1f77b4', 'def': '#ff7f0e', 'mid': '#2ca02c', 'fwd': '#d62728'}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_predictions():
    data = {}
    for suf in POSITIONS:
        data[suf] = pd.read_csv(HIST_DIR / f'{suf}_preds.csv')
    return data


def per_round_metrics(df):
    rows = []
    for rnd, g in df.groupby('round'):
        rows.append({
            'round': rnd,
            'mae': mean_absolute_error(g['xP_actual'], g['xP_pred']),
            'rmse': np.sqrt(mean_squared_error(g['xP_actual'], g['xP_pred'])),
            'r2': r2_score(g['xP_actual'], g['xP_pred']),
            'n': len(g),
        })
    return pd.DataFrame(rows).sort_values('round')


def plot_mae_evolution(per_round):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for suf, pos in POSITIONS.items():
        pr = per_round[suf]
        ax.plot(pr['round'], pr['mae'], marker='o', markersize=3, linewidth=1.5,
                color=COLORS[suf], label=pos)
    ax.set_xlabel('Game week')
    ax.set_ylabel('MAE (xP)')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'xP MAE Evolution.png', dpi=200)
    plt.close(fig)


def plot_rmse_evolution(per_round):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for suf, pos in POSITIONS.items():
        pr = per_round[suf]
        ax.plot(pr['round'], pr['rmse'], marker='o', markersize=3, linewidth=1.5,
                color=COLORS[suf], label=pos)
    ax.set_xlabel('Game week')
    ax.set_ylabel('RMSE (xP)')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'xP RMSE Evolution.png', dpi=200)
    plt.close(fig)


def plot_r2_evolution(per_round):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for suf, pos in POSITIONS.items():
        pr = per_round[suf]
        ax.plot(pr['round'], pr['r2'], marker='o', markersize=3, linewidth=1.5,
                color=COLORS[suf], label=pos)
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Game week')
    ax.set_ylabel('$R^2$ (xP)')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'xP R2 Evolution.png', dpi=200)
    plt.close(fig)


def plot_predicted_vs_actual(data):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))
    for ax, (suf, pos) in zip(axes.ravel(), POSITIONS.items()):
        df = data[suf]
        overall_r2 = r2_score(df['xP_actual'], df['xP_pred'])
        ax.scatter(df['xP_actual'], df['xP_pred'], s=6, alpha=0.25, color=COLORS[suf], linewidths=0)
        lims = [min(df['xP_actual'].min(), df['xP_pred'].min()),
                max(df['xP_actual'].max(), df['xP_pred'].max())]
        ax.plot(lims, lims, color='black', linewidth=1, linestyle='--')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f'{pos} ($R^2$={overall_r2:.3f}, n={len(df)})', fontsize=10)
        ax.set_xlabel('actual xP')
        ax.set_ylabel('predicted xP')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'xP Predicted vs Actual.png', dpi=200)
    plt.close(fig)


def main():
    data = load_predictions()
    per_round = {suf: per_round_metrics(df) for suf, df in data.items()}

    plot_mae_evolution(per_round)
    plot_rmse_evolution(per_round)
    plot_r2_evolution(per_round)
    plot_predicted_vs_actual(data)

    print("Overall accuracy (all game weeks pooled):")
    for suf, pos in POSITIONS.items():
        df = data[suf]
        pr = per_round[suf]
        overall_rmse = np.sqrt(mean_squared_error(df['xP_actual'], df['xP_pred']))
        print(f"  {pos:12s} R2={r2_score(df['xP_actual'], df['xP_pred']):.3f}  "
              f"MAE={mean_absolute_error(df['xP_actual'], df['xP_pred']):.3f}  "
              f"RMSE={overall_rmse:.3f}  "
              f"n={len(df)}  "
              f"per-round MAE range=[{pr['mae'].min():.3f}, {pr['mae'].max():.3f}]  "
              f"per-round RMSE range=[{pr['rmse'].min():.3f}, {pr['rmse'].max():.3f}]  "
              f"per-round R2 range=[{pr['r2'].min():.3f}, {pr['r2'].max():.3f}]")
    print(f"\nFigures written to {OUT_DIR}")


if __name__ == '__main__':
    main()
