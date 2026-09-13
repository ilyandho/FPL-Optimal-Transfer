"""
Figure and summary stats for Results \\S4.2.1 "Model Family Selection":
  - Model Family Selection Evolution.png

For each position and game week, which regression model family
(evaluate_across_gws -> tune_best_model in utils.py) was actually selected for the
deployed xP model, and how often the selection changed from one game week to the next.

Source data: predictors/models_by_round_{gk,def,mid,fwd}
  dict keyed by game week -> {'mae', 'r2', 'best_model', 'c_final'} for the model
  actually used to predict that game week (the winner of evaluate_across_gws's
  MAE comparison, re-tuned every 5 game weeks per train_position's retune_every).
"""
import pathlib

import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent  # .../FPL Data
PRED_DIR = ROOT / 'predictors'
OUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

POSITIONS = {'gk': 'Goalkeeper', 'def': 'Defender', 'mid': 'Midfielder', 'fwd': 'Forward'}
MODEL_COLORS = {
    'RandomForest': '#1f77b4',
    'HistGradientBoosting': '#ff7f0e',
    'XGBoost': '#2ca02c',
    'GradientBoosting': '#d62728',
    'LightGBM': '#9467bd',
    'Ridge': '#8c564b',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_selection():
    data = {}
    for suf in POSITIONS:
        d = joblib.load(PRED_DIR / f'models_by_round_{suf}')
        rows = [{'round': rnd, 'best_model': r['best_model'], 'mae': r['mae'], 'r2': r['r2']}
                for rnd, r in sorted(d.items())]
        data[suf] = pd.DataFrame(rows)
    return data


def plot_selection_evolution(data):
    all_models = sorted({m for df in data.values() for m in df['best_model'].unique()})

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    y_positions = {suf: i for i, suf in enumerate(POSITIONS)}
    for suf, pos in POSITIONS.items():
        df = data[suf]
        for model in all_models:
            sub = df[df['best_model'] == model]
            if len(sub) == 0:
                continue
            ax.scatter(sub['round'], [y_positions[suf]] * len(sub), color=MODEL_COLORS[model],
                       s=28, marker='s', label=model)

    # de-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.35),
              ncol=len(seen), frameon=False, fontsize=9)

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(POSITIONS.values()))
    ax.set_xlabel('Game week')
    ax.set_ylim(-0.5, len(POSITIONS) - 0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'Model Family Selection Evolution.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    data = load_selection()
    plot_selection_evolution(data)

    all_models = set()
    print("Model family selected, by position (game weeks 6-38):")
    for suf, pos in POSITIONS.items():
        df = data[suf]
        counts = df['best_model'].value_counts()
        all_models |= set(df['best_model'].unique())
        n_transitions = int((df['best_model'] != df['best_model'].shift()).sum() - 1)
        print(f"  {pos:12s} {dict(counts)}  transitions={n_transitions}/{len(df) - 1}")
    print(f"\nFamilies ever selected across all 4 positions: {sorted(all_models)}")
    print(f"\nFigure written to {OUT_DIR}")


if __name__ == '__main__':
    main()
