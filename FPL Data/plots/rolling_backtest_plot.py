"""
Figure for Results \\S4.5.2 "Cumulative Results":
  - Rolling Cumulative Points.png

Cumulative total_points over the season for the rolling-optimized trajectory (fixture
tilt, corrected), the manager's real historical trajectory, and the real weekly
opponents (Table 4.4 / Figure 4.13 style comparison in the thesis).

Source data:
  beta estimators/opponent betas/estimates/rolling_optimization_results_fixed.joblib
      DataFrame from run_rolling_optimization (rolling_optimization.ipynb): one row per
      game week with the simulated trajectory's own points that week
      (sim_optimized_points) and that week's real opponent's actual points
      (opponent_points). Does NOT include the manager's own real historical points --
      those are recomputed here directly from league_selections_df.csv, exactly as
      rolling_optimization.ipynb's own "Season-long comparison" cell does.
  rolled_data_24_25.csv, league_selections_df.csv (project root)

To refresh after a new rolling-optimization run: re-run
`rolling_optimization.ipynb`'s `run_rolling_optimization(...)` cell, re-save its result
to `rolling_optimization_results_fixed.joblib` (or point RESULTS_JOBLIB below at a new
file), then re-run this script.
"""
import ast
import pathlib

import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent  # .../FPL Data
OPP_DIR = ROOT / 'beta estimators' / 'opponent betas'
OUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

OWN_TEAM_ID = 205  # manager 205 -- "us" throughout this whole project
RESULTS_JOBLIB = OPP_DIR / 'estimates' / 'rolling_optimization_results_fixed.joblib'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def real_squad_actuals(team_id, gw, league_selections, player_data):
    """Verbatim from rolling_optimization.ipynb: sum of actual total_points across a
    team's real 15-man squad that gameweek, pulled straight from historical data."""
    rows = league_selections[(league_selections['team_id'] == team_id) & (league_selections['round'] == gw)]
    if len(rows) == 0:
        return None
    squad = ast.literal_eval(rows.iloc[0]['squad'])
    gw_players = player_data[player_data['round'] == gw].drop_duplicates('element').set_index('element')
    rows_p = gw_players.reindex(squad).dropna(subset=['xP'])
    return float(rows_p['total_points'].sum())


def main():
    rolling_df = joblib.load(RESULTS_JOBLIB)
    player_data = pd.read_csv(ROOT / 'rolled_data_24_25.csv')
    league_selections = pd.read_csv(ROOT / 'league_selections_df.csv')

    solved = rolling_df[rolling_df['status'] == 'ok'].copy().reset_index(drop=True)
    solved['real_points'] = solved['GW'].apply(
        lambda gw: real_squad_actuals(OWN_TEAM_ID, gw, league_selections, player_data)
    )
    solved['cum_rolling'] = solved['sim_optimized_points'].cumsum()
    solved['cum_real'] = solved['real_points'].cumsum()
    solved['cum_opponent'] = solved['opponent_points'].cumsum()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(solved['GW'], solved['cum_rolling'], marker='o', markersize=3.5, linewidth=1.8,
            color='#1f77b4', label='Rolling-optimized trajectory (fixture tilt, corrected)')
    ax.plot(solved['GW'], solved['cum_real'], marker='s', markersize=3.5, linewidth=1.8,
            color='#2ca02c', label="Manager's real historical trajectory")
    ax.plot(solved['GW'], solved['cum_opponent'], marker='^', markersize=3.5, linewidth=1.8,
            color='#d62728', linestyle='--', label='Real weekly opponents')
    ax.set_xlabel('Game week')
    ax.set_ylabel('Cumulative total_points')
    ax.legend(loc='upper left', fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'Rolling Cumulative Points.png', dpi=200)
    plt.close(fig)

    print(f"{len(solved)} / {len(rolling_df)} game weeks solved and plotted")
    print(f"cumulative: rolling={solved['cum_rolling'].iloc[-1]:.0f}  "
          f"real={solved['cum_real'].iloc[-1]:.0f}  opponents={solved['cum_opponent'].iloc[-1]:.0f}")
    print(f"\nFigure written to {OUT_DIR}")


if __name__ == '__main__':
    main()
