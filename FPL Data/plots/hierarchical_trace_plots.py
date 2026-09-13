"""
Figures for Results \\S4.3.1/4.3.2 (Cross-Sectional League Baseline / Individual
Longitudinal Update):
  - {GK,DEF,MID,FWD} League Trace.png
  - {GK,DEF,MID,FWD} Manager Trace.png

Loads the already-fitted PyMC InferenceData objects saved by the beta-estimation
pipeline (beta_estimator.ipynb / beta_{gk,def,mid,fwd}.ipynb -> utils.py's
fit_position_pipeline / fit_manager_pipeline) and re-plots them with arviz -- no
re-sampling, so this runs in seconds even though the original NUTS fits took minutes.

Source data:
  beta estimators/estimates/league_models_{gk,def,mid,fwd}
      dict keyed by game week; each entry has 'trace_final' (post-pruning fit) and
      'feats' (the pruned feature list).
  beta estimators/opponent betas/estimates/manager_fits_{GW}.joblib
      dict keyed by game week -> position -> {'trace', 'team_id', 'feats', 'gamma_mean', ...}
      (one manager/opponent's individual update at that game week).

To regenerate for a different game week or opponent: change GW below (a
manager_fits_{GW}.joblib must exist for that week), or load a different team's fit if
manager_fits ever starts storing more than one manager per game week.
"""
import pathlib

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import arviz as az

ROOT = pathlib.Path(__file__).resolve().parent.parent  # .../FPL Data
BETA_DIR = ROOT / 'beta estimators'
OPP_DIR = BETA_DIR / 'opponent betas'
OUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

GW = 38
POSITIONS = {'gk': 'Goalkeeper', 'def': 'Defender', 'mid': 'Midfielder', 'fwd': 'Forward'}

plt.rcParams.update({'font.family': 'serif', 'font.size': 10})


def plot_league_traces():
    print(f"=== League cross-sectional baseline, GW{GW} ===")
    for suf, pos in POSITIONS.items():
        lm = joblib.load(BETA_DIR / 'estimates' / f'league_models_{suf}')[GW]
        trace = lm['trace_final']
        n_feats = len(lm['feats'])
        max_rhat = lm['summary_final']['r_hat'].max()
        div = int(trace.sample_stats['diverging'].sum())

        axes = az.plot_trace(trace, var_names=['beta'], compact=True)
        fig = axes.ravel()[0].figure
        fig.suptitle(f"{pos}: league baseline $\\beta^{{league}}$ posterior, GW{GW} (n={n_feats} features)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f'{suf.upper()} League Trace.png', dpi=150)
        plt.close(fig)
        print(f"  {pos:12s} kept={n_feats} feats={lm['feats']} max_r_hat={max_rhat:.3f} div={div}")


def plot_manager_traces():
    print(f"\n=== Individual longitudinal update, GW{GW} ===")
    mf = joblib.load(OPP_DIR / 'estimates' / f'manager_fits_{GW}.joblib')[GW]
    for suf, pos in POSITIONS.items():
        m = mf[pos]
        trace = m['trace']
        n_feats = len(m['feats'])
        max_rhat = m['summary']['r_hat'].max()
        div = int(trace.sample_stats['diverging'].sum())

        axes = az.plot_trace(trace, var_names=['delta_beta', 'gamma_m'], compact=True)
        fig = axes.ravel()[0].figure
        fig.suptitle(f"{pos}: individual update $(\\Delta\\beta_m, \\gamma_m)$ for opponent {m['team_id']}, GW{GW}")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f'{suf.upper()} Manager Trace.png', dpi=150)
        plt.close(fig)
        print(f"  {pos:12s} team={m['team_id']} n_weeks={m['n_weeks']} "
              f"gamma_mean={m['gamma_mean']:.3f} max_r_hat={max_rhat:.3f} div={div}")


if __name__ == '__main__':
    plot_league_traces()
    plot_manager_traces()
    print(f"\nFigures written to {OUT_DIR}")
