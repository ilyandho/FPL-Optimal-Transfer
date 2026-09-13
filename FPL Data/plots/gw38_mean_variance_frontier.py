"""
Figure for Results \\S4.4 "A Worked Example: Solving the Optimization Problem at
Gameweek 38":
  - GW38 Mean Variance Frontier.png

Unlike the other scripts in this folder, this one has no standalone saved artifact to
read: `optimization.ipynb`'s single-gameweek walkthrough builds `results_df` (the
lambda grid -> Prob(beat opponent)/mean/variance table) in memory and never saves it.
So this script re-executes that notebook's own code cells live (Steps 1-6: rebuilds
mu_delta/Sigma_delta, refits the opponent's copula, runs the Monte Carlo, and solves the
Gurobi MIQP across the lambda grid) up to and including the cell that produces
`results_df`, then plots it. Needs the pycaret env (gurobipy + pyvinecopulib) and takes
roughly 1-2 minutes -- it's doing the real fit, not reading a cached number.

To regenerate for a different game week/opponent/risk level: edit GW / OPPONENT_TEAM_ID
/ R_PCT in the PARAMS_OVERRIDE block below (these override the notebook's own Parameters
cell after it's exec'd).
"""
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent  # .../FPL Data
OPP_DIR = ROOT / 'beta estimators' / 'opponent betas'
NOTEBOOK = OPP_DIR / 'optimization.ipynb'
OUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

# Overrides applied after the notebook's own Parameters cell runs (leave as-is to
# reproduce the thesis's GW38 example exactly).
PARAMS_OVERRIDE = {}  # e.g. {'GW': 30, 'OPPONENT_TEAM_ID': 12345, 'R_PCT': 0.8}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def extract_walkthrough_source(notebook_path):
    """Concatenate optimization.ipynb's code cells up to (not including) the bulk
    'Running across every gameweek' loop, which fits the opponent copula for all 33
    game weeks and isn't needed just to rebuild one game week's frontier."""
    nb = json.load(open(notebook_path))
    chunks = []
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if 'opt_results_by_gw = {}' in src:
            break
        chunks.append(src)
    return '\n\n'.join(chunks)


def main():
    src = extract_walkthrough_source(NOTEBOOK)
    # Run the notebook's own setup/Steps 1-6 in a fresh namespace, cwd matters for its
    # relative data paths ('../../rolled_data_24_25.csv' etc.), so chdir like the
    # notebook itself would.
    import os
    old_cwd = os.getcwd()
    os.chdir(OPP_DIR)
    ns = {}
    try:
        exec(compile(src, str(NOTEBOOK), 'exec'), ns)
        for k, v in PARAMS_OVERRIDE.items():
            ns[k] = v
    finally:
        os.chdir(old_cwd)

    results_df = ns['results_df']
    lam = results_df['lambda'].to_numpy()
    prob = results_df['prob_beat_opponent'].to_numpy()
    mean = results_df['mean_score'].to_numpy()
    var = results_df['var_score'].to_numpy()
    star_idx = int(np.argmax(prob))
    GW = ns['GW']

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    ax = axes[0]
    ax.plot(lam, prob, marker='o', markersize=4, linewidth=1.5, color='#1f77b4')
    ax.scatter([lam[star_idx]], [prob[star_idx]], color='#d62728', zorder=5, s=60,
               label=r'$\lambda^*\approx%.3f$' % lam[star_idx])
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\widehat{Prob}(Y_w>0)$')
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    sc = ax.scatter(var, mean, c=lam, cmap='viridis', s=45,
                     norm=matplotlib.colors.LogNorm(vmin=max(lam[lam > 0].min(), 1e-3), vmax=lam.max()))
    ax.scatter([var[star_idx]], [mean[star_idx]], facecolors='none', edgecolors='#d62728', s=160, linewidths=2,
               label=r'$\lambda^*\approx%.3f$' % lam[star_idx])
    ax.set_xlabel(r'variance, $w^T\Sigma_\delta w$')
    ax.set_ylabel(r'mean score, $w^T\mu_\delta$')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r'$\lambda$')
    ax.legend(frameon=False, fontsize=9, loc='lower right')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'GW38 Mean Variance Frontier.png', dpi=200)
    plt.close(fig)

    print(f"GW{GW}: lambda*={lam[star_idx]:.4f}  Prob(beat opponent)={prob[star_idx]:.1%}")
    print(f"\nFigure written to {OUT_DIR}")


if __name__ == '__main__':
    main()
