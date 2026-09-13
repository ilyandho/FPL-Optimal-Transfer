# Thesis plot scripts

Scripts that regenerate every figure used in `MSC-Thesis/.../Results and Discussion`
(Chapter 4). Each one reads from the project's permanent data/model artifacts (not from
any scratch/temp files), so they can be re-run after retraining a model or re-running a
backtest to refresh the figures with new numbers.

Run each script from anywhere with the `pycaret` micromamba env (needs pandas, numpy,
matplotlib, arviz, joblib, and — for `gw38_mean_variance_frontier.py` only —
gurobipy/pyvinecopulib since it re-executes part of `optimization.ipynb`):

```
/home/ilyandho/micromamba/envs/pycaret/bin/python plots/<script>.py
```

Each script writes its PNG(s) into `plots/output/`. Copy the ones you want into the
current thesis snapshot's `images/` folder (the dated `MSC-Thesis/<date> - .../images/`
directory) and re-run `pdflatex` — the thesis `.tex` already `\includegraphics`-references
these exact filenames.

| Script | Figures produced | Source data |
|---|---|---|
| `xp_predictor_performance.py` | `xP Predicted vs Actual.png`, `xP MAE Evolution.png`, `xP R2 Evolution.png` | `predictors/hist/{pos}_preds.csv` |
| `predictor_target_comparison.py` | `R2 XP vs Points Model.png` | `predictors/hist/{pos}_preds.csv` and `predictors/hist/{pos}_preds_points.csv` |
| `hierarchical_trace_plots.py` | `{GK,DEF,MID,FWD} League Trace.png`, `{GK,DEF,MID,FWD} Manager Trace.png` | `beta estimators/estimates/league_models_{pos}`, `beta estimators/opponent betas/estimates/manager_fits_38.joblib` |
| `rolling_backtest_plot.py` | `Rolling Cumulative Points.png` | `beta estimators/opponent betas/estimates/rolling_optimization_results_fixed.joblib`, `rolled_data_24_25.csv`, `league_selections_df.csv` |
| `gw38_mean_variance_frontier.py` | `GW38 Mean Variance Frontier.png` | re-executes the Steps 1-6 walkthrough cells of `optimization.ipynb` live (Gurobi + copula fit for GW38, ~1-2 min) |

Regenerating a different game week/opponent/season for any of these just means editing
the small "parameters" block near the top of the relevant script (e.g. `GW = 38` in
`hierarchical_trace_plots.py` and `gw38_mean_variance_frontier.py`).
