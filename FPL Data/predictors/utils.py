import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pycaret.regression import *


from scipy.stats import loguniform, uniform, randint

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet

# Optional libraries — only used if installed
try:
    from xgboost import XGBRegressor
    HAS_XGB = False#True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

import joblib

def select_features_by_correlation(df, candidate_feats, target_col='xP', redundancy_threshold=0.6, min_corr=0.0, verbose=True):
    """
    Automatic feature selection in two steps:
      1. Rank candidate_feats by |correlation| with target_col, descending
         -- strongest predictors of xP first.
      2. Walk down the ranked list, greedily keeping a feature only if its
         |correlation| with every already-selected feature is below
         redundancy_threshold. A feature that's individually predictive of
         xP but redundant with something already kept (i.e. it's not
         adding new information) gets dropped.

    This is a filter-based analogue of stepwise selection: it favors
    features that are both informative about the target and
    non-redundant with each other, without needing to actually fit the
    (expensive) Dirichlet model repeatedly.

    Parameters
    ----------
    df                    : DataFrame containing candidate_feats + target_col
    candidate_feats       : list of column names to consider
    target_col            : column to rank relevance against (e.g. 'xP')
    redundancy_threshold  : drop a feature if |corr| with an already-kept
                             feature exceeds this (0-1). Lower = stricter
                             (keeps fewer, more independent features).
    min_corr              : also require |corr| with target_col to exceed
                             this floor to be considered at all (0-1).
                             Default 0.0 = no relevance floor, matching a
                             pure rank-then-deduplicate selection.

    Returns
    -------
    X_feats : list of selected feature names, in original candidate order
    """
    # guard against duplicate feature names -- these make every downstream
    # label-based lookup (ranked[feat], feat_corr.loc[feat, ...]) ambiguous,
    # since pandas returns a Series instead of a scalar for a repeated label
    seen = set()
    deduped = []
    for f in candidate_feats:
        if f in seen:
            if verbose:
                print(f"  dropping duplicate entry {f!r} from candidate_feats")
            continue
        seen.add(f)
        deduped.append(f)
    # candidate_feats = deduped

    df_valid = df[candidate_feats].dropna()

    if len(df_valid) < len(df):
        dropped = len(df) - len(df_valid)
        if verbose:
            print(f"  dropped {dropped} rows with NaNs before computing correlations")

    target_corr = df_valid[candidate_feats].corr(method="pearson")["xP"].abs()#.sort_values(ascending=False).abs() #.corrwith(df_valid[target_col]).abs()
    ranked = target_corr.sort_values(ascending=False)

    feat_corr = df_valid[candidate_feats].corr().abs()

    selected = []
    for feat in ranked.index:
        if pd.isna(ranked[feat]):
            if verbose:
                print(f"skipping {feat!r} -- undefined correlation with {target_col} "
                      f"(likely constant column)")
            continue
        if ranked[feat] < min_corr:
            if verbose:
                print(f"  skipping {feat!r} -- |corr| with {target_col} "
                      f"({ranked[feat]:.3f}) below min_corr={min_corr}")
            continue
        if not selected:
            selected.append(feat)
            continue
        max_corr_with_selected = feat_corr.loc[feat, selected].max()
        if max_corr_with_selected < redundancy_threshold:
            selected.append(feat)
        elif verbose:
            worst = feat_corr.loc[feat, selected].idxmax()
            print(f"  dropping {feat!r} (|corr| with {target_col}={target_corr[feat]:.3f}) "
                  f"-- |corr|={max_corr_with_selected:.3f} with already-selected {worst!r}")

    # restore original candidate order for readability/downstream indexing
    X_feats = [f for f in candidate_feats if f in selected]

    if verbose:
        print(f"\nSelected {len(X_feats)} / {len(candidate_feats)} features "
              f"(ranked by |corr| with {target_col!r}, redundancy_threshold={redundancy_threshold}, "
              f"min_corr={min_corr}):")
        print(X_feats)

    return X_feats

# --------------------------------------------------------------------------
# 1. Chronological split
# --------------------------------------------------------------------------
def chronological_split(df, gw, target_col="xP", drop_cols=("xP", "round", "element")):
    """Train on rounds < gw, test on round == gw. No random shuffling —
    this mirrors how the model will actually be used (predict an unseen
    upcoming gameweek from past ones)."""
    df_train = df[df["round"] < gw]
    df_test = df[df["round"] == gw]

    if df_train.empty or df_test.empty:
        raise ValueError(f"Empty train or test split for gw={gw}. Check round range.")

    y_train = df_train[target_col]
    y_test = df_test[target_col]
    X_train = df_train.drop(columns=list(drop_cols))
    X_test = df_test.drop(columns=list(drop_cols))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_test_scaled, y_train, y_test

# --------------------------------------------------------------------------
# 2. Model zoo
# --------------------------------------------------------------------------
def build_models(random_state=42):
    """Each model is wrapped in TransformedTargetRegressor(log1p/expm1).
    A shift constant `c` is applied upstream in evaluate_all() so that
    the target is guaranteed positive before log1p is applied."""
    models = {
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
        "Ridge": Ridge(alpha=1.0)
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=random_state, n_jobs=-1, verbosity=0,
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1, verbosity=-1
        )
    return models

# --------------------------------------------------------------------------
# 3. Evaluate all models on one chronological split
# --------------------------------------------------------------------------
def evaluate_all(df, gw, target_col="xP", drop_cols=("xP", "round", "element"),
                  use_log_transform=True, verbose=True):
    X_train, X_test, y_train, y_test = chronological_split(df, gw, target_col, drop_cols)

    # Shift constant computed from TRAIN only (avoids test-set leakage).
    # Buffer of 1.0 instead of 0.1 gives more headroom near the log1p boundary.
    c = abs(y_train.min()) + 1.0 if use_log_transform else 0.0
    y_train_shifted = y_train + c

    results = []
    fitted_models = {}

    for name, base_model in build_models().items():
        if use_log_transform:
            model = TransformedTargetRegressor(
                regressor=base_model, func=np.log1p, inverse_func=np.expm1
            )
        else:
            model = base_model

        try:
            model.fit(X_train, y_train_shifted if use_log_transform else y_train)
            preds = model.predict(X_test)
            preds_original = preds - c if use_log_transform else preds
        except Exception as e:
            if verbose:
                print(f"[{name}] failed: {e}")
            continue

        mae = mean_absolute_error(y_test, preds_original)
        rmse = np.sqrt(mean_squared_error(y_test, preds_original))
        r2 = r2_score(y_test, preds_original)

        results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        fitted_models[name] = model

        if verbose:
            print(f"{name:>22s} | MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

    results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    return results_df, fitted_models

# --------------------------------------------------------------------------
# 4. (Optional) evaluate across several gameweeks and average
# --------------------------------------------------------------------------
def evaluate_across_gws(df, gw_list, **kwargs):
    """Run evaluate_all() for each gw in gw_list and return per-gw results
    plus a summary averaged across gameweeks. Useful because a single
    train/test split can be noisy — averaging over several rounds gives
    a more stable comparison between models."""
    all_rows = []
    for gw in gw_list:
        try:
            res, _ = evaluate_all(df, gw, verbose=False, **kwargs)
        except ValueError as e:
            print(f"Skipping gw={gw}: {e}")
            continue
        res["gw"] = gw
        all_rows.append(res)

    if not all_rows:
        raise ValueError("No valid gameweeks evaluated.")

    combined = pd.concat(all_rows, ignore_index=True)
    summary = (
        combined.groupby("model")[["MAE", "RMSE", "R2"]]
        .mean()
        .sort_values("MAE")
        .reset_index()
    )
    return combined, summary

"""
Hyperparameter-tune whichever model came out on top in evaluate_across_gws()
(from compare_models.py), using CV folds that respect gameweek order —
no random K-fold, since that would let future rounds leak into training
during the search itself.

Usage:
    from compare_models import evaluate_across_gws
    combined, summary = evaluate_across_gws(mid_df, gw_list=range(4, 9))

    best_model, search = tune_best_model(
        mid_df, summary, val_gws=range(4, 9), final_test_gw=9
    )
"""

# --------------------------------------------------------------------------
# 4. Search spaces, keyed by the same names used in build_models()
# --------------------------------------------------------------------------
PARAM_DISTRIBUTIONS = {
    "HistGradientBoosting": {
        "regressor__learning_rate": loguniform(0.01, 0.3),
        "regressor__max_iter": randint(100, 600),
        "regressor__max_leaf_nodes": randint(15, 127),
        "regressor__max_depth": [None, 3, 5, 7, 10],
        "regressor__min_samples_leaf": randint(5, 60),
        "regressor__l2_regularization": loguniform(1e-3, 10),
    },
    "RandomForest": {
        "regressor__n_estimators": randint(100, 600),
        "regressor__max_depth": [None, 5, 10, 20, 30],
        "regressor__min_samples_split": randint(2, 15),
        "regressor__min_samples_leaf": randint(1, 10),
        "regressor__max_features": ["sqrt", "log2", None],
    },
    "GradientBoosting": {
        "regressor__n_estimators": randint(100, 400),
        "regressor__learning_rate": loguniform(0.01, 0.3),
        "regressor__max_depth": randint(2, 6),
        "regressor__subsample": uniform(0.6, 0.4),
        "regressor__min_samples_leaf": randint(1, 20),
    },
    "Ridge": {
        "regressor__alpha": loguniform(0.01, 100),
    },
    "XGBoost": {
        "regressor__n_estimators": randint(100, 600),
        "regressor__max_depth": randint(3, 9),
        "regressor__learning_rate": loguniform(0.01, 0.3),
        "regressor__subsample": uniform(0.6, 0.4),
        "regressor__colsample_bytree": uniform(0.6, 0.4),
        "regressor__reg_alpha": loguniform(1e-3, 10),
        "regressor__reg_lambda": loguniform(1e-3, 10),
    },
    "LightGBM": {
        "regressor__n_estimators": randint(100, 600),
        "regressor__num_leaves": randint(15, 127),
        "regressor__learning_rate": loguniform(0.01, 0.3),
        "regressor__subsample": uniform(0.6, 0.4),
        "regressor__colsample_bytree": uniform(0.6, 0.4),
        "regressor__reg_alpha": loguniform(1e-3, 10),
        "regressor__reg_lambda": loguniform(1e-3, 10),
    },
}

# --------------------------------------------------------------------------
# 5. Chronological CV folds (expanding window over rounds)
# --------------------------------------------------------------------------
def rolling_origin_splits(rounds_array, val_gws):
    """Given the 'round' values aligned positionally with X, build
    (train_idx, test_idx) pairs for each validation gameweek: train on
    everything before it, test on it. Mirrors how the model will
    actually be deployed (predict the next unseen gameweek)."""
    rounds_array = np.asarray(rounds_array)
    splits = []
    for gw in val_gws:
        train_idx = np.where(rounds_array < gw)[0]
        test_idx = np.where(rounds_array == gw)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    if not splits:
        raise ValueError("No valid folds — check val_gws against available rounds.")
    return splits

# --------------------------------------------------------------------------
# 6. Tune the best model from the comparison summary
# --------------------------------------------------------------------------
def tune_best_model(
    df,
    summary_df,
    val_gws,
    final_test_gw,
    target_col="xP",
    drop_cols=("xP", "round", "element"),
    n_iter=40,
    random_state=42,
    verbose=True,
):
    best_name = summary_df.sort_values("MAE").iloc[0]["model"]

    if best_name not in PARAM_DISTRIBUTIONS:
        raise ValueError(
            f"No param grid defined for '{best_name}'. "
            f"Add one to PARAM_DISTRIBUTIONS."
        )
    if verbose:
        print(f"Best model from comparison: {best_name}")

    # CV pool = everything strictly before the held-out final test gw
    pool = df[df["round"] < final_test_gw].reset_index(drop=True)
    X_pool = pool.drop(columns=list(drop_cols))
    y_pool = pool[target_col]
    rounds_pool = pool["round"]

    # Shift constant from the CV pool only — final_test_gw stays untouched
    c = abs(y_pool.min()) + 1.0
    y_pool_shifted = y_pool + c

    cv_splits = rolling_origin_splits(rounds_pool, val_gws)

    base_model = build_models(random_state=random_state)[best_name]
    wrapped = TransformedTargetRegressor(
        regressor=base_model, func=np.log1p, inverse_func=np.expm1
    )

    search = RandomizedSearchCV(
        estimator=wrapped,
        param_distributions=PARAM_DISTRIBUTIONS[best_name],
        n_iter=n_iter,
        cv=cv_splits,
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=-1,
        verbose=1 if verbose else 0,
    )

    search.fit(X_pool, y_pool_shifted)

    if verbose:
        print(f"\nBest CV MAE: {-search.best_score_:.3f}")
        print("Best params:")
        for k, v in search.best_params_.items():
            print(f"  {k}: {v}")

    # Final check: refit best estimator on full pool, evaluate on the
    # untouched holdout gameweek
    X_train_final, X_holdout, y_train_final, y_holdout = chronological_split(
        df, final_test_gw, target_col, drop_cols
    )

    c_final = abs(y_train_final.min()) + 1.0
    best_estimator = search.best_estimator_
    best_estimator.fit(X_train_final, y_train_final + c_final)
    preds = best_estimator.predict(X_holdout) - c_final

    mae = mean_absolute_error(y_holdout, preds)
    rmse = np.sqrt(mean_squared_error(y_holdout, preds))
    r2 = r2_score(y_holdout, preds)

    if verbose:
        print(f"\nHoldout (round {final_test_gw}) performance:")
        print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

    return best_name, best_estimator, search, c_final, mae, rmse, r2
