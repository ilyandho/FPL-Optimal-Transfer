import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az

import matplotlib.pyplot as plt


def select_features_by_correlation(df, candidate_feats, target_col='xP',
                                    redundancy_threshold=0.8, min_corr=0.0,
                                    verbose=True):
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
    df_valid = df[candidate_feats + [target_col]].dropna()
    if len(df_valid) < len(df):
        dropped = len(df) - len(df_valid)
        if verbose:
            print(f"  dropped {dropped} rows with NaNs before computing correlations")

    target_corr = df_valid[candidate_feats].corrwith(df_valid[target_col]).abs()
    ranked = target_corr.sort_values(ascending=False)

    feat_corr = df_valid[candidate_feats].corr().abs()

    selected = []
    for feat in ranked.index:
        if pd.isna(ranked[feat]):
            if verbose:
                print(f"  skipping {feat!r} -- undefined correlation with {target_col} "
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


"""
Cross-sectional league baseline: fit -> standardize -> prune -> refit pipeline,
generalized across positions (g, d, m, f). Wraps the exact recipe that worked
for goalkeepers so it can be reused for d/m/f without re-deriving it each time.
"""
def fit_dirichlet_regression(X, select_probs, n_features, sigma=1.0,
                              draws=2000, tune=2000, chains=4, cores=4,
                              target_accept=0.99, max_treedepth=12,
                              random_seed=None):
    """
    One NUTS fit of p ~ Dirichlet(exp(X @ beta)), beta ~ Normal(0, sigma).
    Returns (trace, summary_df).
    """
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0.0, sigma=sigma, shape=n_features) # pm.Flat("beta", shape=n_features)

        alpha = pm.math.exp(pt.dot(X, beta))
        alpha = pt.clip(alpha, 1e-6, 1e6)
        pm.Dirichlet("p", a=alpha, observed=select_probs)

        trace = pm.sample(
            draws, tune=tune, chains=chains, cores=cores,
            target_accept=target_accept, max_treedepth=max_treedepth,
            random_seed=random_seed

        )

    summary = az.summary(trace, var_names=["beta"])
    n_div = int(trace.sample_stats["diverging"].sum())
    print(f"  divergences: {n_div}")
    if n_div > 0:
        print("  WARNING: divergences present -- treat this fit as provisional")
    return trace, summary


def _hdi_overlap(lo1, hi1, lo2, hi2):
    """
    Fraction of the *narrower* interval's width that's covered by the
    overlap with the other interval. 1.0 = one interval fully contains
    (or exactly equals) the other; 0.0 = no overlap at all.
    """
    overlap = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    narrower_width = min(hi1 - lo1, hi2 - lo2)
    if narrower_width <= 0:
        return 0.0
    return overlap / narrower_width


def prune_by_hdi(summary, feats, overlap_threshold=0.8, verbose=True):
    """
    Keep only features whose 94% HDI excludes zero, AND deduplicate
    features whose HDIs are near-identical to one another (a sign the
    two are capturing essentially the same effect -- often leftover
    collinearity that survived the raw-feature correlation check,
    since it can also arise from *combinations* of correlated features
    rather than a single pairwise correlation).

    Greedy strategy: process candidates in order of decreasing |mean|
    (strongest effect first). A candidate is dropped if its HDI overlaps
    an already-kept feature's HDI by more than `overlap_threshold`
    (as a fraction of the narrower interval's width) -- i.e. the weaker
    of the two near-duplicate effects is the one that goes.

    Returns (kept_feats, kept_indices).
    """
    mask = (summary['hdi_3%'] > 0) | (summary['hdi_97%'] < 0)
    sig = summary[mask].copy()
    labels = sig.index.tolist()
    indices = [int(lbl.split('[')[1].split(']')[0]) for lbl in labels]

    if len(sig) == 0:
        return [], []

    # process strongest effect first, so a redundant *weaker* duplicate
    # is the one dropped rather than an arbitrary member of the pair
    order = np.argsort(-sig['mean'].abs().to_numpy())

    kept_local = []
    for i in order:
        lo_i, hi_i = sig['hdi_3%'].iloc[i], sig['hdi_97%'].iloc[i]
        redundant_with = None
        for k in kept_local:
            lo_k, hi_k = sig['hdi_3%'].iloc[k], sig['hdi_97%'].iloc[k]
            if _hdi_overlap(lo_i, hi_i, lo_k, hi_k) > overlap_threshold:
                redundant_with = k
                break
        if redundant_with is None:
            kept_local.append(i)
        elif verbose:
            print(f"  dropping {feats[indices[i]]!r} (label {labels[i]}) -- "
                  f"HDI near-duplicate of {feats[indices[redundant_with]]!r} "
                  f"(label {labels[redundant_with]})")

    kept_local = sorted(kept_local)  # restore original feature order
    kept_indices = [indices[i] for i in kept_local]
    kept_feats = [feats[i] for i in kept_indices]
    return kept_feats, kept_indices


def fit_position_pipeline(player_data, feats, gw,  select_probs_col='league_select_probs',
                           position_name='', eps=1e-6, random_seed=42):
    """
    Full pipeline for one position:
      1. Build X (raw features) and select_probs from player_data.
      2. Standardize X.
      3. Fit Dirichlet regression with a weakly-informative prior.
      4. Prune features whose 94% HDI includes zero.
      5. Re-standardize on the pruned feature set and refit.

    Returns a dict with everything needed to reuse this model later
    (predicting alpha_{j,T}^league on a new week's features).
    """
    print(f"\n{'=' * 60}\nPosition: {position_name}\n{'=' * 60}")

    # --- 1. raw data -----------------------------------------------------
    select_probs = player_data[select_probs_col] + eps
    select_probs = (select_probs / select_probs.sum()).to_numpy(dtype=float)

    X_raw = player_data[feats].to_numpy(dtype=float)
    assert not np.isnan(X_raw).any(), f"NaNs in features for {position_name}"
    assert not np.isinf(X_raw).any(), f"Infs in features for {position_name}"
    assert X_raw.shape[0] == select_probs.shape[0], "row mismatch"

    n_players, n_features = X_raw.shape
    print(f"n_players={n_players}, n_features={n_features}")

    # --- 2. standardize ----------------------------------------------------
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std[X_std == 0] = 1.0
    X = (X_raw - X_mean) / X_std


    # import numpy as np

    # X = data[feats].to_numpy(dtype='float64')

    # print(X)

    rank = np.linalg.matrix_rank(X)
    n_features = X.shape[1]
    cond_number = np.linalg.cond(X)

    print(f"X shape: {X.shape}")
    print(f"rank: {rank}  (full rank would be {n_features})")
    print(f"condition number: {cond_number:.2e}")
    print("  (as a rough guide: >1e3 is concerning, >1e6 is basically singular)")

    if rank < n_features:
        print(f"\nWARNING: X is rank-deficient by {n_features - rank} -- "
            f"there is at least one exact linear dependency among your "
            f"standardized features. This alone can cause exactly the "
            f"single-chain-catastrophically-diverges pattern you're seeing, "
            f"independent of the prior.")

    # pairwise check for near-exact duplicates (|corr| very close to 1, not just > 0.8)
    corr = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(corr, 0)
    near_dupe = np.argwhere(np.abs(corr) > 0.999)
    if len(near_dupe):
        print("\nNear-exact duplicate columns (|corr| > 0.999):")
        seen = set()
        for i, j in near_dupe:
            if i < j and (i, j) not in seen:
                seen.add((i, j))
                print(f"  columns {i} and {j}: r={corr[i, j]:.6f}")



    # --- 3. initial fit ------------------------------------------------------
    print("Fitting initial model...")
    trace, summary = fit_dirichlet_regression(
        X, select_probs, n_features, random_seed=random_seed
    )



    # az.plot_trace(trace, combined=True)

    # fig = plt.gcf()
    # fig.set_size_inches(14, 3)

    # plt.suptitle(f"Innitial trace Plot of β Coefficients for {position_name} in gw {gw}", y=1.02)
    # plt.tight_layout()
    # plt.show()
    # --- 4. prune ------------------------------------------------------------
    feats_kept, idx_kept = prune_by_hdi(summary, feats)
    print(f"Kept {len(feats_kept)} / {len(feats)} features: {feats_kept}")

    if len(feats_kept) == 0:
        print("WARNING: no features survived pruning -- keeping full set instead")
        feats_kept, idx_kept = feats, list(range(n_features))

    # --- 5. refit on pruned features ------------------------------------------
    X_raw_kept = X_raw[:, idx_kept]
    X_mean_kept = X_raw_kept.mean(axis=0)
    X_std_kept = X_raw_kept.std(axis=0)
    X_std_kept[X_std_kept == 0] = 1.0
    X_kept = (X_raw_kept - X_mean_kept) / X_std_kept

    print("Refitting on pruned feature set...")
    trace_final, summary_final = fit_dirichlet_regression(
        X_kept, select_probs, len(feats_kept), random_seed=random_seed
    )

    # # axes = az.plot_trace(trace_final, combined=True)
    # az.plot_trace(trace_final, combined=True)
    # fig = plt.gcf()
    # fig.set_size_inches(14, 3)

    # plt.suptitle(f"Final trace Plot of β Coefficients for {position_name} in gw {gw}", y=1.02)
    # plt.tight_layout()
    # plt.show()


    return {
        "feats": feats_kept,
        "X_mean": X_mean_kept,
        "X_std": X_std_kept,
        "trace": trace,
        "summary": summary,
        "trace_final": trace_final,
        "summary_final": summary_final,
        "beta_mean": trace_final.posterior["beta"].mean(dim=("chain", "draw")).values,
    }


def alpha_league(fit_result, X_new_raw):
    """
    Apply a fitted position model (dict returned by fit_position_pipeline) to
    new-week raw features, standardizing with the model's *training*
    mean/std, per eq:alpha_league.
    """
    X_std_new = (X_new_raw - fit_result["X_mean"]) / fit_result["X_std"]
    linear = np.clip(X_std_new @ fit_result["beta_mean"], -30, 30)
    return np.exp(linear)

