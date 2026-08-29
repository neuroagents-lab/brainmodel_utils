import numpy as np
import xarray as xr
from joblib import Parallel, delayed
import copy
from brainmodel_utils.core.utils import dict_app, dict_np, make_list
from brainmodel_utils.metrics.utils import (
    sphalf_input_checker,
    str_to_metric_func,
    generic_trial_avg,
    get_splithalves,
    concat_dict_sp,
)
from brainmodel_utils.neural_mappers import PipelineNeuralMap
from brainmodel_utils.neural_mappers.utils import generate_train_test_splits


def sb_correction(r):
    # spearman brown correction for split halves
    return (2.0 * r) / (1.0 + r)


def get_consistency_per_neuron(X, Y, X1, X2, Y1, Y2, metric="pearsonr"):
    """
    Note: "r_xy_n_sb" is the main consistency metric. The rest are its components
    that are stored for convenience.
    """
    if "rsa" in metric:
        dim_val = 2
    else:
        dim_val = 1
    sphalf_input_checker(X=X, Y=Y, X1=X1, X2=X2, Y1=Y1, Y2=Y2, dim_val=dim_val)

    metric_func = str_to_metric_func(metric)
    # we use the full responses here since Spearman-Brown Correction applies to split-half self-consistency
    r_xy = metric_func(X, Y)[0]
    r_xx = metric_func(X1, X2)[0]
    r_xx_sb = sb_correction(r_xx)
    r_yy = metric_func(Y1, Y2)[0]
    r_yy_sb = sb_correction(r_yy)

    # Check if the product is negative or close to zero to avoid runtime warning
    # functionally, the code is the same as before, since when the denominator is complex or zero, then r_xy_n_sb is still NaN
    # this should be the case since the metric's derivation is theoretically undefined if r_xx and r_yy are negative,
    # as transitive property of correlation is no longer guaranteed in this case
    r_product = r_xx_sb * r_yy_sb
    if (r_product < 0) or np.isclose(r_product, 0):
        denom_sb = np.nan
        r_xy_n_sb = np.nan
    else:
        denom_sb = (r_product) ** 0.5
        r_xy_n_sb = r_xy / denom_sb

    reg_metrics = {
        "r_xy_n_sb": r_xy_n_sb,
        "r_xx": r_xx,
        "r_xx_sb": r_xx_sb,
        "r_yy": r_yy,
        "r_yy_sb": r_yy_sb,
        "r_xy": r_xy,
        "denom_sb": denom_sb,
    }
    return reg_metrics


def get_linregress_consistency_persplit(
    X,
    Y,
    X1,
    X2,
    Y1,
    Y2,
    map_kwargs,
    train_idx,
    test_idx,
    metric="pearsonr",
):
    if map_kwargs is None:
        return

    sphalf_input_checker(X=X, Y=Y, X1=X1, X2=X2, Y1=Y1, Y2=Y2)

    reg_metrics = {
        "r_xy_n_sb": [],
        "r_xx": [],
        "r_xx_sb": [],
        "r_yy": [],
        "r_yy_sb": [],
        "r_xy": [],
        "denom_sb": [],
    }

    assert isinstance(map_kwargs, dict)

    reg_metrics_train = copy.deepcopy(reg_metrics)
    reg_metrics_test = copy.deepcopy(reg_metrics)
    neural_map_full = PipelineNeuralMap(**map_kwargs)
    neural_map_1 = PipelineNeuralMap(**map_kwargs)
    neural_map_2 = PipelineNeuralMap(**map_kwargs)

    if "rsa" in metric:
        train_res = get_consistency_per_neuron(
            X=X, Y=Y, X1=X1, X2=X2, Y1=Y1, Y2=Y2,
            metric=metric,
        )
        dict_app(d=reg_metrics_train, curr=train_res)

        test_res = get_consistency_per_neuron(
            X=X, Y=Y, X1=X1, X2=X2, Y1=Y1, Y2=Y2,
            metric=metric,
        )
        dict_app(d=reg_metrics_test, curr=test_res)
    else:
        X_train, Y_train, X_test, Y_test = (
            X[train_idx],
            Y[train_idx],
            X[test_idx],
            Y[test_idx],
        )
        X1_train, Y1_train, X1_test, Y1_test = (
            X1[train_idx],
            Y1[train_idx],
            X1[test_idx],
            Y1[test_idx],
        )
        X2_train, Y2_train, X2_test, Y2_test = (
            X2[train_idx],
            Y2[train_idx],
            X2[test_idx],
            Y2[test_idx],
        )
        neural_map_full.fit(X_train, Y_train)

        Y_pred_train = neural_map_full.predict(X_train)
        assert Y_pred_train.shape == Y_train.shape
        Y_pred_test = neural_map_full.predict(X_test)
        assert Y_pred_test.shape == Y_test.shape
        neural_map_1.fit(X1_train, Y1_train)
        Y1_pred_train = neural_map_1.predict(X1_train)
        assert Y1_pred_train.shape == Y1_train.shape
        Y1_pred_test = neural_map_1.predict(X1_test)
        assert Y1_pred_test.shape == Y1_test.shape
        neural_map_2.fit(X2_train, Y2_train)
        Y2_pred_train = neural_map_2.predict(X2_train)
        assert Y2_pred_train.shape == Y2_train.shape
        Y2_pred_test = neural_map_2.predict(X2_test)
        assert Y2_pred_test.shape == Y2_test.shape
        for n in range(Y_train.shape[1]):
            curr_train_res = get_consistency_per_neuron(
                X=Y_pred_train[:, n],
                Y=Y_train[:, n],
                X1=Y1_pred_train[:, n],
                X2=Y2_pred_train[:, n],
                Y1=Y1_train[:, n],
                Y2=Y2_train[:, n],
                metric=metric,
            )
            dict_app(d=reg_metrics_train, curr=curr_train_res)

            curr_test_res = get_consistency_per_neuron(
                X=Y_pred_test[:, n],
                Y=Y_test[:, n],
                X1=Y1_pred_test[:, n],
                X2=Y2_pred_test[:, n],
                Y1=Y1_test[:, n],
                Y2=Y2_test[:, n],
                metric=metric,
            )
            dict_app(d=reg_metrics_test, curr=curr_test_res)

    dict_np(reg_metrics_train)
    dict_np(reg_metrics_test)
    # neurons length vector
    return {"train": reg_metrics_train, "test": reg_metrics_test}


def get_linregress_consistency_persphalftrial(
    source,
    target,
    map_kwargs,
    splits=None,
    num_train_test_splits=5,
    train_frac=0.8,
    metric="pearsonr",
    sphseed=0,
):

    if source.ndim == 3:
        X = generic_trial_avg(source)
    else:
        # e.g. model features (stim x units)
        assert source.ndim == 2
        X = source
    # when fully trial averaged, needs to not contain any NaNs
    assert np.isfinite(X).all()
    X1, X2 = get_splithalves(source, seed=sphseed)
    # skip any splithalves that have NaN trials
    if (not np.isfinite(X1).all()) or (not np.isfinite(X2).all()):
        return
    # target is always neural data, so there is a trials dimension
    assert target.ndim == 3
    Y = generic_trial_avg(target)
    # when fully trial averaged, needs to not contain any NaNs
    assert np.isfinite(Y).all()
    Y1, Y2 = get_splithalves(target, seed=sphseed)
    # skip any splithalves that have NaN trials
    if (not np.isfinite(Y1).all()) or (not np.isfinite(Y2).all()):
        return

    if splits is None:
        splits = generate_train_test_splits(
            num_stim=X.shape[0],
            num_splits=num_train_test_splits,
            train_frac=train_frac,
        )
    assert isinstance(splits, list)

    if not isinstance(map_kwargs, list):
        assert isinstance(map_kwargs, dict)
        map_kwargs = make_list(map_kwargs, num_times=len(splits))
    results_arr = [
        get_linregress_consistency_persplit(
            X=X,
            Y=Y,
            X1=X1,
            X2=X2,
            Y1=Y1,
            Y2=Y2,
            train_idx=s["train"],
            test_idx=s["test"],
            map_kwargs=map_kwargs[s_idx],
            metric=metric,
        )
        for s_idx, s in enumerate(splits)
    ]
    return concat_dict_sp(results_arr)


def get_linregress_consistency(
    source,
    target,
    map_kwargs,
    num_bootstrap_iters=1000,
    num_parallel_jobs=1,
    start_seed=1234,
    metric="pearsonr",
    splits=None,
    **kwargs
):
    """
    The main function for computing the linear regression consistency (noise corrected)
    between source and target.

    Arguments:
    -------------------
    source: Either model features (stimuli x units), or neural features (trials x stimuli x units)
    target: Neural features (trials x stimuli x units), usually from a different animal if the source features are neural too.
    map_kwargs: Either a dict or a list of dicts (one per train/test split) specifying the linear regression parameters.
    num_bootstrap_iters: How many split-halves to compute.
    num_parallel_jobs: Number of parallel jobs to parallelize the outermost for loop over split-half trials.
    start_seed: Starting seed for generating split halves (for reproducibility).
    metric (optional): Correlation metric (across stimuli) used per neuron.
            Supported metrics: "pearsonr", "spearmanr", "rsa_pearsonr", "rsa_spearmanr"
    splits (optional): Your own list of {"train", "test"} split indices.
            If "splits" is None, you can additionally specify train_frac and num_train_test_splits to generate your own.

    Returns (see README for full details):
    --------------------------------------
    dict
        A dictionary with two main keys, "train" and "test", each mapping to a dict containing:
        
        - "r_xy_n_sb": Noise-corrected (Spearman-Brown) per-neuron correlation 
                       indicating predictivity of the source for the target.
        - "r_xx", "r_xx_sb": Split-half correlation and Spearman-Brown–corrected
                            correlation for the source’s predicted responses.
        - "r_yy", "r_yy_sb": Split-half correlation and Spearman-Brown–corrected
                            correlation for the actual target responses.
        - "r_xy": Raw correlation between predicted and actual responses (uncorrected).
        - "denom_sb": Denominator for the noise-corrected correlation 
                      (i.e., sqrt(r_xx_sb * r_yy_sb)).

        Each of these values is shaped
        (num_bootstrap_iters, number_of_splits, number_of_units),
        or returned as an xarray.DataArray with labeled dimensions if the target is an xarray.
        The key "r_xy_n_sb" is typically the primary measure of interest.
    """
    if num_parallel_jobs == 1:
        # easier to debug in not parallel mode
        results_arr = [get_linregress_consistency_persphalftrial(
            source=source,
            target=target,
            map_kwargs=map_kwargs,
            splits=splits,
            metric=metric,
            sphseed=sphseed,
            **kwargs
        ) for sphseed in range(start_seed, start_seed + num_bootstrap_iters)]
    else:
        results_arr = Parallel(n_jobs=num_parallel_jobs)(
            delayed(get_linregress_consistency_persphalftrial)(
                source=source,
                target=target,
                map_kwargs=map_kwargs,
                splits=splits,
                metric=metric,
                sphseed=sphseed,
                **kwargs
            )
            for sphseed in range(start_seed, start_seed + num_bootstrap_iters)
        )
    # we format the results as an xarray matching the units dimension of the target
    if isinstance(target, xr.DataArray):
        results_dict = concat_dict_sp(
            results_arr,
            xarray_target=target,
            xarray_dims=["trial_bootstrap_iters", "train_test_splits", "units"],
        )
    else:
        results_dict = concat_dict_sp(results_arr)
    return results_dict
