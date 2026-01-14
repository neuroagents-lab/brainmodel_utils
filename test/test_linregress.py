from brainmodel_utils.metrics.consistency import get_linregress_consistency
import numpy as np

from utils.linregress_utils import make_splits, get_rxy_ryy_filtered

np.random.seed(42)
NUM_STIMULI = 6

def _test_get_linregress_consistency(**linregress_kwargs):
    # make highly correlated dummy data
    model_data = np.random.randn(NUM_STIMULI, 14)
    noise = np.random.randn(100, NUM_STIMULI, 14) * 0.1  # small noise
    animal_neurons = model_data[None, :, :] + noise

    kwargs = dict(
        source=model_data,
        target=animal_neurons,
        num_parallel_jobs=1,
        num_bootstrap_iters=10,
    )
    kwargs.update(linregress_kwargs)
    results = get_linregress_consistency(**kwargs)

    assert type(results) == dict
    assert "train" in results and "test" in results
    assert "r_xy_n_sb" in results["test"]
    rxysb = results["test"]["r_xy_n_sb"]
    assert rxysb.ndim == 3

    print("\nmetric", kwargs["metric"])
    print(rxysb.shape)

    return results


def test_ridge():
    results = _test_get_linregress_consistency(
        metric="pearsonr",
        splits=make_splits(0.5, num_stimuli=NUM_STIMULI),
        map_kwargs={
            "map_type": "sklinear",
            "map_kwargs": {
                "regression_type": "Ridge",
                "regression_kwargs": { "alpha": 1 } }}
    )
    median_rxy = np.nanmedian(np.nanmean(results["test"]["r_xy_n_sb"], axis=(0, 1)))
    median_ryy = np.nanmedian(np.nanmean(results["test"]["r_yy_sb"], axis=(0, 1)))
    print(f"median rxy: {median_rxy}")
    print(f"median ryy: {median_ryy}")


def test_rsa():
    result = _test_get_linregress_consistency(
        metric="rsa_pearsonr",
        map_kwargs={ "map_type": "identity" },
        splits=make_splits(0, num_stimuli=NUM_STIMULI),
    )
    print(f"median rxysb: {np.nanmedian(result["test"]["r_xy_n_sb"])}")

    result = _test_get_linregress_consistency(
        metric="rsa_spearmanr",
        map_kwargs={ "map_type": "identity" },
        splits=make_splits(0, num_stimuli=NUM_STIMULI),
    )
    print(f"median rxysb: {np.nanmedian(result["test"]["r_xy_n_sb"])}")