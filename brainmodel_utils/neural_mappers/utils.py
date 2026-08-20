import numpy as np
import xarray as xr


def map_from_str(map_type):
    if map_type.lower() == "pls":
        from brainmodel_utils.neural_mappers import PLSNeuralMap

        return PLSNeuralMap
    elif map_type.lower() == "percentile":
        from brainmodel_utils.neural_mappers import PercentileNeuralMap

        return PercentileNeuralMap
    elif map_type.lower() == "identity":
        from brainmodel_utils.neural_mappers import IdentityNeuralMap

        return IdentityNeuralMap
    elif map_type.lower() == "sklinear":
        from brainmodel_utils.neural_mappers import SKLinearNeuralMap

        return SKLinearNeuralMap
    else:
        raise ValueError(f"{map_type.lower()} is not supported.")


def generate_train_test_splits(
    num_stim, num_splits=5, train_frac=0.8, start_seed=0,
):
    if train_frac > 0:
        train_test_splits = []
        for s in range(start_seed, start_seed + num_splits):
            rand_idx = np.random.RandomState(seed=s).permutation(num_stim)
            num_train = (int)(np.ceil(train_frac * len(rand_idx)))
            train_idx = rand_idx[:num_train]
            test_idx = rand_idx[num_train:]

            curr_sp = {"train": train_idx, "test": test_idx}
            train_test_splits.append(curr_sp)
    else:
        print("Train fraction is 0, make sure your map has no parameters!")
        # we apply no random permutation in this case as there is no training of parameters
        # (e.g. rsa)
        train_test_splits = [
            {"train": np.array([], dtype=int), "test": np.arange(num_stim)}
        ]
    return train_test_splits


def convert_dict_to_tuple(d):
    assert isinstance(d, dict)
    d_tuple = []
    for k, v in d.items():
        assert not isinstance(k, dict)
        if isinstance(v, dict):
            d_tuple.append(tuple([k, convert_dict_to_tuple(v)]))
        else:
            d_tuple.append(tuple([k, v]))
    d_tuple = tuple(d_tuple)
    return d_tuple


def convert_tuple_to_dict(d_tuple):
    assert isinstance(d_tuple, tuple)
    d = dict(d_tuple)
    for k, v in d.items():
        if isinstance(v, tuple):
            d[k] = convert_tuple_to_dict(v)
        else:
            d[k] = v
    return d


def get_cv_best_params(results, metric="r_xy_n_sb", verbose=False, params_as_dict=True):
    assert isinstance(results, list) or isinstance(
        results, np.ndarray
    )  # of length num_train_test_splits
    num_splits = len(results)
    animals = list(results[0].keys())
    parameters = list(results[0][animals[0]].keys())
    split_skip = []
    for s in range(num_splits):
        skip = False
        for a in animals:
            curr_res = results[s][a][parameters[0]]["test"][metric]
            if isinstance(curr_res, list) and (curr_res == []):
                skip |= True
            else:
                skip &= False
        if skip:
            split_skip.append(s)
    assert len(split_skip) < num_splits
    for s in range(num_splits):
        if s not in split_skip:
            exemplar_result = results[s][animals[0]][parameters[0]]["test"][metric]
            assert (
                exemplar_result.ndim == 3
            )  # trials x num_cv_train_test_splits x units
            res_xarray = isinstance(exemplar_result, xr.DataArray)

    map_kwargs = []
    # for every train/test split, we find the best cross validated parameters
    for s in range(num_splits):
        if s in split_skip:
            map_kwargs.append(None)
            continue
        best_res = -np.inf
        best_params = None
        for curr_param in parameters:
            curr_res_animals = [
                results[s][a][curr_param]["test"][metric] for a in animals
            ]
            if res_xarray:
                curr_pop_res = xr.concat(curr_res_animals, dim="units")
                assert curr_pop_res.ndim == 3
                curr_res = curr_pop_res.mean(dim="trial_bootstrap_iters")
                # note: these are the cross validated train/test splits for that train/test split
                curr_res = curr_res.mean(dim="train_test_splits")
                curr_res = curr_res.median(dim="units", skipna=True)
                curr_res = curr_res.data
            else:
                curr_pop_res = np.concatenate(curr_res_animals, axis=-1)
                assert curr_pop_res.ndim == 3
                # average across bootstrap iters
                curr_res = np.mean(curr_pop_res, axis=0)
                # average across cv train_test_splits
                curr_res = np.mean(curr_pop_res, axis=0)
                # median across neurons
                curr_res = np.nanmedian(curr_res, axis=-1)

            if curr_res > best_res:
                best_res = curr_res
                best_params = curr_param

        if params_as_dict and not isinstance(best_params, dict):
            # assumes we saved it originally as an immutable type
            assert isinstance(best_params, tuple)
            best_params = convert_tuple_to_dict(best_params)
        if verbose:
            print(f"Split: {s}, best result: {best_res}, best params: {best_params}")
        map_kwargs.append(best_params)
    return map_kwargs
