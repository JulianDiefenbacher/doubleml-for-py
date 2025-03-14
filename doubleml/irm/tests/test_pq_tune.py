import math

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_pq_manual import fit_pq, tune_nuisance_pq


@pytest.fixture(scope="module", params=[0, 1])
def treatment(request):
    return request.param


@pytest.fixture(scope="module", params=[0.25, 0.5, 0.75])
def quantile(request):
    return request.param


@pytest.fixture(scope="module", params=[RandomForestClassifier(max_depth=5, random_state=42)])
def learner_g(request):
    return request.param


@pytest.fixture(scope="module", params=[RandomForestClassifier(max_depth=5, random_state=42)])
def learner_m(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def tune_on_folds(request):
    return request.param


def get_par_grid(learner):
    if learner.__class__ in [RandomForestClassifier]:
        par_grid = {"n_estimators": [5, 10, 15, 20]}
    else:
        par_grid = {}
    return par_grid


@pytest.fixture(scope="module")
def dml_pq_fixture(generate_data_quantiles, treatment, quantile, learner_g, learner_m, normalize_ipw, tune_on_folds):
    par_grid = {"ml_g": get_par_grid(learner_g), "ml_m": get_par_grid(learner_m)}
    n_folds_tune = 4
    n_folds = 2

    # collect data
    (x, y, d) = generate_data_quantiles
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)
    np.random.seed(42)
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)
    smpls = all_smpls[0]

    np.random.seed(42)
    dml_pq_obj = dml.DoubleMLPQ(
        obj_dml_data,
        clone(learner_g),
        clone(learner_m),
        treatment=treatment,
        quantile=quantile,
        n_folds=n_folds,
        n_rep=1,
        normalize_ipw=normalize_ipw,
        trimming_threshold=0.01,
        draw_sample_splitting=False,
    )

    # synchronize the sample splitting
    dml_pq_obj.set_sample_splitting(all_smpls=all_smpls)
    # tune hyperparameters
    np.random.seed(42)
    tune_res = dml_pq_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune, return_tune_res=False)
    assert isinstance(tune_res, dml.DoubleMLPQ)

    np.random.seed(42)
    dml_pq_obj.fit()

    np.random.seed(42)
    if tune_on_folds:
        g_params, m_params = tune_nuisance_pq(
            y,
            x,
            d,
            clone(learner_g),
            clone(learner_m),
            smpls,
            treatment,
            quantile,
            n_folds_tune,
            par_grid["ml_g"],
            par_grid["ml_m"],
        )
    else:
        xx = [(np.arange(len(y)), np.array([]))]
        g_params, m_params = tune_nuisance_pq(
            y,
            x,
            d,
            clone(learner_g),
            clone(learner_m),
            xx,
            treatment,
            quantile,
            n_folds_tune,
            par_grid["ml_g"],
            par_grid["ml_m"],
        )

        g_params = g_params * n_folds
        m_params = m_params * n_folds

    np.random.seed(42)
    res_manual = fit_pq(
        y,
        x,
        d,
        quantile,
        learner_g=clone(learner_g),
        learner_m=clone(learner_m),
        all_smpls=all_smpls,
        treatment=treatment,
        n_rep=1,
        trimming_threshold=0.01,
        normalize_ipw=normalize_ipw,
        g_params=g_params,
        m_params=m_params,
    )

    res_dict = {
        "coef": dml_pq_obj.coef.item(),
        "coef_manual": res_manual["pq"],
        "se": dml_pq_obj.se.item(),
        "se_manual": res_manual["se"],
    }

    return res_dict


@pytest.mark.ci
def test_dml_pq_coef(dml_pq_fixture):
    assert math.isclose(dml_pq_fixture["coef"], dml_pq_fixture["coef_manual"], rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_pq_se(dml_pq_fixture):
    assert math.isclose(dml_pq_fixture["se"], dml_pq_fixture["se_manual"], rel_tol=1e-9, abs_tol=1e-4)
