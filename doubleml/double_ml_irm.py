import patsy
import numpy as np
import statsmodels.api as sm
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm

from .double_ml import DoubleML
from ._utils import _dml_cv_predict, _get_cond_smpls, _dml_tune, _check_finite_predictions, _calculate_bootstrap_tstat,\
    _polynomial_fit, _splines_fit, _calculate_orthogonal_polynomials, _create_regressor_grid_gate,\
    _calculate_bootstrap_tstat_gate


class DoubleMLIRM(DoubleML):
    """Double machine learning for interactive regression models

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`g_0(D,X) = E[Y|X,D]`.
        For a binary outcome variable :math:`Y` (with values 0 and 1), a classifier implementing ``fit()`` and
        ``predict_proba()`` can also be specified. If :py:func:`sklearn.base.is_classifier` returns ``True``,
        ``predict_proba()`` is used otherwise ``predict()``.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`m_0(X) = E[D|X]`.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitons for the sample splitting.
        Default is ``1``.

    score : str or callable
        A str (``'ATE'`` or ``'ATTE'``) specifying the score function
        or a callable object / function with signature ``psi_a, psi_b = score(y, d, g_hat0, g_hat1, m_hat, smpls)``.
        Default is ``'ATE'``.

    dml_procedure : str
        A str (``'dml1'`` or ``'dml2'``) specifying the double machine learning algorithm.
        Default is ``'dml2'``.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-12``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    apply_cross_fitting : bool
        Indicates whether cross-fitting should be applied.
        Default is ``True``.

    Examples
    --------
    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml.datasets import make_irm_data
    >>> from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    >>> np.random.seed(3141)
    >>> ml_g = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> ml_m = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
    >>> data = make_irm_data(theta=0.5, n_obs=500, dim_x=20, return_type='DataFrame')
    >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
    >>> dml_irm_obj = dml.DoubleMLIRM(obj_dml_data, ml_g, ml_m)
    >>> dml_irm_obj.fit().summary
           coef   std err         t     P>|t|     2.5 %    97.5 %
    d  0.414073  0.238529  1.735941  0.082574 -0.053436  0.881581

    Notes
    -----
    **Interactive regression (IRM)** models take the form

    .. math::

        Y = g_0(D, X) + U, & &\\mathbb{E}(U | X, D) = 0,

        D = m_0(X) + V, & &\\mathbb{E}(V | X) = 0,

    where the treatment variable is binary, :math:`D \\in \\lbrace 0,1 \\rbrace`.
    We consider estimation of the average treatment effects when treatment effects are fully heterogeneous.
    Target parameters of interest in this model are the average treatment effect (ATE),

    .. math::

        \\theta_0 = \\mathbb{E}[g_0(1, X) - g_0(0,X)]

    and the average treatment effect of the treated (ATTE),

    .. math::

        \\theta_0 = \\mathbb{E}[g_0(1, X) - g_0(0,X) | D=1].
    """
    def __init__(self,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 n_folds=5,
                 n_rep=1,
                 score='ATE',
                 dml_procedure='dml2',
                 trimming_rule='truncate',
                 trimming_threshold=1e-12,
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         dml_procedure,
                         draw_sample_splitting,
                         apply_cross_fitting)

        self._check_data(self._dml_data)
        self._check_score(self.score)
        ml_g_is_classifier = self._check_learner(ml_g, 'ml_g', regressor=True, classifier=True)
        _ = self._check_learner(ml_m, 'ml_m', regressor=False, classifier=True)
        self._learner = {'ml_g': ml_g, 'ml_m': ml_m}
        if ml_g_is_classifier:
            if obj_dml_data.binary_outcome:
                self._predict_method = {'ml_g': 'predict_proba', 'ml_m': 'predict_proba'}
            else:
                raise ValueError(f'The ml_g learner {str(ml_g)} was identified as classifier '
                                 'but the outcome variable is not binary with values 0 and 1.')
        else:
            self._predict_method = {'ml_g': 'predict', 'ml_m': 'predict_proba'}
        self._initialize_ml_nuisance_params()

        valid_trimming_rule = ['truncate']
        if trimming_rule not in valid_trimming_rule:
            raise ValueError('Invalid trimming_rule ' + trimming_rule + '. ' +
                             'Valid trimming_rule ' + ' or '.join(valid_trimming_rule) + '.')
        self.trimming_rule = trimming_rule
        self.trimming_threshold = trimming_threshold

    def cate(self, cate_var: str, method: str, alpha: float, n_grid_nodes: int,
             n_samples_bootstrap: int, cv: bool, poly_degree: int = None, splines_knots: int = None,
             splines_degree: int = None, ortho: bool = False, x_grid: np.array = None) -> dict:
        """
        Calculates the CATE with respect to variable X by polynomial or splines approximation.
        It calculates it on an equidistant grid of n_grid_nodes points over the range of X

        Parameters
        ----------
        cate_var: variable whose CATE is calculated
        method: "poly" or "splines", chooses which method to approximate with
        alpha: p-value for the confidence intervals
        n_grid_nodes: number of points of X for which we wish to calculate the CATE
        n_samples_bootstrap: how many samples to use to calculate the t-statistics described in 2.6
        cv: whether to perform cross-validation to find the degree of the polynomials / number of knots
        poly_degree: maximum degree of the polynomial approximation
        splines_knots: max knots for the splines approximation
        splines_degree: degree of the polynomials used in the splines
        ortho: whether to use orthogonal polynomials
        x_grid: grid points on which to evaluate the CATE

        Returns
        -------
        A dictionary containing the estimated CATE (g_hat), with upper and lower confidence bounds (both simultaneous as
        well as pointwise), fitted linear model and grid of X for which the CATE was calculated

        """
        X = np.array(self._dml_data.data[cate_var])
        y = self.psi_b.reshape(-1, 1)
        if x_grid is None:
            print("Authomatically generating grid for CATE evaluation")
            x_grid = np.linspace(np.round(np.min(X), 2), np.round(np.max(X), 2), n_grid_nodes).reshape(-1, 1)
        else:
            print("User-provided grid will be used for CATE evaluation")

        if method == "poly":
            assert poly_degree is not None, "poly_degree must be specified for method 'poly'"

            fitted_model, poly_degree = _polynomial_fit(X=X, y=y, max_degree=poly_degree, cv=cv, ortho=ortho)

            # Build the set of datapoints in X that will be used for prediction
            if ortho:
                regressors_grid = _calculate_orthogonal_polynomials(x_grid, poly_degree)
            else:
                pf = PolynomialFeatures(degree=poly_degree, include_bias=True)
                regressors_grid = pf.fit_transform(x_grid)
            n_knots = np.nan

        elif method == "splines":
            assert splines_knots is not None, "splines_knots must be specified for method 'splines'"
            assert splines_degree is not None, "poly_degree must be specified for method 'splines'"

            fitted_model, n_knots = _splines_fit(X=X, y=y, degree=splines_degree, max_knots=splines_knots, cv=cv)

            # Build the set of datapoints in X that will be used for prediction
            breaks = np.quantile(X, q=np.array(range(0, n_knots + 1)) / n_knots)
            regressors_grid = patsy.bs(x_grid, knots=breaks[1:-1], degree=splines_degree)
            regressors_grid = sm.add_constant(regressors_grid)
            poly_degree = np.nan

        else:
            raise NotImplementedError("The specified method is not implemented. Please use 'poly' or 'splines'")

        g_hat = regressors_grid @ fitted_model.params
        # we can get the HCO matrix directly from the model object
        hcv_coeff = fitted_model.cov_HC0
        standard_error = np.sqrt(np.diag(regressors_grid @ hcv_coeff @ np.transpose(regressors_grid)))
        # Lower pointwise CI
        g_hat_lower_point = g_hat + norm.ppf(q=alpha / 2) * standard_error
        # Upper pointwise CI
        g_hat_upper_point = g_hat + norm.ppf(q=1 - alpha / 2) * standard_error

        max_t_stat = _calculate_bootstrap_tstat(regressors_grid=regressors_grid,
                                                omega_hat=hcv_coeff,
                                                alpha=alpha,
                                                n_samples_bootstrap=n_samples_bootstrap)
        # Lower simultaneous CI
        g_hat_lower = g_hat - max_t_stat * standard_error
        # Upper simultaneous CI
        g_hat_upper = g_hat + max_t_stat * standard_error
        results_dict = {
            "g_hat": g_hat,
            "g_hat_lower": g_hat_lower,
            "g_hat_upper": g_hat_upper,
            "g_hat_lower_point": g_hat_lower_point,
            "g_hat_upper_point": g_hat_upper_point,
            "x_grid": x_grid,
            "fitted_model": fitted_model,
            "n_knots": [n_knots] * len(g_hat),
            "poly_degree": [poly_degree] * len(g_hat)
        }
        return results_dict

    def gate(self, gate_var: str, gate_type: object, n_quantiles: int, alpha: float,
             n_samples_bootstrap: int) -> dict:
        """
        Calculates the GATE for variable X and the confidence intervals.

        Parameters
        ----------
        X: variable for which to calculate the GATE
        y: robust score of the response variable
        gate_type: "quantile" or "categorical", defines whether or not to use the categories already defined in
        n_quantiles: number of quantiles to calculate
        alpha: p-value for the confidence intervals
        n_samples_bootstrap: how many samples to use to calculate the t-statistics described in 2.6

        Returns
        -------
        A dictionary containing the estimated GATE (g_hat), with upper and lower confidence bounds (both simultaneous as
        well as pointwise), fitted linear model and categories for which the GATE was calculated
        """

        X = np.array(self._dml_data.data[gate_var])
        y = self.psi_b.reshape(-1, 1)
        regressors_grid = _create_regressor_grid_gate(X, gate_type, n_quantiles)
        class_names = regressors_grid.columns
        model = sm.OLS(y, regressors_grid)
        fitted_model = model.fit()
        hcv_coeff = fitted_model.cov_HC0
        g_hat = fitted_model.params
        standard_error = np.sqrt(np.diag(hcv_coeff))

        max_t_stat = _calculate_bootstrap_tstat_gate(len(g_hat), alpha, n_samples_bootstrap)

        g_hat_lower_point = g_hat + norm.ppf(q=alpha / 2) * standard_error
        g_hat_upper_point = g_hat + norm.ppf(q=1 - alpha / 2) * standard_error

        g_hat_lower = g_hat - max_t_stat * standard_error
        g_hat_upper = g_hat + max_t_stat * standard_error

        results_dict = {
            "g_hat": g_hat,
            "g_hat_lower": g_hat_lower,
            "g_hat_upper": g_hat_upper,
            "g_hat_lower_point": g_hat_lower_point,
            "g_hat_upper_point": g_hat_upper_point,
            "fitted_model": fitted_model,
            "gate_type": gate_type,
            "n_quantiles": n_quantiles,
            "class_name": class_names
        }
        return results_dict

    def _initialize_ml_nuisance_params(self):
        valid_learner = ['ml_g0', 'ml_g1', 'ml_m']
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in valid_learner}

    def _check_score(self, score):
        if isinstance(score, str):
            valid_score = ['ATE', 'ATTE']
            if score not in valid_score:
                raise ValueError('Invalid score ' + score + '. ' +
                                 'Valid score ' + ' or '.join(valid_score) + '.')
        else:
            if not callable(score):
                raise TypeError('score should be either a string or a callable. '
                                '%r was passed.' % score)
        return

    def _check_data(self, obj_dml_data):
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s). '
                             'To fit an interactive IV regression model use DoubleMLIIVM instead of DoubleMLIRM.')
        one_treat = (obj_dml_data.n_treat == 1)
        binary_treat = (type_of_target(obj_dml_data.d) == 'binary')
        zero_one_treat = np.all((np.power(obj_dml_data.d, 2) - obj_dml_data.d) == 0)
        if not(one_treat & binary_treat & zero_one_treat):
            raise ValueError('Incompatible data. '
                             'To fit an IRM model with DML '
                             'exactly one binary variable with values 0 and 1 '
                             'needs to be specified as treatment variable.')
        return

    def _nuisance_est(self, smpls, n_jobs_cv):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)
        # get train indices for d == 0 and d == 1
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, d)

        # nuisance g
        g_hat0 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_d0, n_jobs=n_jobs_cv,
                                 est_params=self._get_params('ml_g0'), method=self._predict_method['ml_g'])
        _check_finite_predictions(g_hat0, self._learner['ml_g'], 'ml_g', smpls)

        if self._dml_data.binary_outcome:
            binary_preds = (type_of_target(g_hat0) == 'binary')
            zero_one_preds = np.all((np.power(g_hat0, 2) - g_hat0) == 0)
            if binary_preds & zero_one_preds:
                raise ValueError(f'For the binary outcome variable {self._dml_data.y_col}, '
                                 f'predictions obtained with the ml_g learner {str(self._learner["ml_g"])} are also '
                                 'observed to be binary with values 0 and 1. Make sure that for classifiers '
                                 'probabilities and not labels are predicted.')

        g_hat1 = None
        if (self.score == 'ATE') | callable(self.score):
            g_hat1 = _dml_cv_predict(self._learner['ml_g'], x, y, smpls=smpls_d1, n_jobs=n_jobs_cv,
                                     est_params=self._get_params('ml_g1'), method=self._predict_method['ml_g'])
            _check_finite_predictions(g_hat1, self._learner['ml_g'], 'ml_g', smpls)

            if self._dml_data.binary_outcome:
                binary_preds = (type_of_target(g_hat1) == 'binary')
                zero_one_preds = np.all((np.power(g_hat1, 2) - g_hat1) == 0)
                if binary_preds & zero_one_preds:
                    raise ValueError(f'For the binary outcome variable {self._dml_data.y_col}, '
                                     f'predictions obtained with the ml_g learner {str(self._learner["ml_g"])} are also '
                                     'observed to be binary with values 0 and 1. Make sure that for classifiers '
                                     'probabilities and not labels are predicted.')

        # nuisance m
        m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'])
        _check_finite_predictions(m_hat, self._learner['ml_m'], 'ml_m', smpls)

        psi_a, psi_b = self._score_elements(y, d, g_hat0, g_hat1, m_hat, smpls)
        preds = {'ml_g0': g_hat0,
                 'ml_g1': g_hat1,
                 'ml_m': m_hat}

        return psi_a, psi_b, preds

    def _score_elements(self, y, d, g_hat0, g_hat1, m_hat, smpls):
        # fraction of treated for ATTE
        p_hat = None
        if self.score == 'ATTE':
            p_hat = np.full_like(d, np.nan, dtype='float64')
            for _, test_index in smpls:
                p_hat[test_index] = np.mean(d[test_index])

        if (self.trimming_rule == 'truncate') & (self.trimming_threshold > 0):
            m_hat[m_hat < self.trimming_threshold] = self.trimming_threshold
            m_hat[m_hat > 1 - self.trimming_threshold] = 1 - self.trimming_threshold

        # compute residuals
        u_hat0 = y - g_hat0
        u_hat1 = None
        if self.score == 'ATE':
            u_hat1 = y - g_hat1

        if isinstance(self.score, str):
            if self.score == 'ATE':
                psi_b = g_hat1 - g_hat0 \
                    + np.divide(np.multiply(d, u_hat1), m_hat) \
                    - np.divide(np.multiply(1.0-d, u_hat0), 1.0 - m_hat)
                psi_a = np.full_like(m_hat, -1.0)
            else:
                assert self.score == 'ATTE'
                psi_b = np.divide(np.multiply(d, u_hat0), p_hat) \
                    - np.divide(np.multiply(m_hat, np.multiply(1.0-d, u_hat0)),
                                np.multiply(p_hat, (1.0 - m_hat)))
                psi_a = - np.divide(d, p_hat)
        else:
            assert callable(self.score)
            psi_a, psi_b = self.score(y=y, d=d,
                                      g_hat0=g_hat0, g_hat1=g_hat1, m_hat=m_hat,
                                      smpls=smpls)

        return psi_a, psi_b

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)
        # get train indices for d == 0 and d == 1
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, d)

        if scoring_methods is None:
            scoring_methods = {'ml_g': None,
                               'ml_m': None}

        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_d0 = [train_index for (train_index, _) in smpls_d0]
        train_inds_d1 = [train_index for (train_index, _) in smpls_d1]
        g0_tune_res = _dml_tune(y, x, train_inds_d0,
                                self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        g1_tune_res = list()
        if self.score == 'ATE':
            g1_tune_res = _dml_tune(y, x, train_inds_d1,
                                    self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                                    n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        m_tune_res = _dml_tune(d, x, train_inds,
                               self._learner['ml_m'], param_grids['ml_m'], scoring_methods['ml_m'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        g0_best_params = [xx.best_params_ for xx in g0_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        if self.score == 'ATTE':
            params = {'ml_g0': g0_best_params,
                      'ml_m': m_best_params}
            tune_res = {'g0_tune': g0_tune_res,
                        'm_tune': m_tune_res}
        else:
            g1_best_params = [xx.best_params_ for xx in g1_tune_res]
            params = {'ml_g0': g0_best_params,
                      'ml_g1': g1_best_params,
                      'ml_m': m_best_params}
            tune_res = {'g0_tune': g0_tune_res,
                        'g1_tune': g1_tune_res,
                        'm_tune': m_tune_res}

        res = {'params': params,
               'tune_res': tune_res}

        return res


