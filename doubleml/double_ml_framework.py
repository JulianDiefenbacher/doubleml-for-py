import numpy as np
import pandas as pd
from scipy.stats import norm

from .double_ml_base import DoubleMLBase
from ._utils_base import _draw_weights
from ._utils_checks import _check_bootstrap


class DoubleMLFramework():
    """Double Machine Learning Framework to combine DoubleMLBase classes and compute confidendence intervals."""

    def __init__(
            self,
            dml_base_obj=None,
            n_thetas=1,
            n_rep=1,
            n_obs=1,
    ):
        # set dimensions
        if dml_base_obj is None:
            # set scores and parameters
            self._n_thetas = n_thetas
            self._n_rep = n_rep
            self._n_obs = n_obs
        else:
            assert isinstance(dml_base_obj, DoubleMLBase)
            # set scores and parameters according to dml_base_obj
            self._n_thetas = 1
            self._n_rep = dml_base_obj.n_rep
            self._n_obs = dml_base_obj.n_obs

        # initalize arrays
        self._thetas = np.full(self._n_thetas, np.nan)
        self._ses = np.full(self._n_thetas, np.nan)
        self._all_thetas = np.full((self._n_rep, self._n_thetas), np.nan)
        self._all_ses = np.full((self._n_rep, self._n_thetas), np.nan)
        self._psi = np.full((self._n_obs, self._n_rep, self._n_thetas), np.nan)
        self._psi_deriv = np.full((self._n_obs, self._n_rep, self._n_thetas), np.nan)

        if dml_base_obj is not None:
            # initalize arrays from double_ml_base_obj
            self._thetas[0] = np.array([dml_base_obj.theta])
            self._ses[0] = np.array([dml_base_obj.se])
            self._all_thetas[:, 0] = dml_base_obj.all_thetas
            self._all_ses[:, 0] = dml_base_obj.all_ses
            self._psi[:, :, 0] = dml_base_obj.psi
            self._psi_deriv[:, :, 0] = dml_base_obj.psi_deriv

    @property
    def dml_base_objs(self):
        """
        Sequence of DoubleMLBase objects.
        """
        return self._dml_base_objs

    @property
    def n_thetas(self):
        """
        Number of target parameters.
        """
        return self._n_thetas

    @property
    def n_rep(self):
        """
        Number of repetitions.
        """
        return self._n_rep

    @property
    def n_obs(self):
        """
        Number of observations.
        """
        return self._n_obs

    @property
    def thetas(self):
        """
        Estimated target parameters.
        """
        return self._thetas

    @property
    def all_thetas(self):
        """
        Estimated target parameters for each repetition.
        """
        return self._all_thetas

    @property
    def ses(self):
        """
        Estimated standard errors.
        """
        return self._ses

    @property
    def all_ses(self):
        """
        Estimated standard errors for each repetition.
        """
        return self._all_ses

    def confint(self, joint=False, level=0.95, aggregated=True):
        """
        Confidence intervals for DoubleML models.

        Parameters
        ----------
        joint : bool
            Indicates whether joint confidence intervals are computed.
            Default is ``False``

        level : float
            The confidence level.
            Default is ``0.95``.

        Returns
        -------
        df_ci : pd.DataFrame
            A data frame with the confidence interval(s).
        """

        if not isinstance(joint, bool):
            raise TypeError('joint must be True or False. '
                            f'Got {str(joint)}.')

        if not isinstance(level, float):
            raise TypeError('The confidence level must be of float type. '
                            f'{str(level)} of type {str(type(level))} was passed.')
        if (level <= 0) | (level >= 1):
            raise ValueError('The confidence level must be in (0,1). '
                             f'{str(level)} was passed.')

        alpha = 1 - level
        ab = np.array([alpha / 2, 1. - alpha / 2])
        if joint:
            # TODO: add bootstraped critical values
            pass
        else:
            if np.isnan(self.thetas).any():
                raise ValueError('Apply estimate_thetas() before confint().')
            critical_value = norm.ppf(ab)

            ci = np.vstack((self.all_thetas + self.all_ses * critical_value[0],
                            self.all_thetas + self.all_ses * critical_value[1]))
        # TODO: add treatment names
        df_ci = pd.DataFrame(
            ci,
            columns=['{:.1f} %'.format(i * 100) for i in ab])
        return df_ci

    def bootstrap(self, method='normal', n_rep_boot=500):
        """
        Multiplier bootstrap for DoubleMLFrameworks.

        Parameters
        ----------
        method : str
            A str (``'Bayes'``, ``'normal'`` or ``'wild'``) specifying the multiplier bootstrap method.
            Default is ``'normal'``

        n_rep_boot : int
            The number of bootstrap replications.

        Returns
        -------
        self : object
        """

        _check_bootstrap(method, n_rep_boot)

        J_vec = np.mean(self._psi_deriv, axis=0)
        score_scaling = self._n_obs * np.multiply(self._all_ses, np.mean(self._psi_deriv, axis=0))

        for i_rep in range(self.n_rep):
            boot = np.matmul(weights, self._psi[:, i_rep, :])

        standardized_scores = np.multiply(score_scaling, self._psi)

        weights = _draw_weights(method, n_rep_boot, self._n_obs)

        return self
