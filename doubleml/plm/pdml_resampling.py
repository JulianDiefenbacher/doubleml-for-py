import numpy as np

from sklearn.model_selection import KFold, GroupKFold

class DoubleMLPanelResampling:
    def __init__(self,
                 n_folds,
                 n_obs,
                 method,
                 groups=None):
        self.n_folds = n_folds
        self.n_obs = n_obs
        self.method = method
        self.groups = groups

        if n_folds < 2:
            raise ValueError('n_folds must be greater than 1. '
                             'You can use set_sample_splitting with a tuple to only use one fold.')

        if method == 'random':
            self.resampling = KFold(n_splits=n_folds)
        else:
            self.resampling = GroupKFold(n_splits=n_folds)

    def split_samples(self):

        if self.method == 'random' or self.method == 'by_group':
            all_smpls = [(train, test) for train, test in self.resampling.split(X=np.zeros(self.n_obs), y=None, groups=self.groups)]
            smpls = [all_smpls]

        elif self.method == 'time_folds' or self.method == 'time_folds_neighbors':
            smpls_time_var = []
            time_var = self.groups
            time_periods = np.unique(time_var)
            n_time_periods = len(time_periods)

            time_periods_per_fold = len(time_periods) // self.n_folds

            group = np.repeat(np.arange(self.n_folds), time_periods_per_fold)

            smpls_time_var.append([(time_periods[train], time_periods[test])
                                        for train, test in self.resampling.split(np.zeros(n_time_periods), groups=group)])

            if self.method == 'time_folds_neighbors':
                for i in range(self.n_folds):
                    train_up = smpls_time_var[0][i][0]
                    smpls_time_var[0][i] = train_up[~np.isin(train_up, [smpls_time_var[0][i][1] - time_periods_per_fold,
                                                                        smpls_time_var[0][i][1] + time_periods_per_fold])], smpls_time_var[0][i][1]

            all_smpls = []

            for i_smpl in range(self.n_folds):
                ind_train = np.full(self.n_obs, True)
                ind_test = np.full(self.n_obs, True)

                train_timefolds = smpls_time_var[0][i_smpl][0]
                test_timefolds = smpls_time_var[0][i_smpl][1]

                ind_train = ind_train & np.in1d(time_var, train_timefolds)
                ind_test = ind_test & np.in1d(time_var, test_timefolds)

                train_set = np.arange(self.n_obs)[ind_train]
                test_set = np.arange(self.n_obs)[ind_test]
                all_smpls.append((train_set, test_set))

            smpls = [all_smpls]

        else:
            raise ValueError('invalid method')

        return smpls
