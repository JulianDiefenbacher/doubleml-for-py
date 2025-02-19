import pandas as pd
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess


def panel_sim_CP(N=250, T=10, p=30, x_var=5, seed=None, a=0.25, b=0.5, dgp='DGP3', theta=0.5):

    ids = np.repeat(np.arange(1,N+1),T)
    time = np.tile(np.arange(1,T+1),N)

    np.random.seed(seed)
    c_i = np.repeat(np.random.standard_normal(N), T)
    a_i = np.repeat(np.random.normal(0, 0.95, N), T)

    x_it = np.random.multivariate_normal(np.zeros(p),
                                         np.diag(np.full(p, x_var)), size=[N*T, ])

    u_it = np.random.standard_normal(N*T)
    v_it = np.random.standard_normal(N*T)

    # DGP, a = 0.25, b = 0.5
    if dgp == 'DGP1':
        l_0 = a * x_it[:,0] + x_it[:,2]
        m_0 = a * x_it[:,0] + x_it[:,2]
    elif dgp == 'DGP2':
        l_0 = (np.exp(x_it[:,0]) / (1 + np.exp(x_it[:,0]))) + a * np.cos(x_it[:,2])
        m_0 = np.cos(x_it[:,0]) + a * (np.exp(x_it[:,2]) / (1 + np.exp(x_it[:,2])))
    elif dgp == 'DGP3':
        l_0 = b * (x_it[:,0] * x_it[:,2]) + a * (x_it[:,2] * np.where(x_it[:,2] > 0, 1, 0))
        m_0 = a * (x_it[:,0] * np.where(x_it[:,0] > 0, 1, 0)) + b * (x_it[:,0] * x_it[:,2])
    else:
        raise ValueError('Invalid dgp')

    # treatment
    theta = theta

    d_it = m_0 + c_i + v_it

    def alpha_d(d_it, N):
        d_i = np.array_split(d_it, N)
        d_i_mean = np.repeat(np.mean(d_i, axis=1), T) - np.mean(d_it)
        return d_i_mean

    def alpha_x(x_it, N):
        x_i_split = np.array_split(x_it[:, [0, 2]], N)
        x_i_mean = np.sum(np.mean(x_i_split, axis=1), axis=1)
        x_i_term = np.repeat(x_i_mean, T)
        return x_i_term


    alpha_i = 0.25 * alpha_d(d_it, N) + 0.25 * alpha_x(x_it, N) + a_i

    y_it = d_it * theta + l_0 + alpha_i + u_it


    x_cols = [f'x{i + 1}' for i in np.arange(p)]

    data = pd.DataFrame(np.column_stack((ids, time, d_it, y_it, x_it)),
                                columns=['id', 'time', 'd', 'y'] + x_cols).astype({'id': 'int64', 'time': 'int64'})

    return data


def panel_sim_FP(N=100, T=10, J=1, form='linear', autocorr=False, rho=0.9, beta=1, dgp='dgp3', two_way = False, seed=None):

    np.random.seed(seed)

    # covariate functional form
    def g_0_m_0(X_it, form='linear'):
        if form == 'linear':
            g_0 = X_it
            m_0 = X_it
        elif form == 'u-shaped':
            g_0 = X_it ** 2
            m_0 = X_it ** 2
        else:
            raise ValueError('Invalid functional form')
        return g_0, m_0

    # generate AR(1) terms
    def ar1_err(n_namples, rho):
        sim_ar1 = ArmaProcess(ar=np.array([1, -rho]),
                            ma=None).generate_sample(nsample=n_namples,
                                                    scale=1, distrvs=np.random.standard_normal)
        return sim_ar1

    # error terms for x, w, y
    if J > 1:
        A = np.random.standard_normal(J ** 2).reshape((J,J))
        sigma = A.transpose() @ A
        epsilon_it = np.random.multivariate_normal(np.zeros(J), sigma, size=[N*T, ])
    else:
        epsilon_it = np.random.standard_normal(T*N).reshape((-1,1))

    eta_it = np.random.standard_normal(T*N)

    if autocorr:
        auto_list = [ar1_err(T, rho) for _ in range(N)]
        mu_it = np.concatenate(auto_list, axis=0)
        print('auto')
    else:
        mu_it = np.random.standard_normal(T*N) # autocorr with rho = 0 identical

    # coefficients for covariates and U_i
    gamma_j = np.random.standard_normal(J) / J

    if dgp == 'dgp1':
        delta_x = delta_wy = 0
    elif dgp == 'dgp2':
        delta_x = 0
        delta_wy = np.random.standard_normal(1)
    elif dgp == 'dgp3':
        delta_x = delta_wy = np.random.standard_normal(1)
    else:
        raise ValueError('Invalid dgp')

    beta = beta # causal effect

    # unobserved heterogeneity
    U_i = np.repeat(np.random.standard_normal(N), T)

    if two_way:
        U_t = np.tile(np.random.standard_normal(T), N)
    else:
        U_t = 0

    # intercept(s) for covariate(s)
    alpha0 = np.random.standard_normal(J)

    # covariates
    X_it = alpha0 + delta_x * np.column_stack((U_i,)*J) + delta_x * np.column_stack((U_t,)*J) + epsilon_it

    g_0, m_0 = g_0_m_0(X_it, form=form)

    # treatment variable
    W_it = np.random.standard_normal(1) + g_0 @ gamma_j + delta_wy * U_i + delta_wy * U_t + eta_it

    # outcome variable
    Y_it = np.random.standard_normal(1) + beta * W_it + m_0 @ gamma_j + delta_wy * U_i + delta_wy * U_t + mu_it

    # create data frame
    ids = np.repeat(np.arange(1,N+1),T)
    time = np.tile(np.arange(1,T+1),N)
    x_cols = [f'x{i + 1}' for i in np.arange(J)]

    data = pd.DataFrame(np.column_stack((ids, time, Y_it, W_it, X_it)),
                        columns=['id', 'time', 'y', 'w'] + x_cols).astype({'id': 'int64', 'time': 'int64'})
    return data
