import numpy as np
import pandas as pd
import scipy
from patsy import dmatrix
from sklearn.model_selection import KFold
from qpsolvers import solve_qp
from cvxopt import matrix, solvers
import statistics
import math
import quadprog
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots

from statsmodels.tools.linalg import transf_constraints


unadjusted_prob = np.random.uniform(0, 1, 1000)
p_new = np.random.uniform(0, 1, 1000)
win_var = np.random.binomial(1, unadjusted_prob, len(unadjusted_prob))
win_id = 1

def ecap(unadjusted_prob, win_var, p_new, win_id, bias_indicator=False, lambda_grid=np.power(10, np.linspace(-6, 0, num=13)),
         gamma_grid=np.linspace(0.001,0.05,num=50), theta_grid=np.linspace(-4, 2, num=61, endpoint=True)):
    ## Win and Lose index's for later
    win_index = np.where(win_var == win_id)
    lose_index = np.where(win_var != win_id)

    ## Store the data
    greater_half = pd.Series(greater_half_indicator_vec(unadjusted_prob))
    probs = pd.concat([pd.Series(unadjusted_prob), pd.Series(greater_half), pd.Series(win_var)], axis=1)
    probs.columns = ['p_tilde', 'greater_half', 'win_var']

    ## Convert all probabilities to between 0 and 1/2
    p_flip = prob_flip_fcn_vec(probs['p_tilde'])
    probs['p_flip'] = p_flip
    probs.sort_values(by=['p_flip'])

    ## Generate basis function / omega matrix from p_tilde
    probs_flip = probs['p_flip']
    quantiles = pd.Series(np.linspace(0, 0.5, num=51))

    ## Generate basis matrix and its corresponding 1st and 2nd derivatives
    basis_0 = _eval_bspline_basis(x=probs.p_flip, knots=quantiles, degree=3, deriv=0, include_intercept=True)
    basis_1 = _eval_bspline_basis(x=probs.p_flip, knots=quantiles, degree=3, deriv=1, include_intercept=True)
    # basis_2 = pd.DataFrame(_eval_bspline_basis(x=probs.p_flip, knots=quantiles, degree=3, der=2))
    basis_sum = basis_0.transpose().dot(basis_0)
    # sum_b_d1 = basis_1.transpose().dot(np.repeat(1, basis_1.shape[0]))

    ## We also want to calculate Omega on a fine grid of points
    fine_grid = np.linspace(0, 0.5, num=501, endpoint=True)
    basis_fine_grid = _eval_bspline_basis(x=fine_grid, knots=quantiles, degree=3, deriv=0, include_intercept=True)
    # basis_fine_grid_1 = pd.DataFrame(_eval_bspline_basis(x=fine_grid, knots=quantiles, degree=3, der=1))
    basis_fine_grid_2 = _eval_bspline_basis(x=fine_grid, knots=quantiles, degree=3, deriv=2, include_intercept=True)
    omega = (1/basis_fine_grid.shape[0]) * basis_fine_grid_2.transpose().dot(basis_fine_grid_2)

    ## Grid for the optimization algorithm
    pt = np.linspace(10**-12, 0.5, num=500, endpoint=True)
    basis_0_grid = _eval_bspline_basis(x=pt, knots=quantiles, degree=3, deriv=0, include_intercept=True)
    basis_1_grid = _eval_bspline_basis(x=pt, knots=quantiles, degree=3, deriv=1, include_intercept=True)
    # basis_sum_grid = basis_0_grid.transpose().dot(basis_0_grid)

    ## Risk function for lambda and grid for gamma ##
    ## CV Set Up to get the min value of lambda from risk function
    rows_rand = pd.Series(range(1, basis_0.shape[0])).sample(frac=1)

    ## Declare the number of groups that we want
    n_group = 10

    ## Return a list with 10 approx equal vectors of rows
    ## Here we are going to pick the best value of lambda through cross validation
    kf = KFold(n_splits=n_group, shuffle=True)
    r_cv_split_vec = [risk_cvsplit_fcn(lambda_grid, train_index, test_index, basis_0, basis_1, np.array(probs_flip),
                                       pt, omega, basis_0_grid, basis_1_grid)
                      for train_index, test_index in kf.split(rows_rand)]

    ## Get the value of lambda that corresponds to the smallest risk
    lambda_opt = lambda_grid[pd.DataFrame(r_cv_split_vec).apply(statistics.mean).idxmin()]

    ## Eta hat from optimal lambda above
    eta_hat_opt = eta_min_fcn(lambda_opt, probs['p_flip'], pt, omega, basis_0, basis_1, basis_sum, basis_0_grid, basis_1_grid)

    ## 2D grid search for gamma and theta (1D if the user specifies there is no bias)
    if bias_indicator == False:
        gamma_storage = [tweed_adj_fcn(eta_hat_opt, g, 0, probs['p_tilde'], p_flip, probs, omega, basis_0, basis_1, basis_sum,
                                  basis_0_grid, basis_1_grid, win_index, lose_index) for g in gamma_grid]
        theta_opt = 0.0
        gamma_opt = gamma_grid[np.argmin(gamma_storage)]
    else:
        g_len = len(gamma_grid)
        t_len = len(theta_grid)
        gamma_theta_matrix = pd.DataFrame(np.zeros([g_len, t_len], dtype=float))
        gamma_theta_matrix.columns = theta_grid
        gamma_theta_matrix.index = gamma_grid

        for ii in range(0, g_len):
            g = gamma_grid[ii]
            for jj in range(0, t_len):
                t = theta_grid[jj]
                score = tweed_adj_fcn(eta_hat_opt, g, t, probs['p_tilde'], p_flip, probs, omega, basis_0, basis_1,
                                      basis_sum, basis_0_grid, basis_1_grid, win_index, lose_index)
                (gamma_theta_matrix.iloc[ii]).iloc[jj] = score
                gamma_opt = gamma_theta_matrix.idxmin()[0]
                theta_opt = (gamma_theta_matrix.idxmin()).idxmin()

    ## Use these parameters to generate ECAP estimates on a test set of probability estimates
    new_flip = prob_flip_fcn_vec(p_new)

    ## Combine new probs with the old ones
    p_old_new = np.concatenate((unadjusted_prob, p_new), axis=0)
    p_old_new_flip = np.concatenate((probs['p_flip'], new_flip), axis=0)
    probs_new_flip = np.sort(p_old_new_flip)

    # Generate the basis matrix and its correspoding 1st and 2nd deriv's
    basis_0_new = _eval_bspline_basis(x=probs_new_flip, knots=quantiles, degree=3, deriv=0, include_intercept=True)
    basis_1_new = _eval_bspline_basis(x=probs_new_flip, knots=quantiles, degree=3, deriv=1, include_intercept=True)
    # basis_2_new = pd.DataFrame(_eval_bspline_basis(x=probs_flip, knots=quantiles, degree=3, der=2))
    basis_sum_new = basis_0_new.transpose().dot(basis_0_new)
    # sum_b_d1 = basis_1_new.transpose().dot(np.repeat(1, basis_1_new.shape[0]))

    ecap_old_new = tweedie_est(lambda_opt, gamma_opt, theta_opt, p_old_new, p_old_new_flip, pt,
                          omega, basis_0_new, basis_1_new, basis_sum_new, basis_0_grid, basis_1_grid)

    ecap_new = ecap_old_new[-len(p_new):]

    ## Return valid probabilities
    ecap_new[np.where(ecap_new < 0)] = 0
    ecap_new[np.where(ecap_new > 1)] = 1

    return ecap_new












