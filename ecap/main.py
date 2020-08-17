import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import statistics
import math
import quadprog

def ecap(unadjusted_prob, win_var, win_id, bias_indicator=False, lambda_grid=np.power(10, np.linspace(-6, 0, num=13)),
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
    probs = probs.sort_values(by=['p_flip'])

    ## Generate basis function / omega matrix from p_tilde
    probs_flip = probs['p_flip']
    quantiles = np.linspace(0, 0.5, num=51)

    ## Generate basis matrix and its corresponding 1st and 2nd derivatives
    basis_0 = _eval_bspline_basis(x=probs.p_flip, knots=quantiles, degree=3, deriv=0, include_intercept=True)
    basis_1 = _eval_bspline_basis(x=probs.p_flip, knots=quantiles, degree=3, deriv=1, include_intercept=True)
    basis_sum = basis_0.transpose().dot(basis_0)

    ## We also want to calculate Omega on a fine grid of points
    fine_grid = np.linspace(0, 0.5, num=501, endpoint=True)
    basis_fine_grid = _eval_bspline_basis(x=fine_grid, knots=quantiles, degree=3, deriv=0, include_intercept=True)
    basis_fine_grid_2 = _eval_bspline_basis(x=fine_grid, knots=quantiles, degree=3, deriv=2, include_intercept=True)
    omega = (1/basis_fine_grid.shape[0]) * basis_fine_grid_2.transpose().dot(basis_fine_grid_2)

    ## Grid for the optimization algorithm
    pt = np.linspace(10**-12, 0.5, num=501, endpoint=True)
    basis_0_grid = _eval_bspline_basis(x=pt, knots=quantiles, degree=3, deriv=0, include_intercept=True)
    basis_1_grid = _eval_bspline_basis(x=pt, knots=quantiles, degree=3, deriv=1, include_intercept=True)

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
        gamma_theta_matrix = np.zeros([g_len, t_len], dtype=float)

        for ii in range(0, g_len):
            g = gamma_grid[ii]
            for jj in range(0, t_len):
                t = theta_grid[jj]
                score = tweed_adj_fcn(eta_hat_opt, g, t, probs['p_tilde'], p_flip, probs, omega, basis_0, basis_1,
                                      basis_sum, basis_0_grid, basis_1_grid, win_index, lose_index)
                (gamma_theta_matrix[ii])[jj] = score
        min_index = np.where(gamma_theta_matrix == np.min(gamma_theta_matrix))
        gamma_opt = gamma_grid[min_index[0]]
        theta_opt = theta_grid[min_index[1]]

    ###  Save all of the estimation info in a dictionary ####
    eta_hat = eta_min_fcn(lambda_opt, probs['p_flip'], pt, omega, basis_0, basis_1,
                          basis_sum, basis_0_grid, basis_1_grid)
    g_hat = np.dot(basis_0, eta_hat)
    g_hat_d1 = np.dot(basis_1, eta_hat)

    ## ecap adjust the training probabilities
    ecap_adj_training_p = tweedie_est(lambda_opt, gamma_opt, theta_opt, probs['p_tilde'], probs['p_flip'], pt,
                                      omega, basis_0, basis_1, basis_sum, basis_0_grid, basis_1_grid)


    return_dict = {
        "lambda_opt": lambda_opt,
        "gamma_opt": gamma_opt,
        "theta_opt": theta_opt,
        "g_hat": g_hat,
        "g_hat_d1": g_hat_d1,
        "unadjusted_prob": unadjusted_prob,
        "unadjusted_flip": p_flip,
        "ecap_training_probabilities": ecap_adj_training_p,
        "lambda_grid": lambda_grid,
        "gamma_grid": gamma_grid,
        "theta_grid": theta_grid
    }

    return return_dict

## Use the fit ecap model to adjust a new set of probability estimates.
def predict_ecap(object, new_unadjusted):
    ## Objects needed for quadratic program and omega matrix (same as ecap fcn)
    quantiles = np.linspace(0, 0.5, num=51)

    fine_grid = np.linspace(0, 0.5, num=501, endpoint=True)
    basis_fine_grid = _eval_bspline_basis(x=fine_grid, knots=quantiles, degree=3, deriv=0, include_intercept=True)
    basis_fine_grid_2 = _eval_bspline_basis(x=fine_grid, knots=quantiles, degree=3, deriv=2, include_intercept=True)
    omega = (1/basis_fine_grid.shape[0]) * basis_fine_grid_2.transpose().dot(basis_fine_grid_2)

    ## Grid for the optimization algorithm
    pt = np.linspace(10**-12, 0.5, num=500, endpoint=True)
    basis_0_grid = _eval_bspline_basis(x=pt, knots=quantiles, degree=3, deriv=0, include_intercept=True)
    basis_1_grid = _eval_bspline_basis(x=pt, knots=quantiles, degree=3, deriv=1, include_intercept=True)

    ## Use these parameters to generate ECAP estimates on a test set of probability estimates
    new_flip = prob_flip_fcn_vec(new_unadjusted)

    ## Combine new probs with the old ones
    p_old_new = np.concatenate((object['unadjusted_prob'], p_new), axis=0)
    p_old_new_flip = np.concatenate((object['unadjusted_flip'], new_flip), axis=0)
    probs_new_flip = np.sort(p_old_new_flip)

    # Generate the basis matrix and its correspoding 1st and 2nd deriv's
    basis_0_new = _eval_bspline_basis(x=probs_new_flip, knots=quantiles, degree=3, deriv=0, include_intercept=True)
    basis_1_new = _eval_bspline_basis(x=probs_new_flip, knots=quantiles, degree=3, deriv=1, include_intercept=True)
    # basis_2_new = pd.DataFrame(_eval_bspline_basis(x=probs_flip, knots=quantiles, degree=3, der=2))
    basis_sum_new = basis_0_new.transpose().dot(basis_0_new)
    # sum_b_d1 = basis_1_new.transpose().dot(np.repeat(1, basis_1_new.shape[0]))

    ecap_old_new = tweedie_est(object['lambda_opt'], object['gamma_opt'], object['theta_opt'], p_old_new,
                               p_old_new_flip, pt, omega, basis_0_new, basis_1_new, basis_sum_new,
                               basis_0_grid, basis_1_grid)

    ecap_new = ecap_old_new[-len(new_unadjusted):]

    ## Return valid probabilities
    ecap_new[np.where(ecap_new < 0)] = 0
    ecap_new[np.where(ecap_new > 1)] = 1

    return ecap_new





## Indicator function if a probability is greater than 1/2
def greater_half_indicator(p_tilde_cur):
    if p_tilde_cur > 0.5:
        return True
    else:
        return False

greater_half_indicator_vec = np.vectorize(greater_half_indicator)

## Flip all probs greater than 1/2
def prob_flip_fcn(p_tilde_cur):
    if p_tilde_cur > 0.5:
        return (1-p_tilde_cur)
    else:
        return p_tilde_cur

prob_flip_fcn_vec = np.vectorize(prob_flip_fcn)

def dvec_terms_fcn(p_flip_term, basis_0_part, basis_1_part):
    return ((1 - 2 * p_flip_term) * basis_0_part) + ((p_flip_term * (1 - p_flip_term)) * basis_1_part)

## Quadratic Program eta min function
def eta_min_fcn(lambda_param, p_flip, pt, omega, basis_0, basis_1, basis_sum, basis_0_grid, basis_1_grid):
    n = basis_1.shape[0]
    end_row = np.where(pt == 0.5)

    ## Set up into the correct form
    Dmat = (2*((1/n) * basis_sum + lambda_param*omega))
    dvec_terms = pd.DataFrame([dvec_terms_fcn(p_flip[ii], basis_0[ii], basis_1[ii])
                               for ii in range(0, len(p_flip)-1)])
    dvec = ((2/n)*dvec_terms.sum(axis=0)).to_numpy()

    ## Constraint vectors
    Amat = basis_0_grid[end_row]
    b_vec = np.array([0.0])

    return_obj = qpsolvers.solve_qp(P=Dmat, q=dvec, A=Amat, b=b_vec)
    return return_obj


def risk_hat_fcn(eta, b_g_test, b_g_d1_test, n, p_test):
    g_hat = np.dot(b_g_test, eta)
    g_hat_d1 = np.dot(b_g_d1_test, eta)
    return_object = (1/n)*(g_hat**2).sum() + (2/n)*(g_hat*(1-2*p_test)+p_test*(1-p_test)*g_hat_d1).sum()
    return return_object

def risk_cvsplit_fcn(lambda_grid, train_index, test_index, basis_0, basis_1, probs_flip, pt, omega, basis_0_grid, basis_1_grid):
    b_g_cv = basis_0[train_index]
    b_g_d1_cv = basis_1[train_index]
    p_train = probs_flip[train_index]

    ## I can do the below without thinking about lambda because it doesn't depend on it
    ## Calculate every term in the sum
    basis_sum_train = b_g_cv.transpose().dot(b_g_cv)

    ## Calculate the column wise sum of the first derivative of our basis matrix

    ## Do this inversion for every value of lambda
    eta_g = [eta_min_fcn(l, np.array(p_train), pt, omega, b_g_cv, b_g_d1_cv, basis_sum_train, basis_0_grid, basis_1_grid)
             for l in lambda_grid]

    ## Now, lets consider the test data
    b_g_test = basis_0[test_index]
    b_g_d1_test = basis_1[test_index]
    p_test = probs_flip[test_index]
    n = b_g_test.shape[0]

    risk_hat = [risk_hat_fcn(eta_cur, b_g_test, b_g_d1_test, n, p_test) for eta_cur in eta_g]
    return risk_hat

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def min_half_fcn(p):
    if p > 0.5:
        return 0.5
    else:
        return p
min_half_fcn_vec = np.vectorize(min_half_fcn)

def mle_binomial(p_hat, win_index, lose_index):
    log_term = sum([math.log(n) for n in p_hat[win_index]])
    minus_log_term = sum([math.log(n) for n in (1-p_hat[lose_index])])
    mle_sum = log_term + minus_log_term
    return mle_sum

def tweed_adj_fcn(eta_hat, gamma_param, theta_param, p_tilde, p_flip, probs, omega, basis_0, basis_1, basis_sum,
                  basis_0_grid, basis_1_grid, win_index, lose_index):
    g_hat = np.dot(basis_0, eta_hat)
    g_hat_d1 = np.dot(basis_1, eta_hat)

    mu_hat = p_flip + gamma_param*(g_hat + 1 - 2*p_flip)
    sigma2_hat = gamma_param*p_flip*(1-p_flip) + (gamma_param**2)*p_flip*(1-p_flip)*(g_hat_d1-2)

    exp_p_hat = mu_hat + 0.5*theta_param*(-1*mu_hat-6*mu_hat*sigma2_hat-2*mu_hat**3+3*sigma2_hat+3*mu_hat**2)
    var_p_hat = ((1-0.5*theta_param)**2)*sigma2_hat + theta_param*sigma2_hat*(9*(mu_hat**4)*theta_param-18*(mu_hat**3)*theta_param + 9*(mu_hat**2)*theta_param-(1-theta_param*0.5)*(3*(mu_hat**2)-3*mu_hat))

    p_hat = min_half_fcn_vec(exp_p_hat + var_p_hat/exp_p_hat)
    if (p_hat < 0).sum() > 0:
        p_hat = p_flip

    ## Flip back
    greater = np.where(p_tilde > 0.5)
    p_hat[greater] = 1-p_hat[greater]

    ## Error from binomial likelihood
    Q_gamma = mle_binomial(p_hat, win_index, lose_index)

    return Q_gamma

def tweedie_est(lambda_param, gamma_param, theta_param, p_old_new, p_old_new_flip, pt,
                omega, basis_0, basis_1, basis_sum, basis_0_grid, basis_1_grid):
    eta_hat = eta_min_fcn(lambda_param, p_old_new_flip, pt, omega, basis_0, basis_1, basis_sum, basis_0_grid, basis_1_grid)
    g_hat = np.dot(basis_0, eta_hat)
    g_hat_d1 = np.dot(basis_1, eta_hat)

    mu_hat = p_old_new_flip + gamma_param*(g_hat + 1 - 2*p_old_new_flip)
    sigma2_hat = gamma_param*p_old_new_flip*(1-p_old_new_flip) + (gamma_param**2)*p_old_new_flip*(1-p_old_new_flip)*(g_hat_d1-2)

    exp_p_hat = mu_hat + 0.5*theta_param*(-mu_hat-6*mu_hat*sigma2_hat-2*mu_hat**3+3*sigma2_hat+3*mu_hat**2)
    var_p_hat = ((1-0.5*theta_param)**2)*sigma2_hat + theta_param*sigma2_hat*(9*(mu_hat**4)*theta_param-18*(mu_hat**3)*theta_param + 9*(mu_hat**2)*theta_param-(1-theta_param*(1/2))*(3*(mu_hat**2)-3*mu_hat))

    p_hat = min_half_fcn_vec(exp_p_hat + var_p_hat / exp_p_hat)
    if (p_hat < 0).sum() > 0:
        p_hat = p_old_new_flip

    ## Flip back
    greater = np.where(p_old_new > 0.5)
    p_hat[greater] = 1 - p_hat[greater]

    return p_hat









