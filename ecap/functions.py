import numpy as np
import pandas as pd
import quadprog
import math

## Indicator function if a probability is greater than 1/2
def greater_half_indicator(p_tilde_cur):
    if p_tilde_cur > 0.5:
        return True
    else:
        return False

## Flip all probs greater than 1/2
def prob_flip_fcn(p_tilde_cur):
    if p_tilde_cur > 0.5:
        return (1-p_tilde_cur)
    else:
        return p_tilde_cur

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
    dvec = -1*((2/n)*dvec_terms.sum(axis=0)).to_numpy()

    ## Constraint vectors
    Amat = basis_0_grid[end_row]
    b_vec = np.array([0.0])

    return_obj = quadprog.solve_qp(Dmat, dvec, Amat.transpose(), b_vec, meq=1)[0]
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

def min_half_fcn(p):
    if p > 0.5:
        return 0.5
    else:
        return p

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

    p_hat = np.array([min_half_fcn(p_adj) for p_adj in (exp_p_hat + var_p_hat/exp_p_hat)])
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

    p_hat = np.array([min_half_fcn(p_adj) for p_adj in (exp_p_hat + var_p_hat/exp_p_hat)])
    if (p_hat < 0).sum() > 0:
        p_hat = p_old_new_flip

    ## Flip back
    greater = np.where(p_old_new > 0.5)
    p_hat[greater] = 1 - p_hat[greater]

    return p_hat


