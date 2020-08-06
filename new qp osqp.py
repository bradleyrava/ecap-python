import osqp
import scipy.sparse as sparse

def eta_min_fcn(lambda_param, p_flip, pt, omega, basis_0, basis_1, basis_sum, basis_0_grid, basis_1_grid):
    n = basis_1.shape[0]
    end_row = np.where(pt == 0.5)

    ## Set up into the correct form
    Dmat = sparse.csc_matrix((2*((1/n) * basis_sum + lambda_param*omega)))
    dvec_terms = pd.DataFrame([dvec_terms_fcn(p_flip[ii], basis_0[ii], basis_1[ii])
                               for ii in range(0, len(p_flip)-1)])
    dvec = -1*((2/n)*dvec_terms.sum(axis=0)).to_numpy()

    ## Constraint vectors
    Amat = sparse.csc_matrix(basis_0_grid[end_row])
    b_vec = np.array([0.0])

    ## solve the qp problem with osqp
    prob = osqp.OSQP()
    prob.setup(P=Dmat, q=dvec, A=Amat, l=b_vec, u=b_vec)
    return_object = prob.solve().x

    return return_object