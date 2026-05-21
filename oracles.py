import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        m = len(self.b)
        Ax = self.matvec_Ax(x)
        margins = -self.b * Ax
        loss = np.sum(np.logaddexp(0, margins)) / m
        reg = self.regcoef / 2 * np.dot(x, x)
        return loss + reg
    
    def grad(self, x):
        Ax = self.matvec_Ax(x)
        z = -self.b * Ax
        m = len(self.b)
        probs = expit(z)
        vec = -self.b * probs / m
        return self.matvec_ATx(vec) + self.regcoef * x

    def hess(self, x):
        Ax = self.matvec_Ax(x)
        z = -self.b * Ax
        m = len(self.b)
        probs = expit(z)
        s = probs * (1.0 - probs) / m
        H = self.matmat_ATsA(s)

        if scipy.sparse.issparse(H):
            I_reg = scipy.sparse.diags([self.regcoef], [0], shape=(x.size, x.size))
            return (H + I_reg).toarray()
        else:
            return H + self.regcoef * np.eye(x.size)
        

class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        return self.func(x + alpha * d)

    def grad_directional(self, x, d, alpha):
        return float(self.grad(x + alpha * d) @ d)


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)

    def matmat_ATsA(s):
        if scipy.sparse.issparse(A):
            return A.T.dot(scipy.sparse.diags(s).dot(A))
        else:
            return A.T.dot(s[:, None] * A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)

    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)



def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = x.size
    grad = np.zeros(n)
    fx = func(x)
    for i in range(n):
        x_plus = np.copy(x)
        x_plus[i] += eps
        grad[i] = (func(x_plus) - fx) / eps
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = x.size
    hess = np.zeros((n, n))
    fx = func(x)

    fx_plus = np.zeros(n)
    for i in range(n):
        x_i = np.copy(x)
        x_i[i] += eps
        fx_plus[i] = func(x_i)

    for i in range(n):
        for j in range(i, n):
            x_ij = np.copy(x)
            x_ij[i] += eps
            x_ij[j] += eps
            val = (func(x_ij) - fx_plus[i] - fx_plus[j] + fx) / eps ** 2
            hess[i][j] = val
            hess[j][i] = val

    return hess    

