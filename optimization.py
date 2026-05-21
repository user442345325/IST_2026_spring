from scipy.optimize._linesearch import scalar_search_wolfe2
import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Constant':
            return self.c

        if self._method == 'Wolfe':
            phi = lambda alpha: oracle.func_directional(x_k, d_k, alpha)
            derphi = lambda alpha: oracle.grad_directional(x_k, d_k, alpha)
            
            alpha, *_ = scalar_search_wolfe2(phi, derphi, c1=self.c1, c2=self.c2)

            if alpha is None:  # Wolfe failed, fallback to Armijo
                alpha = float(self.alpha_0)
                phi_0 = oracle.func_directional(x_k, d_k, 0)
                dphi_0 = oracle.grad_directional(x_k, d_k, 0)
                
                while oracle.func_directional(x_k, d_k, alpha) > phi_0 + self.c1 * alpha * dphi_0:
                    alpha /= 2.0
                    if alpha < 1e-12:
                        return None

            return alpha

        if self._method == 'Armijo':
            alpha = float(previous_alpha) if previous_alpha is not None else float(self.alpha_0)
            phi_0 = oracle.func_directional(x_k, d_k, 0)
            dphi_0 = oracle.grad_directional(x_k, d_k, 0)

            while oracle.func_directional(x_k, d_k, alpha) > phi_0 + self.c1 * alpha * dphi_0:
                alpha /= 2.0
                print(f"Halving alpha: {alpha}")
                if alpha < 1e-12:
                    print("alpha is too low")
                    return None

            print(f"return alpha: {alpha}")
            return alpha

        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    grad_at_0 = oracle.grad(x_0)
    print("grad_at_0", grad_at_0)
    norm_at_0 = norm(grad_at_0)
    print("norm_at_0", norm_at_0)
    func_at_i_copy = x_k
    print("func_at_i_copy", func_at_i_copy)    
    grad_at_i_copy = grad_at_0
    print("grad_at_i_copy", grad_at_i_copy)    

    # TODO: Implement gradient descent
    # Use line_search_tool.line_search() for adaptive step size.

    i = 0
    while i < max_iter:
        func_at_i = oracle.func(x_k)
        print("func_at_i",func_at_i)
        grad_at_i = oracle.grad(x_k)
        print("grad_at_i",grad_at_i)
        norm_at_i = norm(grad_at_i)
        print("norm_at_i",norm_at_i)
        if norm_at_i**2 <= tolerance * norm_at_0:
            break
        d_k = -grad_at_i
        print("d_k", d_k)

        alpha_k = line_search_tool.line_search(oracle, func_at_i_copy, grad_at_i_copy)
        print("alpha_k", alpha_k)
        if alpha_k == None:
            return x_k, 'computational_error', history
        func_at_i_copy = func_at_i
        grad_at_i_copy = grad_at_i
        print(alpha_k, d_k)
        x_k += alpha_k * d_k
        i += 1

    if i == max_iter:
        return x_k, 'iterations_exceeded', history
    # elif x_k =
    # computational_error FOR NOW I DON'T KNOW HOW THIS CAN BE ACHIEVED
    return x_k, 'success', history

def norm(vector):
    """We will implement euclidian norm"""
    if not isinstance(vector, np.ndarray):
        raise ValueError('Massive should be an array.')
    if not vector.ndim == 1:
        raise ValueError('Array should be of dimension 1.')
    return sum([x**2 for x in vector])**(1/2)
        

def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    start_time = datetime.now()
    grad_k = oracle.grad(x_k)
    grad_0_norm_sq = np.linalg.norm(grad_k) ** 2

    def record_history(x, grad, time_passed):
        if not trace: return
        history['time'].append(time_passed)
        history['func'].append(oracle.func(x))
        history['grad_norm'].append(np.linalg.norm(grad))
        if x.size <= 2:
            history['x'].append(np.copy(x))

    record_history(x_k, grad_k, 0.0)
    alpha_k = None

    for k in range(max_iter):
        grad_norm_sq = np.linalg.norm(grad_k) ** 2

        if display:
            print(f"Iteration {k}: f(x)={oracle.func(x_k)}, ||grad||={np.sqrt(grad_norm_sq)}")

        if grad_norm_sq <= tolerance * grad_0_norm_sq:
            return x_k, 'success', history

        if not np.all(np.isfinite(x_k)) or not np.all(np.isfinite(grad_k)):
            return x_k, 'computational_error', history

        hess_k = oracle.hess(x_k)

        try:
            if scipy.sparse.issparse(hess_k):
                d_k = scipy.sparse.linalg.spsolve(hess_k, -grad_k)
            else:
                c, low = scipy.linalg.cho_factor(hess_k)
                d_k = scipy.linalg.cho_solve((c, low), -grad_k)
        except (LinAlgError, ValueError):
            return x_k, 'computational_error', history

        try:
            if scipy.sparse.issparse(hess_k):
                d_k = scipy.sparse.linalg.spsolve(hess_k, -grad_k)
            else:
                c, low = scipy.linalg.cho_factor(hess_k)
                d_k = scipy.linalg.cho_solve((c, low), -grad_k)
        except LinAlgError:
            return x_k, 'newton_direction_error', history
        except ValueError:
            return x_k, 'computational_error', history

        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=alpha_k)

        if alpha_k is None:
            return x_k, 'computational_error', history

        x_k = x_k + alpha_k * d_k
        grad_k = oracle.grad(x_k)

        time_passed = (datetime.now() - start_time).total_seconds()
        record_history(x_k, grad_k, time_passed)

    if np.linalg.norm(grad_k) ** 2 <= tolerance * grad_0_norm_sq:
        return x_k, 'success', history
    return x_k, 'iterations_exceeded', history
