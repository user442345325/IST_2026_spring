import numpy as np
from numpy.linalg import LinAlgError
import scipy
import scipy.sparse
import scipy.sparse.linalg
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.
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
        and for a given search direction d_k.
        """
        phi = lambda alpha: oracle.func_directional(x_k, d_k, alpha)
        derphi = lambda alpha: oracle.grad_directional(x_k, d_k, alpha)

        if self._method == 'Constant':
            return self.c

        elif self._method == 'Armijo':
            alpha = previous_alpha if previous_alpha is not None else self.alpha_0
            phi_0 = phi(0.0)
            derphi_0 = derphi(0.0)

            while phi(alpha) > phi_0 + self.c1 * alpha * derphi_0:
                alpha /= 2.0
            return alpha

        elif self._method == 'Wolfe':
            # Custom 1D directional line search to strictly adhere to oracle call budget
            alpha = self._backtracking_wolfe(phi, derphi, previous_alpha)
            if alpha is None:
                # Armijo backtracking fallback on Wolfe failure
                alpha = previous_alpha if previous_alpha is not None else self.alpha_0
                phi_0 = phi(0.0)
                derphi_0 = derphi(0.0)
                while phi(alpha) > phi_0 + self.c1 * alpha * derphi_0:
                    alpha /= 2.0
            return alpha

    def _backtracking_wolfe(self, phi, derphi, previous_alpha):
        """Custom bisection line search enforcing Strong Wolfe conditions."""
        alpha_0 = previous_alpha if previous_alpha is not None else self.alpha_0
        phi_0 = phi(0.0)
        derphi_0 = derphi(0.0)

        alpha = alpha_0
        low = 0.0
        high = np.inf

        for _ in range(50):
            p_val = phi(alpha)
            if p_val > phi_0 + self.c1 * alpha * derphi_0:
                high = alpha
                alpha = (low + high) / 2.0
            else:
                d_val = derphi(alpha)
                if abs(d_val) <= -self.c2 * derphi_0:
                    return alpha
                elif d_val > 0:
                    high = alpha
                    alpha = (low + high) / 2.0
                else:
                    low = alpha
                    if high == np.inf:
                        alpha *= 2.0
                    else:
                        alpha = (low + high) / 2.0
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

        d_k = -grad_k
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=alpha_k)

        if alpha_k is None:
            return x_k, 'computational_error', history

        x_k = x_k + alpha_k * d_k
        grad_k = oracle.grad(x_k)

        if not np.all(np.isfinite(x_k)) or not np.all(np.isfinite(grad_k)):
            return x_k, 'computational_error', history

        time_passed = (datetime.now() - start_time).total_seconds()
        record_history(x_k, grad_k, time_passed)

    if np.linalg.norm(grad_k) ** 2 <= tolerance * grad_0_norm_sq:
        return x_k, 'success', history
    return x_k, 'iterations_exceeded', history


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
