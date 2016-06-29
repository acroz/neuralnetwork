"""
SciPy does not provide a simple gradient descent algorithm - one is implemented
here for comparison.
"""

import numpy as np
from scipy.optimize import OptimizeResult

def gradient_descent(alpha):
    """
    Generate a simple gradient descent optimiser for use with SciPy.

    Parameters
    ----------
    alpha : float
        The training rate to use
    """

    def gradient_descent(fun, x0, args=(), jac=None, gtol=1e-5, callback=None,
                         maxiter=None, **kwargs):
        """
        A simple gradient descent optimisation algorithm.
        """

        x = x0.copy()
        grad = jac(x)
        i = 0
        warnflag = 0
    
        while np.linalg.norm(grad) > gtol:
            
            i += 1
    
            grad = jac(x)
            x = x - alpha * grad
    
            if callback is not None:
               callback(x)
    
            if maxiter is not None and i >= maxiter:
                warnflag = 2
                break
    
        result = OptimizeResult(fun=fun(x), nit=i, nfev=1, njev=i,
                                status=warnflag, success=(warnflag==0), x=x)
    
        return result
    
    return gradient_descent
