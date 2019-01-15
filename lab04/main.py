import numpy as np
import numba as nb
from P2D import EquationParams, SolverMethod, p2d_solver


a = 1
b = 1
mu = 1


@nb.jit(nopython=True)
def f(x, y, t):
    return np.sin(x) * np.sin(y) * \
           (mu * np.cos(mu * t) + (a + b) * np.sin(mu * t))


@nb.jit(nopython=True)
def phi(x, y):
    return 0


@nb.jit(nopython=True)
def mu13(x, t):
    return 0


@nb.jit(nopython=True)
def mu24(x, t):
    return - np.sin(x) * np.sin(mu * t)


e2d10 = EquationParams(x=(0, np.pi / 2),
                       y=(0, np.pi / 2),
                       t=(0, 1),
                       a=1, b=0,
                       f=f, phi=phi,
                       mu1=mu13, mu2=mu24,
                       mu3=mu13, mu4=mu24,
                       solution=lambda x, y, t: np.sin(x) * np.sin(y) * np.sin(mu * t))


p2d_solver(SolverMethod.AlterDirections, e2d10, 10, 10, 10)
