import numpy as np
import numba as nb
from P2D import EquationParams, SolverMethod, norm_inf, p2d_solver
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse as argp


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


t = 1
p2d10 = EquationParams(x=(0, np.pi),
                       y=(0, np.pi),
                       t=(0, t),
                       a=1, b=0,
                       f=f, phi=phi,
                       mu1=mu13, mu2=mu24,
                       mu3=mu13, mu4=mu24,
                       solution=lambda x, y, t: np.sin(x) * np.sin(y) * np.sin(mu * t))

res, nx, ny, hx, hy = p2d_solver(SolverMethod.AlterDirections, p2d10, 50, 50, 50)

sol = np.zeros((nx, ny), dtype=np.float64)
for i in range(0, nx):
    for j in range(0, ny):
        sol[i][j] = p2d10.solution(i * hx, j * hy, t)

x = np.arange(0, np.pi + hx, hx)
y = np.arange(0, np.pi + hy, hy)
X, Y = np.meshgrid(x, y)
Z1 = res.reshape(X.shape)
Z2 = sol.reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('ox')
ax.plot_wireframe(X, Y, Z1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z2)

plt.show()

print(norm_inf(np.abs(res - sol), nx, ny))
