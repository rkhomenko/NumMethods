import numpy as np
from E2D import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_data(x1, x2, y1, y2, hx, hy, sol):
    x = np.arange(x1, x2 + hx, hx)
    y = np.arange(y1, y2 + hy, hy)
    X, Y = np.meshgrid(x, y)
    Z = sol.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    plt.show()

e2d = EquationParams(a=1, b=-2, c=-2, d=-4, e=0,
                     f=lambda x, y: 0,
                     mu1=lambda y: np.exp(-y) * np.cos(y),
                     mu2=lambda y: 0,
                     mu3=lambda x: np.exp(-x) * np.cos(x),
                     mu4=lambda x: 0,
                     solution=lambda x, y: np.exp(-x - y) * np.cos(x) * np.cos(y))

res, nx, ny, hx, hy = e2d_solver_leibmann(e2d, 0, np.pi / 2, 0, np.pi / 2, 200, 200, 1e-5)
print(res)
print("***")

sol = np.zeros((nx, ny), dtype=np.float64)
for i in range(0, nx):
    for j in range(0, ny):
        sol[i][j] = e2d.solution(i * hx, j * hy)
print(sol)
print(np.linalg.norm(np.abs(sol - res), ord=np.inf))
plot_data(0, np.pi / 2, 0, np.pi / 2, hx, hy, sol)
plot_data(0, np.pi / 2, 0, np.pi / 2, hx, hy, res)
