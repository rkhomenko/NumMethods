import numpy as np
from E2D import EquationParams, SolverMethod, norm_inf, e2d_solver
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_data(x1, x2, y1, y2, hx, hy, sol):
    x = np.arange(x1, x2 + hx, hx)
    y = np.arange(y1, y2 + hy, hy)
    X, Y = np.meshgrid(x, y)
    Z = sol.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z)
    plt.show()


e2d10 = EquationParams(x=(0, np.pi / 2),
                       y=(0, np.pi / 2),
                       a=1, b=-2, c=-2, d=-4,
                       mu1=lambda y: np.exp(-y) * np.cos(y),
                       mu2=lambda y: 0,
                       mu3=lambda x: np.exp(-x) * np.cos(x),
                       mu4=lambda x: 0,
                       solution=lambda x, y: np.exp(-x - y) * np.cos(x) * np.cos(y))

e2d7 = EquationParams(x=(0, np.pi / 2),
                      y=(0, np.pi / 2),
                      a=1, b=0, c=0, d=-2,
                      mu1=lambda y: np.cos(y),
                      mu2=lambda y: 0,
                      mu3=lambda x: np.cos(x),
                      mu4=lambda x: 0,
                      solution=lambda x, y: np.cos(x) * np.cos(y))


e2d1 = EquationParams(x=(0, 1),
                      y=(0, 1),
                      a=1, b=0, c=0, d=0,
                      mu1=lambda y: y,
                      mu2=lambda y: 1 + y,
                      mu3=lambda x: x,
                      mu4=lambda x: 1 + x,
                      solution=lambda x, y: x + y)

e2d0 = EquationParams(x=(0, 1),
                      y=(0, 1),
                      a=1, b=0, c=0, d=0,
                      mu1=lambda y: 1000,
                      mu2=lambda y: 250,
                      mu3=lambda x: 500,
                      mu4=lambda x: 750,
                      solution=lambda x, y: 0)


def calc(e2d, n, eps):
    x1, x2 = e2d.x
    y1, y2 = e2d.y
    res1, nx, ny, hx, hy = e2d_solver(SolverMethod.SOR, e2d, n, n, eps)
    sol = np.zeros((nx, ny), dtype=np.float64)
    for i in range(0, nx):
        for j in range(0, ny):
            sol[i][j] = e2d.solution(x1 + i * hx, y1 + j * hy)

    print(n, norm_inf(np.abs(sol - res1), nx, ny))
    # plot_data(x1, x2, y1, y2, hx, hy, sol)
    # plot_data(x1, x2, y1, y2, hx, hy, res1)
    # plot_data(x1, x2, y1, y2, hx, hy, res2)


for n in range(5, 100):
    calc(e2d10, n, 1e-6)
