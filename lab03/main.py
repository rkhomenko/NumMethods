import numpy as np
from E2D import EquationParams, SolverMethod, norm_inf, e2d_solver
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse as argp


SOLVER_METHODS = [
    SolverMethod.Leibmann,
    SolverMethod.Seidel,
    SolverMethod.SOR
]


# Argument names
X = 'x'
Y = 'y'
NX = 'nx'
NY = 'ny'
METHOD = 'method'
EPS = 'eps'
OMEGA = 'omega'
PLOT_ERRORS = 'plot_errors'
PLOT='plot'
SHOW_3D = 'show_3d'


parser = argp.ArgumentParser(description='Lab 3: E2D solver')
parser.add_argument('-x', '--x', nargs='+', type=float, default=[0, np.pi / 2],
                    help='x coordinate domain')
parser.add_argument('-y', '--y', nargs='+', type=float, default=[0, np.pi / 2],
                    help='y coordinate domain')
parser.add_argument('-X', '--nx', type=int, default=5,
                    help='x intervals count')
parser.add_argument('-Y', '--ny', type=int, default=5,
                    help='y intervals count')
parser.add_argument('-E', '--eps', type=float, default=1e-3,
                    help='epsilon')
parser.add_argument('-O', '--omega', type=float, default=1.9,
                    help='omega for SOR')
parser.add_argument('-m', '--method', nargs='+', type=str, choices=SOLVER_METHODS,
                    default=[SolverMethod.SOR],
                    help='solver method')
parser.add_argument('-p', '--plot',
                    default=False,
                    action='store_true',
                    help='plot')
parser.add_argument('-s', '--show-3d',
                    default=False,
                    action='store_true',
                    help='show 3d plot')
parser.add_argument('-e', '--plot-errors',
                    default=False,
                    action='store_true',
                    help='plot error')
args = vars(parser.parse_args())

DESCRIPTION = "Lab 3: E2D solver"
VARIANT = """
 2      2
∂  u   ∂  u        ∂u       ∂u
---- + ---- = - 2 ---- - 2 ---- - 4u
   2      2        ∂x       ∂y
∂ x    ∂ y

           -y
u(0, y) = e   cos y

 /      \\
 | π    |
u|---, y| = 0
 \ 2    /

           -x
u(x, 0) = e   cos x

 /      \\
 |    π |
u|x, ---| = 0
 \    2 /
 
************************* Solution *************************
           - x - y
u(x, y) = e        cos x cos y
************************************************************

"""


def print_options(args_dict):
    x = args_dict[X]
    y = args_dict[Y]
    nx = args_dict[NX]
    ny = args_dict[NY]
    methods = args_dict[METHOD]
    show_3d = args_dict[SHOW_3D]
    eps = args_dict[EPS]
    omega = args_dict[OMEGA]
    plot = args_dict[PLOT]
    plot_errors_flag = args_dict[PLOT_ERRORS]
    print(f"X domain: {x}")
    print(f"X interval count: {nx}")
    print(f"Y domain: {y}")
    print(f"Y interval count: {ny}")
    print(f"Methods: {methods}")

    return x, y, nx, ny, methods, eps, omega, plot, show_3d, plot_errors_flag


print(DESCRIPTION)
print(VARIANT)
x, y, steps_x, steps_y, methods, eps, omega, plot_flag, show_3d, plot_errors_flag = \
    print_options(args)


def plot_errors(errs_list, methods):
    plt.figure()
    plt.title('Errors')
    plt.xlabel('Iteration number')
    plt.ylabel('Error')
    plt.grid()
    for errs, method in zip(errs_list, methods):
        itrs = np.arange(1, len(errs) + 1)
        plt.plot(itrs, errs, label=method)
    plt.legend()


def plot(results, solution, methods, nx, ny):
    y = np.arange(0, ny)
    for i in range(0, nx):
        plt.figure()
        plt.title(f'Plot at x index = {i}')
        plt.xlabel('y index')
        plt.ylabel('u(x, y)')
        plt.grid()
        for res, method in zip(results, methods):
            plt.plot(y, res[i, :], label=method)
        plt.plot(y, solution[i, :], label='Solution')
        plt.legend()


def plot_3d(solution, results, methods, x1, x2, y1, y2, hx, hy):
    x = np.arange(x1, x2 + hx, hx)
    y = np.arange(y1, y2 + hy, hy)
    X, Y = np.meshgrid(x, y)
    Z = solution.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z, label='Solution')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('u(x, y)')
    plt.legend()

    for result, method in zip(results, methods):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, Z, label=method)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('u(x, y)')
        plt.legend()


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


def calc(e2d):
    x1, x2 = e2d.x
    y1, y2 = e2d.y
    results = []
    errs_list = []
    for method in methods:
        result, errs, nx, ny, hx, hy = e2d_solver(method, e2d,
                                                  steps_x, steps_y,
                                                  eps, omega)
        results.append(result)
        errs_list.append(errs)

    solution = np.zeros((nx, ny), dtype=np.float64)
    for i in range(0, nx):
        for j in range(0, ny):
            solution[i][j] = e2d.solution(x1 + i * hx, y1 + j * hy)

    for i, method in zip(range(0, len(methods)), methods):
        err = norm_inf(np.abs(solution - results[i]), nx, ny)
        print(f"Err of '{method}': {err}")

    if plot_errors_flag:
        plot_errors(errs_list, methods)

    if plot_flag:
        plot(results, solution, methods, nx, ny)

    if show_3d:
        plot_3d(solution, results, methods, x1, x2, y1, y2, hx, hy)

    if plot_errors_flag or plot_flag or show_3d:
        plt.show()


calc(e2d10)
