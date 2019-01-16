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


SOLVER_METHODS = [
    SolverMethod.AlterDirections,
    SolverMethod.FractSteps
]


# Argument names
X = 'x'
Y = 'y'
T = 't'
NX = 'nx'
NY = 'ny'
NT = 'nt'
METHOD = 'method'
PLOT_ERRORS = 'plot_errors'
PLOT='plot'
SHOW_3D = 'show_3d'


parser = argp.ArgumentParser(description='Lab 4: P2D solver')
parser.add_argument('-x', '--x', nargs='+', type=float, default=[0, np.pi],
                    help='x coordinate domain')
parser.add_argument('-y', '--y', nargs='+', type=float, default=[0, np.pi],
                    help='y coordinate domain')
parser.add_argument('-t', '--t', nargs='+', type=float, default=[0, 1],
                    help='time domain')
parser.add_argument('-X', '--nx', type=int, default=5,
                    help='x intervals count')
parser.add_argument('-Y', '--ny', type=int, default=5,
                    help='y intervals count')
parser.add_argument('-T', '--nt', type=int, default=5,
                    help='time intervals count')
parser.add_argument('-m', '--method', nargs='+', type=str, choices=SOLVER_METHODS,
                    default=[SolverMethod.AlterDirections],
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


DESCRIPTION = "Lab 4: E2D solver"
VARIANT = "Coming soon"


def print_options(args_dict):
    x = args_dict[X]
    y = args_dict[Y]
    t = args_dict[T]
    nx = args_dict[NX]
    ny = args_dict[NY]
    nt = args_dict[NT]
    methods = args_dict[METHOD]
    show_3d = args_dict[SHOW_3D]
    plot = args_dict[PLOT]
    plot_errors_flag = args_dict[PLOT_ERRORS]
    print(f"X domain: {x}")
    print(f"X interval count: {nx}")
    print(f"Y domain: {y}")
    print(f"Y interval count: {ny}")
    print(f"T domain: {t}")
    print(f"T interval count: {nt}")
    print(f"Methods: {methods}")

    return x, y, t, nx, ny, nt, methods, plot, show_3d, plot_errors_flag


print(DESCRIPTION)
print(VARIANT)
x, y, t, steps_x, steps_y, steps_t, methods, plot_flag, show_3d, plot_errors_flag = \
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
        ax.plot_wireframe(X, Y, result, label=method)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('u(x, y)')
        plt.legend()


p2d10 = EquationParams(x=x,
                       y=y,
                       t=t,
                       a=1, b=0,
                       f=f, phi=phi,
                       mu1=mu13, mu2=mu24,
                       mu3=mu13, mu4=mu24,
                       solution=lambda x, y, t: np.sin(x) * np.sin(y) * np.sin(mu * t))


def calc(e2d):
    x1, x2 = e2d.x
    y1, y2 = e2d.y
    t1, t2 = e2d.t
    results_method = []
    err_list = []

    for method in methods:
        results, nx, ny, nt, hx, hy, ht = p2d_solver(method, e2d,
                                                     steps_x, steps_y, steps_t)
        results_method.append(results)

    solutions = []
    for k in range(1, nt):
        solution = np.zeros((nx, ny), dtype=np.float64)
        for i in range(0, nx):
            for j in range(0, ny):
                solution[i][j] = e2d.solution(x1 + i * hx, y1 + j * hy, t1 + k * nt)
        solutions.append(solution)

    for result, solution in zip(results_method, solutions):
        errs = []
        for k in range(0, nt - 1):
            errs.append(norm_inf(np.abs(result[k] - solution), nx, ny))
        err_list.append(errs)

    for i, method in zip(range(0, len(methods)), methods):
        err = norm_inf(np.abs(solutions[nt - 2] - results_method[i][nt - 2]), nx, ny)
        print(f"Err of '{method}': {err}")

    if plot_errors_flag:
        plot_errors(err_list, methods)

    if plot_flag:
        ress = []
        for i in range(0, len(methods)):
            ress.append(results_method[i][nt - 2])
        plot(ress, solution, methods, nx, ny)

    if show_3d:
        ress = []
        for i in range(0, len(methods)):
            ress.append(results_method[i][nt - 2])
        plot_3d(solution, ress, methods, x1, x2, y1, y2, hx, hy)

    if plot_errors_flag or plot_flag or show_3d:
        plt.show()


calc(p2d10)
