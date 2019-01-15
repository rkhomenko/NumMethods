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
    plot = args_dict[PLOT]
    plot_errors = args_dict[PLOT_ERRORS]
    print(f"X domain: {x}")
    print(f"X interval count: {nx}")
    print(f"Y domain: {y}")
    print(f"Y interval count: {ny}")
    print(f"Methods: {methods}")

    return x, y, nx, ny, methods, plot, show_3d, plot_errors


print(DESCRIPTION)
print(VARIANT)
x, y, steps_x, steps_y, methods, plot, show_3d, plot_errors = print_options(args)

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
    res1, nx, ny, hx, hy = e2d_solver(SolverMethod.SOR, e2d, n, n, eps, 1.9)
    sol = np.zeros((nx, ny), dtype=np.float64)
    for i in range(0, nx):
        for j in range(0, ny):
            sol[i][j] = e2d.solution(x1 + i * hx, y1 + j * hy)

    print(n, norm_inf(np.abs(sol - res1), nx, ny))
    # plot_data(x1, x2, y1, y2, hx, hy, sol)
    # plot_data(x1, x2, y1, y2, hx, hy, res1)
    # plot_data(x1, x2, y1, y2, hx, hy, res2)


# for n in range(5, 100):
#     calc(e2d10, n, 1e-6)
