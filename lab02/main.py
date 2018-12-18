import argparse as argp
import numpy as np
from matplotlib import pyplot as plt
from H1D import *

# Scheme, initial condition and boudary types
BOUNDARY_TYPE = [BoundaryType.A1P2, BoundaryType.A2P3, BoundaryType.A2P2]
INITIAL_TYPE = [InitialConditionType.P1, InitialConditionType.P2]
SCHEME_TYPE = [SchemeType.EXPLICIT, SchemeType.IMPLICIT]

# Argument names
X = 'x'
T = 't'
SIGMA = 'sigma'
N = 'n'
BOUNDARY = 'boundary'
INITIAL = 'initial'
SCHEME = 'scheme'
DELTA_SIGMA = 'delta_sigma'
PLOT_ERRORS = 'plot_errors'

parser = argp.ArgumentParser(description='Lab 2: H1D solver')
parser.add_argument('-x', '--x', nargs='+', type=float, default=[0, np.pi],
                    help='coordinate domain')
parser.add_argument('-t', '--t', nargs='+', type=float, default=[0, 1],
                    help='time domain')
parser.add_argument('-s', '--sigma', type=float, default=0.9,
                    help='sigma')
parser.add_argument('-n', '--n', type=int, default=5,
                    help='intervals count')
parser.add_argument('-b', '--boundary', nargs='+', type=str, choices=BOUNDARY_TYPE,
                    default=[BoundaryType.A1P2],
                    help='boundary approximation type')
parser.add_argument('-i', '--initial', nargs='+', type=str, choices=INITIAL_TYPE,
                    default=[InitialConditionType.P1],
                    help='initial condition')
parser.add_argument('-S', '--scheme', nargs='+', type=str, choices=SCHEME_TYPE,
                    default=[SchemeType.EXPLICIT],
                    help='scheme type')
parser.add_argument('-D', '--delta-sigma', type=float,
                    default=0.1,
                    help='sigma step for error plots')
parser.add_argument('-e', '--plot-errors',
                    default=False,
                    action='store_true',
                    help='plot error')
args = vars(parser.parse_args())

DESCRIPTION = "Lab 2: H1D solver"
VARIANT = """
************************ Variant 10 ************************
 2             2
∂  u   3 ∂u   ∂  u   ∂u              -t
---- + ---- = ---- + -- - u - cos x e
   2    ∂t       2   ∂x
∂ t           ∂ x

u(x, 0) = sin x

∂u
--(x, 0) = - sin x
∂t

∂u          -t
--(0, t) = e
∂x

∂u            -t
--(π, t) = - e
∂x

************************* Solution *************************
           -t
u(x, t) = e   sin x
************************************************************
"""


def print_options(args_dict):
    x = args_dict[X]
    n = args_dict[N]
    t = args_dict[T]
    sigma = args_dict[SIGMA]
    boundaries = args_dict[BOUNDARY]
    initials = args_dict[INITIAL]
    schemes = args_dict[SCHEME]
    delta_sigma = args_dict[DELTA_SIGMA]
    plot_errors = args_dict[PLOT_ERRORS]
    print(f"X domain: {x}")
    print(f"Interval count: {n}")
    print(f"T domain: {t}")
    print(f"Sigma: {sigma}")
    print(f"Scheme type: {schemes}")
    print(f"Initial condition: {initials}")
    print(f"Boundary approximation: {boundaries}")

    return x, n, t, sigma, schemes, initials, boundaries, delta_sigma, plot_errors


print(DESCRIPTION)
print(VARIANT)
x_arr, n, t, sigma, schemes, initials, boundaries, delta_sigma, plot_errors = print_options(args)

h1d7 = EquationParams(d=3, a=1, b=1, c=-1,
                      f=lambda x, t: -np.cos(x)*np.exp(-t),
                      phi=np.sin,
                      dir1_phi=lambda x: np.cos(x),
                      dir2_phi=lambda x: -np.sin(x),
                      psi=lambda x:-np.sin(x),
                      alpha=1, beta=0, mu1=lambda t: np.exp(-t),
                      gamma=1, delta=0, mu2=lambda t: -np.exp(-t),
                      solution=lambda x, t: np.exp(-t) * np.sin(x))

h = (x_arr[1] - x_arr[0]) / n
x = np.arange(x_arr[0], x_arr[1] + h, h)
u = []
sigma_v = np.arange(delta_sigma, sigma, delta_sigma)
sigma_k = []
error_plots = []
for scheme_type in schemes:
    for initial_type in initials:
        for boundary_type in boundaries:
            s = f'{scheme_type} {initial_type} {boundary_type}'
            uk, tk = h1d_solver(equation_params=h1d7,
                                x1=x_arr[0], x2=x_arr[1],
                                n=n, sigma=sigma,
                                t1=t[0], t2=t[1],
                                scheme=scheme_type,
                                initial=initial_type,
                                boundary=boundary_type)
            u.append((uk, tk, s))
            if plot_errors:
                u_sol = h1d7.solution(x, tk)
                errs = []
                for sigma_i in sigma_v:
                    uk, tk = h1d_solver(equation_params=h1d7,
                                        x1=x_arr[0], x2=x_arr[1],
                                        n=n, sigma=sigma_i,
                                        t1=t[0], t2=t[1],
                                        scheme=scheme_type,
                                        initial=initial_type,
                                        boundary=boundary_type)
                    err = np.linalg.norm(np.abs(u_sol - uk))
                    errs.append(err)
                error_plots.append((errs, s))

# Calculate u*
u_sol = h1d7.solution(x, tk)

# Init pyplot
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax2.set_visible(plot_errors)
f.set_dpi(200)

# Print u*
ax1.plot(x, u_sol, color='r', linestyle='--', label='solution')

for uk, _, s in u:
    err = np.linalg.norm(np.abs(u_sol - uk))
    print(f'{s} error: {err}')
    ax1.plot(x, uk, label=s)

for errs, s in error_plots:
    ax2.plot(sigma_v, errs, label=s)

ax1.legend()
if plot_errors:
    ax2.legend()
plt.show()
