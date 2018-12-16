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
    print(f"X domain: {x}")
    print(f"Interval count: {n}")
    print(f"T domain: {t}")
    print(f"Sigma: {sigma}")
    print(f"Scheme type: {schemes}")
    print(f"Initial condition: {initials}")
    print(f"Boundary approximation: {boundaries}")

    return x, n, t, sigma, schemes, initials, boundaries


print(DESCRIPTION)
print(VARIANT)
x, n, t, sigma, schemes, initials, boundaries = print_options(args)

h1d7 = EquationParams(d=3, a=1, b=1, c=-1,
                      f=lambda x, t: -np.cos(x)*np.exp(-t),
                      phi=np.sin,
                      dir1_phi=lambda x: np.cos(x),
                      dir2_phi=lambda x: -np.sin(x),
                      psi=lambda x:-np.sin(x),
                      alpha=1, beta=0, mu1=lambda t: np.exp(-t),
                      gamma=1, delta=0, mu2=lambda t: -np.exp(-t),
                      solution=lambda x, t: np.exp(-t) * np.sin(x))

u = []
for scheme_type in schemes:
    for initial_type in initials:
        for boundary_type in boundaries:
            s = f'{scheme_type} {initial_type} {boundary_type}'
            uk, tk = h1d_solver(equation_params=h1d7,
                                x1=x[0], x2=x[1],
                                n=n, sigma=sigma,
                                t1=t[0], t2=t[1],
                                scheme=scheme_type,
                                initial=initial_type,
                                boundary=boundary_type)
            u.append((uk, tk, s))

h = (x[1] - x[0]) / n
x = np.arange(x[0], x[1] + h, h)
u_sol = h1d7.solution(x, tk)
u.append((u_sol, tk, 'solution'))

for uk, _, s in u:
    err = np.linalg.norm(np.abs(u_sol - uk))
    print(f'{s} error: {err}')
    plt.plot(x, uk, label=s)

plt.legend()
plt.show()
