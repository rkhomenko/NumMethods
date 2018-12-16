import P1D
import numpy as np
import matplotlib.pyplot as plt


def error_and_plot(p1d, xk, u, t, name):
    solution = [p1d.solution(x, t) for x in xk]
    error = [np.abs(solution[i] - u[i]) for i in range(len(xk))]
    print(f"Error ({name}): ", np.linalg.norm(error, np.inf))
    plt.plot(xk, u, label=name)


p1d7 = P1D.EqParams(a=1, b=0, c=0, f=lambda x, t: 0.5 * np.exp(-0.5 * t) * np.sin(x),
                    phi=np.sin, alpha=1, beta=0, mu1=lambda t: np.exp(-0.5 * t),
                    gamma=1, delta=0, mu2=lambda t: -np.exp(-0.5 * t),
                    solution=lambda x, t: np.exp(-0.5 * t) * np.sin(x))

x, u, t, sol = P1D.solver_explicit(p1d7, 0, np.pi, 20, 0, 1.5, 0.45, P1D.BoundType.A1P2)
solution = [p1d7.solution(x, t) for x in x]
plt.plot(x, solution, ':', label='Solution')
error_and_plot(p1d7, x, u, t, 'Explicit')
x, u, t = P1D.solver_implicit(p1d7, 0, np.pi, 20, 0, 1.5, 0.45, P1D.BoundType.A1P2)
error_and_plot(p1d7, x, u, t, 'Implicit')
x, u, t = P1D.solver_explicit_implicit(p1d7, 0, np.pi, 20, 0, 1.5, 0.45, 0.5, P1D.BoundType.A1P2)
error_and_plot(p1d7, x, u, t, 'Explicit-Implicit')

plt.legend()
plt.show()
