import numpy as np
import numba as nb


class EquationParams:
    def __init__(self, a, b, c, d, e, f,
                 mu1, mu2, mu3, mu4,
                 solution):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.mu4 = mu4
        self.solution = solution


def calculate_grid(start, end, steps):
    n = steps + 1
    h = (end - start) / steps
    return n, h


def e2d_solver_leibmann(e2d, x1, x2, y1, y2, steps_x, steps_y, eps):
    nx, hx = calculate_grid(x1, x2, steps_x)
    ny, hy = calculate_grid(y1, y2, steps_y)

    u1 = np.zeros((nx, ny), dtype=np.float64)
    u2 = np.zeros((nx, ny), dtype=np.float64)

    for j in range(0, ny):
        y = y1 + j * hy
        u1[0][j] = u2[0][j] = e2d.mu1(y)
        u1[nx - 1][j] = u2[nx - 1][j] = e2d.mu2(y)

    for i in range(0, nx):
        x = x1 + i * hx
        u1[i][0] = u2[i][0] = e2d.mu3(x)
        u1[i][ny - 1] = u2[i][ny - 1] = e2d.mu4(x)

    A = - (2 / hx ** 2 + 2 * e2d.a / hy ** 2 - e2d.d)
    B = e2d.b / 2 / hx - 1 / hx ** 2
    C = - (e2d.b / 2 / hx + 1 / hx ** 2)
    D = e2d.c / 2 / hy - e2d.a / hy ** 2
    E = - (e2d.a / hy ** 2 + e2d.c / 2 / hy)

    counter = 0
    while counter != (nx - 2) * (ny - 2):
        counter = 0
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u2[i][j] = B / A * u1[i + 1][j] + C / A * u1[i - 1][j] + \
                    D / A * u1[i][j + 1] + E / A * u1[i][j - 1] + \
                    e2d.e / A * e2d.f(x1 + i * hx, y1 + j * hy)

                if abs(u2[i][j] - u1[i][j]) < eps:
                    counter += 1
        print(counter)
        u1, u2 = u2, u1

    return u1, nx, ny, hx, hy
