import numpy as np
import numba as nb


class EquationParams:
    def __init__(self, x, y, a, b, c, d,
                 mu1, mu2, mu3, mu4,
                 solution):
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.mu4 = mu4
        self.solution = solution


@nb.jit(nopython=True)
def norm_inf(arr, nx, ny):
    result = -1
    for j in range(0, ny):
        sum = 0
        for i in range(0, nx):
            sum += arr[i][j]
        if result < sum:
            result = sum

    return result


@nb.jit(nopython=True)
def leibman_calc(u1, u2, nx, ny, a, b, c, d, e, eps):
    counter = 0
    while True:
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u2[i][j] = b * u1[i + 1][j] + c * u1[i - 1][j] + \
                           d * u1[i][j + 1] + e * u1[i][j - 1]
                u2[i][j] /= a

        if norm_inf(np.abs(u2 - u1), nx, ny) < eps:
            break

        counter += 1
        u1, u2 = u2, u1

    return u1


@nb.jit(nopython=True)
def simple_calc(u1, u2, n, eps):
    counter = 0
    while True:
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                u2[i][j] = 1 / 4 * (u1[i + 1][j] + u1[i - 1][j] +
                                    u1[i][j + 1] + u1[i][j - 1])
        if norm_inf(np.abs(u2 - u1), n, n) < eps:
            break

        counter += 1
        u1, u2 = u2, u1

    return u1


def calculate_grid(start, end, steps):
    n = steps + 1
    h = (end - start) / steps
    return n, h


def e2d_solver_simple(e2d, steps_x, steps_y, eps):
    x1, x2 = e2d.x
    y1, y2 = e2d.y
    n, h = calculate_grid(x1, x2, steps_x)

    u1 = np.zeros((n, n), dtype=np.float64)
    u2 = np.zeros((n, n), dtype=np.float64)

    for j in range(0, n):
        y = y1 + j * h
        u1[0][j] = u2[0][j] = e2d.mu1(y)
        u1[n - 1][j] = u2[n - 1][j] = e2d.mu2(y)

    for i in range(0, n):
        x = x1 + i * h
        u1[i][0] = u2[i][0] = e2d.mu3(x)
        u1[i][n - 1] = u2[i][n - 1] = e2d.mu4(x)

    u = simple_calc(u1, u2, n, eps)

    return u, n, n, h, h


def e2d_solver_leibmann(e2d, steps_x, steps_y, eps):
    x1, x2 = e2d.x
    y1, y2 = e2d.y
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

    a = - (2 / hx ** 2 + 2 * e2d.a / hy ** 2 + e2d.d)
    b = e2d.b / 2 / hx - 1 / hx ** 2
    c = - (e2d.b / 2 / hx + 1 / hx ** 2)
    d = e2d.c / 2 / hy - e2d.a / hy ** 2
    e = - (e2d.a / hy ** 2 + e2d.c / 2 / hy)

    u = leibman_calc(u1, u2, nx, ny, a, b, c, d, e, eps)

    return u, nx, ny, hx, hy
