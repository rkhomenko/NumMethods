import numpy as np
import numba as nb


class EquationParams:
    def __init__(self, x, y, t, a, b,
                 f, phi,
                 mu1, mu2, mu3, mu4,
                 solution):
        self.x = x
        self.y = y
        self.t = t
        self.a = a
        self.b = b
        self.f = f
        self.phi = phi
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.mu4 = mu4
        self.solution = solution


class SolverMethod:
    AlterDirections = 'alt_dir'  # Alternating Directions Method
    FractSteps = 'fract_steps'   # Fractional Steps Method


#@nb.jit(nopython=True)
def tdma(u, a, b, c, d):
    n = len(u)
    c = np.copy(c)

    for i in range(1, n):
        m = a[i] / c[i - 1]
        c[i] = c[i] - m * b[i - 1]
        d[i] = d[i] - m * d[i - 1]

    u[n - 1] = d[n - 1] / c[n - 1]

    for i in reversed(range(0, n - 1)):
        u[i] = (d[i] - b[i] * u[i + 1]) / c[i]


#@nb.jit(nopython=True)
def ad_calc(sigma, omega, hx, hy, ht,
            nx, ny, nt,
            x1, x2,
            y1, y2,
            t1, f, phi,
            mu1, mu2, mu3, mu4):
    ax = np.zeros(nx)
    bx = np.zeros(nx)
    cx = np.zeros(nx)
    dx = np.zeros(nx)

    ay = np.zeros(ny)
    by = np.zeros(ny)
    cy = np.zeros(ny)
    dy = np.zeros(ny)

    for i in range(1, nx - 1):
        ax[i] = sigma
        cx[i] = - (2 * sigma + 1)
        bx[i] = sigma

    for j in range(1, ny - 1):
        ay[j] = omega
        cy[j] = - (2 * omega + 1)
        by[j] = omega

    cx[0] = cy[0] = 1
    bx[0] = by[0] = 0
    ax[nx - 1] = ay[ny - 1] = -1
    cx[nx - 1] = cy[ny - 1] = 1

    u1 = np.zeros((nx, ny), dtype=np.float64)
    u2 = np.zeros((nx, ny), dtype=np.float64)
    u3 = np.zeros((nx, ny), dtype=np.float64)
    ux = np.zeros(nx, dtype=np.float64)
    uy = np.zeros(ny, dtype=np.float64)

    for k in range(0, nt):
        tk1 = t1 + (k + 0.5) * ht
        tk2 = t1 + (k + 1) * ht

        # k + 1/2
        for j in range(1, ny):
            yj = y1 + j * hy

            dx[0] = mu1(yj, tk1)
            dx[nx - 1] = hx * mu2(yj, tk1)

            for i in range(1, nx - 1):
                xi = x1 + i * hx
                dx[i] = - omega * u1[i][j + 1] \
                        + (2 * omega - 1) * u1[i][j] \
                        - omega * u1[i][j - 1] \
                        - ht / 2 * f(xi, yj, tk1)

            tdma(ux, ax, bx, cx, dx)
            for i in range(0, nx):
                u2[i][j] = ux[i]


        # k + 1
        for i in range(1, nx):
            xi = x1 + i * hx

            dy[0] = mu3(xi, tk2)
            dy[ny - 1] = hy * mu4(xi, tk2)

            for j in range(1, ny - 1):
                yj = y1 + j * hy
                dy[j] = - sigma * u2[i + 1][j] \
                        + (2 * sigma - 1) * u2[i][j] \
                        - sigma * u2[i - 1][j + 1] \
                        - ht / 2 * f(xi, yj, tk2)

            tdma(uy, ay, by, cy, dy)
            for j in range(0, ny):
                u3[i][j] = uy[j]

        u1, u3 = u3, u1

    return u1


@nb.jit(nopython=True)
def fs_calc(sigma, omega, hx, hy, ht,
            nx, ny, nt,
            x1, y1, t1,
            f, phi,
            mu1, mu2, mu3, mu4):
    pass


def calculate_grid(start, end, steps):
    n = steps + 1
    h = (end - start) / steps
    return n, h


def p2d_solver(method, p2d, steps_x, steps_y, steps_t):
    x1, x2 = p2d.x
    y1, y2 = p2d.y
    t1, t2 = p2d.t
    nx, hx = calculate_grid(x1, x2, steps_x)
    ny, hy = calculate_grid(y1, y2, steps_y)
    nt, ht = calculate_grid(t1, t2, steps_t)
    sigma = p2d.a * ht / 2 / hx ** 2
    omega = p2d.b * ht / 2 / hy ** 2

    results = None
    if method == SolverMethod.AlterDirections:
        results = ad_calc(sigma, omega, hx, hy, ht,
                          nx, ny, nt,
                          x1, x2, y1, y2, t1,
                          p2d.f, p2d.phi,
                          p2d.mu1, p2d.mu2,
                          p2d.mu3, p2d.mu4)
    elif method == SolverMethod.FractSteps:
        results = fs_calc(sigma, omega, hx, hy, ht,
                          nx, ny, nt,
                          x1, x2, y1, y2, t1,
                          p2d.f, p2d.phi,
                          p2d.mu1, p2d.mu2,
                          p2d.mu3, p2d.mu4)
    else:
        raise RuntimeError("Bad method type")

    return results




