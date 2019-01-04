import numpy as np


class EqParams:
    def __init__(self, a, b, c, f, phi, alpha, beta, mu1, gamma, delta, mu2, solution):
        self.a = a
        self.b = b
        self.c = c
        self.f = f
        self.phi = phi
        self.alpha = alpha
        self.beta = beta
        self.mu1 = mu1
        self.gamma = gamma
        self.delta = delta
        self.mu2 = mu2
        self.solution = solution


class BoundType:
    A1P2 = 0x01
    A2P2 = 0x02
    PHYS = 0x03


def calculate_grid(p1d, x1, x2, steps_count, t1, t2, sigma):
    n = steps_count + 1
    h = (x2 - x1) / steps_count
    tau = sigma * h ** 2 / p1d.a
    omega = tau * p1d.b / 2 / h

    return n, h, tau, omega


def solver_explicit(p1d, x1, x2, steps_count, t1, t2, sigma, bound_type):
    n, h, tau, omega = calculate_grid(p1d, x1, x2, steps_count, t1, t2, sigma)
    x = np.arange(x1, x2 + h, h)
    t = np.arange(t1, t2 + tau, tau)
    ix = np.arange(0, n)

    def left_bound_a1p2(u, u_prev, t):
        return - (p1d.alpha / h) / (p1d.beta - p1d.alpha / h) * u_prev[1] \
               + p1d.mu1(t) / (p1d.beta - p1d.alpha / h)

    def right_bound_a1p2(u, u_prev, t):
        return (p1d.gamma / h) / (p1d.delta + p1d.gamma / h) * u_prev[n - 2] \
               + p1d.mu2(t) / (p1d.delta + p1d.gamma / h)

    def left_bound_a2p2(u, u_prev, t):
        denom = 2 * h * p1d.beta - 3 * p1d.alpha
        return p1d.alpha / denom * u_prev[2] - 4 * p1d.alpha / denom * u_prev[1] \
               + 2 * h / denom * p1d.mu1(t)

    def right_bound_a2p2(u, u_prev, t):
        denom = 2 * h * p1d.delta + 3 * p1d.gamma
        return 4 * p1d.gamma / denom * u_prev[n - 2] - p1d.gamma / denom * u_prev[n - 3] \
               + 2 * h / denom * p1d.mu2(t)

    def left_bound_phys(u, u_prev, t):
        a0 = - 2 * p1d.a ** 2 / h - h / tau + p1d.c * h + \
             p1d.beta / p1d.alpha * (2 * p1d.a ** 2 - p1d.b * h)
        return 1 / a0 * (-2 * p1d.a ** 2 / h * u[1] - h / tau * u_prev[0]
                - h * p1d.f(x[0], t) + (2 * p1d.a ** 2 - p1d.b * h) / p1d.alpha  * p1d.mu1(t))

    def right_bound_phys(u, u_prev, t):
        an = 2 * p1d.a ** 2 / h + h / tau - p1d.c * h + \
             p1d.delta / p1d.gamma * (2 * p1d.a ** 2 + p1d.b * h)
        return 1 / an * (2 * p1d.a ** 2 / h * u[n - 2] + h / tau * u_prev[n - 1]
                         + h * p1d.f(x[n - 1], t) + (2 * p1d.a ** 2 + p1d.b * h) / p1d.gamma * p1d.mu2(t))

    if bound_type == BoundType.A1P2:
        left_bound = left_bound_a1p2
        right_bound = right_bound_a1p2
    elif bound_type == BoundType.A2P2:
        left_bound = left_bound_a2p2
        right_bound = right_bound_a2p2
    elif bound_type == BoundType.PHYS:
        left_bound = left_bound_phys
        right_bound = right_bound_phys
    else:
        raise ValueError("Bad bound type")

    u1 = p1d.phi(x)
    u1[0] = left_bound(u1, u1, t1)
    u1[n - 1] = right_bound(u1, u1, t1)
    u2 = np.zeros(n)
    solution = []

    def calc(i, t, u):
        if i == 0 or i == n - 1:
            return 0

        return (sigma + omega) * u[i + 1] \
               + (1 - 2 * sigma + tau * p1d.c) * u[i] \
               + (sigma - omega) * u[i - 1] \
               + tau * p1d.f(x[i], t)

    u = u2
    for k in range(1, t.size):
        u_prev, u = (u1, u2) if k % 2 != 0 else (u2, u1)
        tk = t1 + k * tau

        for i in ix:
            u[i] = calc(i, tk, u_prev)
        u[0] = left_bound(u, u_prev, tk)
        u[n - 1] = right_bound(u, u_prev, tk)
        solution.append(u)

    return x, u, t[-1], solution


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


def solver_implicit(p1d, x1, x2, steps_count, t1, t2, sigma, bound_type):
    n, h, tau, omega = calculate_grid(p1d, x1, x2, steps_count, t1, t2, sigma)

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    x = np.arange(x1, x2 + h, h)
    t = np.arange(t1, t2 + tau, tau)

    u1 = p1d.phi(x)
    u2 = np.zeros(n)

    for i in range(1, n - 1):
        a[i] = -sigma + omega
        c[i] = 1 + 2 * sigma - p1d.c * tau
        b[i] = -(sigma + omega)

    if bound_type == BoundType.A1P2:
        c[0] = 1
        b[0] = p1d.alpha / h / (p1d.beta - p1d.alpha / h)
        a[n - 1] = - p1d.gamma / h / (p1d.delta + p1d.gamma / h)
        c[n - 1] = 1
    elif bound_type == BoundType.A2P2:
        k1 = 2 * h * p1d.beta - 3 * p1d.alpha
        k2 = 2 * h * p1d.delta + 3 * p1d.gamma
        c[0] = p1d.alpha * (omega - sigma) / k1 / (- sigma - omega) + 1
        b[0] = p1d.alpha * ((1 + 2 * sigma - p1d.c * tau) + 4 * (-sigma - omega)) \
                / k1 / (-sigma - omega)
        a[n - 1] = p1d.gamma * (-(1 + 2 * sigma - p1d.c * tau) - 4 * (-sigma + omega)) \
                / k2 / (-sigma + omega)
        c[n - 1] = p1d.gamma * (omega + sigma) / k2 / (- sigma + omega) + 1
    elif bound_type == BoundType.PHYS:
        c[0] = - 2 * p1d.a ** 2 / h - h / tau + p1d.c * h + \
               p1d.beta / p1d.alpha * (2 * p1d.a ** 2 - p1d.b * h)
        b[0] = 2 * p1d.a ** 2 / h
        a[n - 1] = -b[0]
        c[n - 1] = 2 * p1d.a ** 2 / h + h / tau - p1d.c * h + \
                   p1d.delta / p1d.gamma * (2 * p1d.a ** 2 + p1d.b * h)
    else:
        raise ValueError("Bad bound type")

    u = u2
    for k in range(1, t.size):
        u_prev, u = (u1, u2) if k % 2 != 0 else (u2, u1)
        tk = t1 + k * tau
        if bound_type == BoundType.A1P2:
            d[0] = 1 / (p1d.beta - p1d.alpha / h) * p1d.mu1(tk)
            d[n - 1] = 1 / (p1d.delta + p1d.gamma / h) * p1d.mu2(tk)
        elif bound_type == BoundType.A2P2:
            d[0] = (p1d.alpha * (u_prev[1] + tau * p1d.f(x[0], tk)) +
                        2 * h * (-sigma - omega) * p1d.mu1(tk)) \
                        / k1 / (-sigma - omega)
            d[n - 1] = (- p1d.gamma * (u_prev[n - 2] + tau * p1d.f(x[n - 1], tk)) +
                        2 * h * (-sigma + omega) * p1d.mu2(tk)) \
                        / k2 / (-sigma + omega)
        elif bound_type == BoundType.PHYS:
            d[0] = - h / tau * u_prev[0] - h * p1d.f(x[0], tk) + \
                   (2 * p1d.a ** 2 - p1d.b * h) / p1d.alpha * p1d.mu1(tk)
            d[n - 1] = h / tau * u_prev[0] + h * p1d.f(x[n - 1], tk) + \
                       (2 * p1d.a ** 2 + p1d.b * h) / p1d.alpha * p1d.mu2(tk)
        for i in range(1, n - 1):
            d[i] = u_prev[i] + tau * p1d.f(x[i], tk)

        tdma(u, a, b, c, d)
        print(u)

    return x, u, tk


def solver_explicit_implicit(p1d, x1, x2, steps_count, t1, t2, sigma, kappa, bound_type):
    n, h, tau, omega = calculate_grid(p1d, x1, x2, steps_count, t1, t2, sigma)

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    x = np.arange(x1, x2 + h, h)
    t = np.arange(t1, t2 + tau, tau)

    u1 = p1d.phi(x)
    u2 = np.zeros(n)

    for i in range(1, n - 1):
        a[i] = kappa * (omega - sigma)
        c[i] = 1 + 2 * kappa * sigma - p1d.c * tau * kappa
        b[i] = -(sigma + omega) * kappa

    if bound_type == BoundType.A1P2:
        c[0] = 1
        b[0] = p1d.alpha / h / (p1d.beta - p1d.alpha / h)
        a[n - 1] = - p1d.gamma / h / (p1d.delta + p1d.gamma / h)
        c[n - 1] = 1
    elif bound_type == BoundType.A2P2:
        k1 = 2 * h * p1d.beta - 3 * p1d.alpha
        k2 = 2 * h * p1d.delta + 3 * p1d.gamma
        c[0] = p1d.alpha * (omega - sigma) / k1 / (- sigma - omega) + 1
        b[0] = p1d.alpha * ((1 + 2 * sigma * kappa - p1d.c * tau * kappa) + 4 * kappa * (-sigma - omega)) \
               / k1 / kappa / (-sigma - omega)
        a[n - 1] = p1d.gamma * (-(1 + 2 * sigma * kappa - p1d.c * kappa * tau) - 4 * kappa * (-sigma + omega)) \
                   / k2 / kappa / (-sigma + omega)
        c[n - 1] = p1d.gamma * (omega + sigma) / k2 / (- sigma + omega) + 1
    elif bound_type == BoundType.PHYS:
        c[0] = - 2 * p1d.a ** 2 / h - h / tau + p1d.c * h + \
               p1d.beta / p1d.alpha * (2 * p1d.a ** 2 - p1d.b * h)
        b[0] = 2 * p1d.a ** 2 / h
        a[n - 1] = -b[0]
        c[n - 1] = 2 * p1d.a ** 2 / h + h / tau - p1d.c * h + \
                   p1d.delta / p1d.gamma * (2 * p1d.a ** 2 + p1d.b * h)
    else:
        raise ValueError("Bad bound type")

    def d_i(u, u_prev, i):
        p = 1 - kappa
        return p * (sigma + omega) * u_prev[i + 1] \
               + (1 - 2 * sigma * p + tau * p * p1d.c) * u_prev[i] \
               + p * (sigma - omega) * u_prev[i - 1] \
               + kappa * tau * p1d.f(x[i - i] + kappa * h, tk - tau) + \
               + p * tau * p1d.f(x[i], tk)

    u = u2
    for k in range(1, t.size):
        u_prev, u = (u1, u2) if k % 2 != 0 else (u2, u1)
        tk = t1 + k * tau
        if bound_type == BoundType.A1P2:
            d[0] = 1 / (p1d.beta - p1d.alpha / h) * p1d.mu1(tk)
            d[n - 1] = 1 / (p1d.delta + p1d.gamma / h) * p1d.mu2(tk)
        elif bound_type == BoundType.A2P2:
            d[0] = (2 * h * kappa * (-omega - sigma) * p1d.mu1(tk) + p1d.alpha * d_i(u, u_prev, 1)) / \
                   k1 / kappa / (-omega - sigma)
            d[n - 1] = (-p1d.gamma * d_i(u, u_prev, n - 2) + 2 * h * p1d.mu2(tk) * kappa * (omega - sigma)) / \
                       k2 / kappa / (omega - sigma)
        elif bound_type == BoundType.PHYS:
            d[0] = - h / tau * u_prev[0] - h * p1d.f(x[0], tk) + \
                   (2 * p1d.a ** 2 - p1d.b * h) / p1d.alpha * p1d.mu1(tk)
            d[n - 1] = h / tau * u_prev[0] + h * p1d.f(x[n - 1], tk) + \
                       (2 * p1d.a ** 2 + p1d.b * h) / p1d.alpha * p1d.mu2(tk)

        p = 1 - kappa
        for i in range(1, n - 1):
            d[i] = p * (sigma + omega) * u_prev[i + 1] \
               + (1 - 2 * sigma * p + tau * p * p1d.c) * u_prev[i] \
               + p * (sigma - omega) * u_prev[i - 1] \
               + kappa * tau * p1d.f(x[i - i] + kappa * h, tk - tau) + \
               + p * tau * p1d.f(x[i], tk)

        tdma(u, a, b, c, d)
        print(u)

    return x, u, tk
