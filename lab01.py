import numpy as np
import matplotlib.pyplot as plt
import tkinter


class P1D:
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


def calculate_steps(p1d, x1, x2, steps_count, t1, t2, sigma):
    n = steps_count + 1
    h = (x2 - x1) / steps_count
    tau = sigma * h ** 2 / p1d.a

    return n, h, tau


def p1d_explicit_fdm(p1d, x1, x2, steps_count, t1, t2, sigma):
    n, h, tau = calculate_steps(p1d, x1, x2, steps_count, t1, t2, sigma)

    u1 = np.zeros(n)
    u2 = np.zeros(n)
    xk = np.zeros(n)

    for i in range(0, n):
        xk[i] = x1 + i * h

    # Bound values
    def u_left(u, t):
        return - (p1d.alpha / h) / (p1d.beta - p1d.alpha / h) * u[1] \
               + p1d.mu1(t) / (p1d.beta - p1d.alpha / h)

    def u_right(u, t):
        return (p1d.gamma / h) / (p1d.delta + p1d.gamma / h) * u[n - 2] \
               + p1d.mu2(t) / (p1d.delta + p1d.gamma / h)

    # Apply initial condition
    for i in range(1, n - 1):
        u1[i] = p1d.phi(x1 + i * h)

    u1[0] = u_left(u1, 0)
    u1[n - 1] = u_right(u1, 0)

    def calculate_u(u, u_prev, t):
        omega = tau * p1d.b / 2 / h
        for i in range(1, n - 1):
            u[i] = (sigma + omega) * u_prev[i + 1] \
                + (1 - 2 * sigma + tau * p1d.c) * u_prev[i] \
                + (sigma - omega) * u_prev[i - 1] \
                + tau * p1d.f(xk[i], t)

            u[0] = u_left(u_prev, t)
            u[n - 1] = u_right(u_prev, t)

    tk = t1 + tau
    step = 0
    while tk <= t2:
        if step % 2 == 0:
            u_prev = u1
            u = u2
        else:
            u_prev = u2
            u = u1

        calculate_u(u, u_prev, tk)
        tk += tau
        step += 1

    return xk, u


def tdma(u, a, b, c, d, k1, k2):
    n = len(u)

    k1[0] = -c[0] / b[0]
    k2[0] = d[0] / b[0]

    for i in range(1, n - 1):
        k1[i] = - c[i] / (b[i] + a[i] * k1[i - 1])
        k2[i] = (d[i] - a[i] * k2[i - 1]) / (b[i] + a[i] * k1[i - 1])

    k1[n - 1] = 0
    k2[n - 1] = (d[n - 1] - a[n - 1] * k2[n - 2]) / (b[n - 1] + a[n - 1] * k1[n - 2])

    u[n - 1] = k2[n - 1]

    for i in reversed(range(0, n - 1)):
        u[i] = k1[i] * u[i + 1] + k2[i]


def p1d_implicit_fdm(p1d, x1, x2, steps_count, t1, t2, sigma):
    n, h, tau = calculate_steps(p1d, x1, x2, steps_count, t1, t2, sigma)
    omega = tau * p1d.b / 2 / h

    u1 = np.zeros(n)
    u2 = np.zeros(n)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    xk = np.zeros(n)
    k1 = np.zeros(n)
    k2 = np.zeros(n)

    a[0] = 0
    b[0] = 1
    c[0] = p1d.alpha / h / (p1d.beta - p1d.alpha / h)

    for i in range(1, n - 1):
        a[i] = omega - sigma
        b[i] = 1 + 2 * sigma - p1d.c * tau
        c[i] = -(sigma + omega)

    a[n - 1] = - p1d.gamma / h / (p1d.delta + p1d.gamma / h)
    b[n - 1] = 1
    c[n - 1] = 0

    for i in range(0, n):
        xk[i] = x1 + i * h
        u1[i] = p1d.phi(xk[i])

    tk = t1 + tau
    step = 0
    while tk <= t2:
        if step % 2 == 0:
            u_prev = u1
            u = u2
        else:
            u_prev = u2
            u = u1

        d[0] = 1 / (p1d.beta - p1d.alpha / h) * p1d.mu1(tk)
        for i in range(1, n - 1):
            d[i] = u_prev[i] + tau * p1d.f(xk[i], tk)
        d[n - 1] = 1 / (p1d.delta + p1d.gamma / h) * p1d.mu2(tk)

        tdma(u, a, b, c, d, k1, k2)

        tk += tau
        step += 1

    return xk, u


def p1d_crank_nicolson_fdm(p1d, x1, x2, steps_count, t1, t2, sigma, teta):
    n, h, tau = calculate_steps(p1d, x1, x2, steps_count, t1, t2, sigma)
    omega = tau * p1d.b / 2 / h

    u1 = np.zeros(n)
    u2 = np.zeros(n)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    xk = np.zeros(n)
    k1 = np.zeros(n)
    k2 = np.zeros(n)

    a[0] = 0
    b[0] = 1
    c[0] = p1d.alpha / h / (p1d.beta - p1d.alpha / h)

    for i in range(1, n - 1):
        a[i] = teta * (omega - sigma)
        b[i] = 1 + teta * (2 * sigma - p1d.c * tau)
        c[i] = - teta * (sigma + omega)

    a[n - 1] = - p1d.gamma / h / (p1d.delta + p1d.gamma / h)
    b[n - 1] = 1
    c[n - 1] = 0

    for i in range(0, n):
        xk[i] = x1 + i * h
        u1[i] = p1d.phi(xk[i])

    f = np.zeros(n)
    tk_prev = t1
    tk = t1 + tau
    step = 0
    while tk <= t2:
        if step % 2 == 0:
            u_prev = u1
            u = u2
        else:
            u_prev = u2
            u = u1

        d[0] = 1 / (p1d.beta - p1d.alpha / h) * p1d.mu1(tk)
        for i in range(1, n - 1):
            d[i] = (1 - teta) * (sigma + omega) * u_prev[i + 1] + \
                   (1 + (1 - teta) * (- 2 * sigma + p1d.c * tau)) * u_prev[i] + \
                   (1 - teta) * (sigma - omega) * u_prev[i - 1] + \
                   teta * tau * p1d.f(xk[i], tk_prev + teta * tau) + \
                   (1 - teta) * tau * p1d.f(xk[i], tk_prev + (1 - teta) * tau)

        d[n - 1] = 1 / (p1d.delta + p1d.gamma / h) * p1d.mu2(tk)

        tdma(u, a, b, c, d, k1, k2)

        tk_prev = tk
        tk += tau
        step += 1

    return xk, u


def error_and_plot(p1d, xk, u, t):
    solution = [p1d.solution(x, t) for x in xk]
    error = [np.abs(solution[i] - u[i]) for i in range(len(xk))]
    print(error)
    plt.plot(xk, u, label='Calculated')
    plt.plot(xk, solution, label='Solution', ls='--')
    plt.legend()
    plt.show()


a = 1
b = 1
c = -1
t1 = 0
t2 = 1
sigma = 0.45
p1d_7 = P1D(a=a, b=0, c=0, f=lambda x, t: 0.5 * np.exp(-0.5 * t) * np.sin(x),
            phi=np.sin, alpha=1, beta=0, mu1=lambda t: np.exp(-0.5 * t),
            gamma=1, delta=0, mu2=lambda t: -np.exp(-0.5 * t),
            solution=lambda x, t: np.exp(-0.5 * t) * np.sin(x))
p1d_10 = P1D(a=a, b=b, c=c, f=lambda x, t: 0,
             phi=np.sin, alpha=1, beta=1, mu1=lambda t: np.exp((c - a) * t) * (np.cos(b * t) + np.sin(b * t)),
             gamma=1, delta=1, mu2=lambda t: -np.exp((c - a) * t) * (np.cos(b * t) + np.sin(b * t)),
             solution=lambda x, t: np.exp((c - a) * t) * np.sin(x + b * t))
# xk, u = p1d_explicit_fdm(p1d_10, 0, np.pi, 100, t1, t2, sigma)
# error_and_plot(p1d_10, xk, u, t2)
# xk, u = p1d_implicit_fdm(p1d_10, 0, np.pi, 20, t1, t2, 0.4)
# error_and_plot(p1d_10, xk, u, t2)
xk, u = p1d_crank_nicolson_fdm(p1d_7, 0, np.pi, 20, t1, t2, sigma, 0.5)
error_and_plot(p1d_7, xk, u, t2)
