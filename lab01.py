import numpy as np
import matplotlib.pyplot as plt
import tkinter


class P1D:
    def __init__(self, a, b, c, f, phi, alpha, beta, mu1, gamma, delta, mu2):
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


def p1d_explicit_fdm(p1d, x1, x2, steps_count, t1, t2, sigma):
    n = steps_count + 1
    h = (x2 - x1) / steps_count
    tau = sigma * h ** 2 / p1d.a

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

    # Debug plot
    plt.plot(xk, u1)

    def calculate_u(u, u_prev, t):
        omega = tau * p1d.b / 2 / h
        for i in range(1, n - 1):
            u[i] = (sigma + omega) * u_prev[i + 1] \
                + (1 - 2 * sigma + p1d.c) * u_prev[i] \
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

        plt.plot(xk, u)


p1d = P1D(a=1, b=0, c=0, f=lambda x, t: 0.5 * np.exp(-0.5 * t) * np.sin(x),
          phi=np.sin, alpha=1, beta=0, mu1=lambda t: np.exp(-0.5 * t),
          gamma=1, delta=0, mu2=lambda t: -np.exp(-0.5 * t))
p1d_explicit_fdm(p1d, 0, np.pi, 10, 0, 2, 0.45)
plt.show()
