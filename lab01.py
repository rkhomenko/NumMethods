import numpy as np
import matplotlib.pyplot as plt
from tkinter import *


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


# GUI creation procedures

def create_input_one(root, row, col, text, string_var):
    label = Label(root, text=text)
    label.grid(row=row, column=col)
    entry = Entry(root, width=20, textvariable=string_var)
    entry.grid(row=row, column=col + 1)


def create_input_two(root, row, col, text1, text2, string_vars):
    create_input_one(root, row, col, text1, string_vars[0])
    create_input_one(root, row, col + 2, text2, string_vars[1])


def create_listbox(root, row, col, entries):
    list_box = Listbox(root, height=3, width=15, selectmode=SINGLE)
    for entry in entries:
        list_box.insert(END, entry)
    list_box.grid(row=row, column=col)
    return list_box


root = Tk()
root.title('NumMethods Lab 1')

string_vars = [StringVar() for i in range(7)]

create_input_two(root, 0, 0, "x1:", "x2:", string_vars[0:2])
create_input_two(root, 1, 0, "t1:", "t2:", string_vars[2:4])
create_input_one(root, 2, 0, "Steps:", string_vars[4])
create_input_one(root, 3, 0, "Sigma:", string_vars[5])
create_input_one(root, 4, 0, "Teta:", string_vars[6])
scheme = create_listbox(root, 0, 4, ["Explicit", "Implicit", "Krank-Nicolson"])
bound_cond = create_listbox(root, 0, 5, ["2p1", "3p2", "2p2"])


def callback():
    float_vars = [float(s.get()) for s in string_vars]
    print(float_vars)
    x1 = float_vars[0]
    x2 = float_vars[1]
    t1 = float_vars[2]
    t2 = float_vars[3]
    steps = int(float_vars[4])
    sigma = float_vars[5]
    teta = float_vars[6]

    scheme_dict = {0: p1d_explicit_fdm,
                   1: p1d_implicit_fdm,
                   2: lambda p1d, x1, x2, steps, t1, t2, sigma: p1d_crank_nicolson_fdm(p1d, x1, x2, steps, t1, t2,
                                                                                       sigma, teta)}

    p1d_7 = P1D(a=1, b=0, c=0, f=lambda x, t: 0.5 * np.exp(-0.5 * t) * np.sin(x),
                phi=np.sin, alpha=1, beta=0, mu1=lambda t: np.exp(-0.5 * t),
                gamma=1, delta=0, mu2=lambda t: -np.exp(-0.5 * t),
                solution=lambda x, t: np.exp(-0.5 * t) * np.sin(x))
    scheme_type = scheme.curselection()[0]
    xk, u = scheme_dict[scheme_type](p1d_7, x1, x2, steps, t1, t2, sigma)
    error_and_plot(p1d_7, xk, u, t2)


b = Button(root, text="Calc&Plot", command=callback)
b.grid(row=5, column=0)
root.mainloop()
