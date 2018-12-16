import numpy as np

class BoundaryType:
    A1P2 = 'a1p2'
    A2P3 = 'a2p3'
    A2P2 = 'a2p2'


class InitialConditionType:
    P1 = 'p1'
    P2 = 'p2'


class SchemeType:
    EXPLICIT = 'explicit'
    IMPLICIT = 'implicit'


class EquationParams:
    def __init__(self, d, a, b, c, f,
                 phi, dir1_phi, dir2_phi, psi,
                 alpha, beta, mu1,
                 gamma, delta, mu2,
                 solution):
        self.d = d
        self.a = a
        self.b = b
        self.c = c
        self.f = f
        self.phi = phi
        self.dir1_phi = dir1_phi
        self.dir2_phi = dir2_phi
        self.psi = psi
        self.alpha = alpha
        self.beta = beta
        self.mu1 = mu1
        self.gamma = gamma
        self.delta = delta
        self.mu2 = mu2
        self.solution = solution


def calculate_grid(h1d, x1, x2, steps_count, t1, t2, sigma):
    n = steps_count + 1
    h = (x2 - x1) / steps_count
    tau = np.sqrt(sigma * h ** 2 / h1d.a)
    omega = tau ** 2 * h1d.b / 2 / h

    return n, h, tau, omega


def initial_condition_approximation(h1d,
                                    u_k_2, x, tau,
                                    approximation):
    if approximation == InitialConditionType.P1:
        return u_k_2 + tau * h1d.psi(x)
    elif approximation == InitialConditionType.P2:
        k = tau ** 2 / 2
        return (1 + h1d.c * k) * h1d.phi(x) + \
            h1d.a * k * h1d.dir2_phi(x) + \
            h1d.b * k * h1d.dir1_phi(x) + \
            (tau - h1d.d * k) * h1d.psi(x) + \
            k * h1d.f(x, 0)
    else:
        raise RuntimeError("Initial condition approximation type")


def h1d_solver_explicit(h1d, x1, x2, n, sigma, t1, t2,
                        initial, boundary):
    n, h, tau, omega = calculate_grid(h1d, x1, x2, n, t1, t2, sigma)
    xi = h1d.d * tau / 2

    x = np.arange(x1, x2 + h, h)
    ix = np.arange(0, n - 1)
    t = np.arange(t1, t2 + tau, tau)

    u_k_2 = h1d.phi(x)
    u_k_1 = initial_condition_approximation(h1d, u_k_2, x, tau, initial)
    u_k = np.zeros(u_k_1.shape)
    
    def left_bound_a1p2(u_k, u_k_1, u_k_2, t):
        return - (h1d.alpha / h) / (h1d.beta - h1d.alpha / h) * u_k_1[1] \
               + h1d.mu1(t) / (h1d.beta - h1d.alpha / h)

    def right_bound_a1p2(u_k, u_k_1, u_k_2, t):
        return (h1d.gamma / h) / (h1d.delta + h1d.gamma / h) * u_k_1[n - 2] \
               + h1d.mu2(t) / (h1d.delta + h1d.gamma / h)
    
    def left_bound_a2p3(u_k, u_k_1, u_k_2, t):
        denom = 2 * h * h1d.beta - 3 * h1d.alpha
        return h1d.alpha / denom * u_k_1[2] - 4 * h1d.alpha / denom * u_k_1[1] \
               + 2 * h / denom * h1d.mu1(t)

    def right_bound_a2p3(u_k, u_k_1, u_k_2, t):
        denom = 2 * h * h1d.delta + 3 * h1d.gamma
        return 4 * h1d.gamma / denom * u_k_1[n - 2] - h1d.gamma / denom * u_k_1[n - 3] \
               + 2 * h / denom * h1d.mu2(t)

    def left_bound_a2p2(u_k, u_k_1, u_k_2, t):
        k = h1d.c * h - 2 * h1d.a / h - h / tau ** 2 - h1d.d * h / 2 / tau + \
            h1d.beta / h1d.alpha * (2 * h1d.a - h1d.b * h)
        return 1 / k * (- 2 * h1d.a / h * u_k[1] +
                        h / tau ** 2 * (u_k_2[0] - 2 * u_k_1[0]) +
                        - h1d.d * h / 2 / tau * u_k_2[0] +
                        - h * h1d.f(x[0], t) +
                        (2 * h1d.a - h1d.b * h) / h1d.alpha * h1d.mu1(t))

    def right_bound_a2p2(u_k, u_k_1, u_k_2, t):
        k = - h1d.c * h + 2 * h1d.a / h + h / tau ** 2 + h1d.d * h / 2 / tau + \
            h1d.delta / h1d.gamma * (2 * h1d.a + h1d.b * h)
        return 1 / k * (2 * h1d.a / h * u_k[n - 2] +
                        h / tau ** 2 * (2 * u_k_1[n - 1] - u_k_2[n - 1]) +
                        h1d.d * h / 2 / tau * u_k_2[n - 1] +
                        h * h1d.f(x[n - 1], t) +
                        (2 * h1d.a + h1d.b * h) / h1d.gamma * h1d.mu2(t))

    if boundary == BoundaryType.A1P2:
        left_bound = left_bound_a1p2
        right_bound = right_bound_a1p2
    elif boundary == BoundaryType.A2P3:
        left_bound = left_bound_a2p3
        right_bound = right_bound_a2p3
    elif boundary == BoundaryType.A2P2:
        left_bound = left_bound_a2p2
        right_bound = right_bound_a2p2
    else:
        raise RuntimeError('Boundary approximation type')

    u1 = u_k_2
    u2 = u_k_1
    u3 = u_k

    c = 1 / (1 + xi)
    for k in range(1, t.size):
        tk = t1 + k * tau

        for i in ix:
            u3[i] = c * ((xi - 1) * u1[i] +
                         (sigma - omega) * u2[i - 1] +
                         (h1d.c * tau ** 2 - 2 * sigma + 2) * u2[i] +
                         (sigma + omega) * u2[i + 1] +
                         tau ** 2 * h1d.f(x[i], tk))
        u3[0] = left_bound(u3, u2, u1, tk)
        u3[n - 1] = right_bound(u3, u2, u1, tk)
        tmp = u1
        u1 = u2
        u2 = u3
        u3 = tmp

    return u2, tk


def h1d_solver_implicit(equation_params, x1, x2, n, sigma, t1, t2,
                        initial, boundary):
    pass


def h1d_solver(equation_params, x1, x2, n, sigma, t1, t2,
               scheme, initial, boundary):

    if scheme == SchemeType.EXPLICIT:
        return h1d_solver_explicit(equation_params, x1, x2, n, sigma, t1, t2,
                                   initial, boundary)
    elif scheme == SchemeType.IMPLICIT:
        h1d_solver_implicit(equation_params, x1, x2, n, sigma, t1, t2,
                            initial, boundary)
    else:
        raise ValueError("Scheme type")
