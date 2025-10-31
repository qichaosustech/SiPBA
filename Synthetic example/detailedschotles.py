import numpy as np
from scipy.optimize import fsolve
import torch
import time

loaded_data = torch.load('initial.pt')
x_ini_list = loaded_data['x']
y_ini_list = loaded_data['y']


x_list = np.array(x_ini_list)
y_list = np.array(y_ini_list)
time_list_all=[]
x_list_all=[]
y_list_all=[]
loss_list_all=[]
def error(x,y,n):
    e_0 = np.ones_like(x)
    x_star = e_0 / 2
    y_star = e_0 / (2 * n ** 0.5)
    loss_to_point = 1 / n * np.linalg.norm(x - x_star) ** 2 + 1 / n * np.linalg.norm(y - y_star) ** 2
    return loss_to_point
def solve_kkt_system():
    experiments=10

    for exp in range(experiments):
        n = 100
        np.random.seed(0)
        x0 = x_ini_list[exp]
        y0 = y_ini_list[exp]
        u0 = np.zeros(n)
        alpha_0 = np.zeros(2 * n)
        beta_0 = np.zeros(n)
        gamma_0 = np.zeros(n)
        delta_0 = np.zeros(n)
        mu_0 = np.zeros(n)
        initial_guess = np.concatenate([x0, y0, u0, alpha_0, beta_0, gamma_0, delta_0,mu_0])
        error0 = error(x0, y0, n)
        t = 0.1
        solution = initial_guess
        max_iter = 10

        time_list = []
        x_list = []
        y_list = []
        loss_list = []
        time_total=0
        time_list.append(time_total)
        x_list.append(x0)
        y_list.append(y0)
        relative_error = error0 / error0
        loss_list.append(relative_error)
        for k in range(max_iter):

            fun = lambda vars: kkt_system(vars, t, n)
            time_start = time.time()
            solution, infodict, ier, msg = fsolve(fun, solution, full_output=True, xtol=1e-6, maxfev=10000)
            time_end = time.time()
            if ier != 1:
                print(f"Warning: fsolve did not converge properly at iteration {k}. Message: {msg}")
            x = solution[:n]
            y= solution[n:2 * n]
            time_total += time_end - time_start
            time_list.append(time_total)
            x_list.append(x)
            y_list.append(y)
            errork=error(x, y, n)
            relative_error = errork / error0
            loss_list.append(relative_error)
            violations = kkt_system(solution, t, n)
            max_violation = np.max(np.abs(violations))
            print(f"Max violation: {max_violation:.4e}")
            print('Optimal x* (first 5 elements):')
            print(x[:5])
            print('Optimal y* (first 5 elements):')
            print(y[:5])
            t = t * 0.1
        time_list_all.append(time_list)
        x_list_all.append(x_list)
        y_list_all.append(y_list)
        loss_list_all.append(loss_list)
    save_data = {
        'point_error_list_all': loss_list_all,
        'time_list_all ': time_list_all,
        'x_list_all': x_list_all,
        'y_list_all': y_list_all,
    }
    torch.save(save_data, 'results_detailed.pt')
    return solution


def kkt_system(vars, epsilon, n):
    x = vars[:n]
    y = vars[n:2 * n]
    u = vars[2 * n:3 * n]
    alpha = vars[3 * n:5 * n]
    beta = vars[5 * n:6 * n]
    gamma = vars[6 * n:7 * n]
    delta = vars[7 * n:8 * n]
    mu = vars[8 * n:9 * n]

    F_val = F_func(x, y, n)
    f_val = f_func(x, y, n)
    G_val = G_func(x, n)
    g_val = g_func(x, y, n)
    l_val = l_func(x, y, u)
    grad_F_x_val = grad_F_x(x, y, n)
    grad_F_y_val = grad_F_y(x, y, n)
    grad_G_x_val = grad_G_x(x, n)
    grad_L_x_val = jacobian_l_x(x, y, u)
    grad_L_y_val = jacobian_l_y(x, y, u)
    grad_g_x_val = grad_g_x(x, y, n)
    grad_g_y_val = grad_g_y(x, y, n)

    t_vec = epsilon * np.ones(n)

    # Equations
    eq1 = grad_F_x_val + grad_G_x_val.T @ alpha + grad_L_x_val.T @ beta + grad_g_x_val.T @ (delta * u - gamma)
    eq2 = grad_F_y_val + grad_L_y_val.T @ beta + grad_g_y_val.T @ (delta * u - gamma)
    eq3 = mu-grad_g_y_val @ beta + delta * g_val
    eq4 = np.sqrt(alpha ** 2 + G_val ** 2 + 2 * epsilon) - alpha + G_val
    eq5 = np.sqrt(gamma ** 2 + g_val ** 2 + 2 * epsilon) - gamma + g_val
    eq6 = np.sqrt(u ** 2 + mu ** 2 + 2 * epsilon) - u +mu
    eq7 = np.sqrt(delta ** 2 + (-u * g_val - t_vec) ** 2 + 2 * epsilon) - delta + (-u * g_val - t_vec)
    eq8 = l_val
    '''eq1 = grad_F_x_val + grad_G_x_val.T @ alpha + grad_L_x_val.T @ beta + grad_g_x_val.T @ (delta * u - gamma)
    eq2 = grad_F_y_val + grad_L_y_val.T @ beta + grad_g_y_val.T @ (delta * u - gamma)
    eq3 = np.sqrt(alpha ** 2 + G_val ** 2 + 2 * epsilon) - alpha + G_val
    eq4 = np.sqrt(gamma ** 2 + g_val ** 2 + 2 * epsilon) - gamma + g_val
    eq5 = np.sqrt(u ** 2 + (grad_g_y_val @ beta - delta * g_val) ** 2 + 2 * epsilon) - u + (
                grad_g_y_val @ beta - delta * g_val)
    eq6 = np.sqrt(delta ** 2 + (-u * g_val - t_vec) ** 2 + 2 * epsilon) - delta + (-u * g_val - t_vec)
    eq7 = l_val'''

    return np.concatenate([eq1, eq2, eq3, eq4, eq5, eq6, eq7,eq8])


def F_func(x, y, n):
    e = np.ones(n)
    return (1 / n) * np.linalg.norm(x - e) ** 2 - np.linalg.norm(y - e) ** 2


def f_func(x, y, n):
    e = np.ones(n)
    return (np.linalg.norm(x) - np.dot(y, e)) ** 2


def G_func(x, n):
    e = np.ones(n)
    return np.concatenate([0.1 * e - x, x - 10 * e])


def g_func(x, y, n):
    e = np.ones(n)
    return (1 / (2 * np.sqrt(n))) * e - y


def grad_F_x(x, y, n):
    e = np.ones(n)
    return (2 / n) * (x - e)


def grad_F_y(x, y, n):
    e = np.ones(n)
    return -2 * (y - e)


def grad_G_x(x, n):
    return np.vstack([-np.eye(n), np.eye(n)])

def l_func(x, y, u):
    n = len(x)
    e = np.ones(n)
    norm_x = np.linalg.norm(x)
    return -2 * (norm_x - e @ y) * e - u

def jacobian_l_x(x, y, u):
    n = len(x)
    e = np.ones(n)
    norm_x = np.linalg.norm(x)
    if norm_x == 0:
        return np.zeros((n, n))
    return -2 * np.outer(x / norm_x, e)


def jacobian_l_y(x, y, u):
    n = len(x)
    e = np.ones(n)
    return 2 * np.outer(e, e)

def grad_g_x(x, y, n):
    return np.zeros((n, n))

def grad_g_y(x, y, n):
    return -np.eye(n)

# Run the solver
solution = solve_kkt_system()