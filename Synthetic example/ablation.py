import random
import numpy as np
import time
import torch

def F(x, y):
    e = torch.ones_like(x)
    return torch.sum((x - e) ** 2) / n - torch.sum((y - e) ** 2)
def f(x, y):
    e = torch.ones_like(y)
    return (torch.norm(x) - torch.dot(e, y)) ** 2
def aggregation(x, y, z, rho, sigma):
    return F(x, y) - rho * (f(x, y) - f(x, z)) - sigma * torch.sum(y * z) + sigma / 2 * torch.sum(z ** 2)



n = 100
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)


exps=10
X_list=[]
Y_list=[]

for exp in range(exps):
    bound = 1 / (2 * n ** 0.5)
    x0 = torch.rand(n, requires_grad=True) * (10 - 0.1) + 0.1
    y0 = torch.rand(n, requires_grad=True) * (10 - bound) + bound
    X_list.append(x0)
    Y_list.append(y0)
def run_experiment(alpha_0,beta_0,p,q,s):
    iter_total = []
    time_total_num = []
    for exp in range(exps):
        x0 = X_list[exp]
        y0 = Y_list[exp]
        z0 = Y_list[exp]
        rho_0 = 10
        sigma_0 = 0.01
        iterations = 200000
        time_total = 0
        x=x0.detach().clone().requires_grad_()
        y=y0.detach().clone().requires_grad_()
        z=z0.detach().clone().requires_grad_()
        e = torch.ones_like(x)
        x_star = 0.5 * e
        y_star = e * 1 / (2 * n ** 0.5)
        loss_to_point_0 = 1/n*torch.norm(x - x_star) ** 2 + 1/n*torch.norm(y - y_star) ** 2
        for i in range(iterations):
            loss_to_point = 1/n*torch.norm(x - x_star) ** 2 + 1/n*torch.norm(y - y_star) ** 2
            if loss_to_point/loss_to_point_0 < 1e-4:
                break

            alpha = alpha_0* (i + 1) ** -s
            beta = beta_0 * (i + 1) ** (- 2*p - q)
            eta = beta_0 * (i + 1) ** (-2*p - q)
            rho = rho_0* (i + 1) ** p
            sigma = sigma_0* (i + 1) ** (-q)
            time_start = time.time()
            loss = aggregation(x, y, z, rho, sigma)
            # Manually compute gradients
            d_y = torch.autograd.grad(loss, y, retain_graph=True)[0]
            d_z = torch.autograd.grad(loss, z)[0]

            y.data += beta * d_y
            y = torch.clamp(y, 1 / (2 * n ** 0.5), None)
            z.data -= eta * d_z
            z = torch.clamp(z, 1 / (2 * n ** 0.5), None)

            loss = aggregation(x, y, z, rho, sigma)
            d_x = torch.autograd.grad(loss, x, retain_graph=True)[0]
            x.data -= alpha * d_x
            x = torch.clamp(x, 0.1, 10)
            time_end = time.time()
            time_total = time_total + time_end - time_start
        iter_total.append(i)
        time_total_num.append(time_total)
    print("total iteration %.1f+-(%.1f), total time %.2f+-(%.2f)" % (np.mean(np.array(iter_total)), np.std(np.array(iter_total)),np.mean(np.array(time_total_num)),np.std(np.array(time_total_num))))
def main():
    run_experiment(alpha_0=0.1, beta_0=0.001, p=0.001, q=0.001, s=0.1)
    run_experiment(alpha_0=1, beta_0=0.001, p=0.001, q=0.001, s=0.1)
    run_experiment(alpha_0=0.01, beta_0=0.001, p=0.001, q=0.001, s=0.1)
    run_experiment(alpha_0=0.1, beta_0=0.01, p=0.001, q=0.001, s=0.1)
    run_experiment(alpha_0=0.1, beta_0=0.0001, p=0.001, q=0.001, s=0.1)
    run_experiment(alpha_0=0.1, beta_0=0.001, p=0.01, q=0.001, s=0.1)
    run_experiment(alpha_0=0.1, beta_0=0.001, p=0.0001, q=0.001, s=0.1)
    run_experiment(alpha_0=0.1, beta_0=0.001, p=0.001, q=0.01, s=0.1)
    run_experiment(alpha_0=0.1, beta_0=0.001, p=0.001, q=0.0001, s=0.1)
    run_experiment(alpha_0=0.1, beta_0=0.001, p=0.001, q=0.001, s=0.3)
    run_experiment(alpha_0=0.1, beta_0=0.001, p=0.001, q=0.001, s=0.016)
    run_experiment(alpha_0=0.1, beta_0=0.001, p=0.01, q=0.01, s=0.16)
if __name__ == "__main__":
    main()