import torch
import time
from torch.optim.lr_scheduler import LambdaLR
# SiPBA




def SiPBA(hp0,p0,Xg,yg,Xf,yf,Xv,yv,n,alpha_0,beta_0,K=500):
    def inner_func(params, hparams, b=0.1):
        H = hparams[0]
        w = params[0]

        g = (0.5 * (torch.norm(Xg @ H @ w - yg)) ** 2) / n + 0.5 * b * (torch.norm(w)) ** 2
        return g  # .squeeze()

    def outer_func(params, hparams):
        H = hparams[0]
        w = params[0]

        f = (0.5 * (torch.norm(Xf @ H @ w - yf)) ** 2) / n

        return f  # .squeeze()

    def test_func(params, hparams):
        H = hparams[0]
        w = params[0]
        f = (0.5 * (torch.norm(Xv @ H @ w - yv)) ** 2) / n
        return f







    p_SiPBA = 0.01
    q_SiPBA = 0.01
    x = [hp.detach().clone() for hp in hp0]
    x = [hp.requires_grad_(True) for hp in x]
    y=[p.detach().clone() for p in p0]
    y=[p.requires_grad_(True) for p in y]
    z=[p.detach().clone() for p in p0]
    z=[p.requires_grad_(True) for p in z]

    total_time, test_losses, running_time, hg_norms = 0, [], [0], []
    test_loss = test_func(p0, hp0)
    test_losses.append(test_loss.item())

    for i in range(K):
        beta = beta_0 * (i + 1) ** (-2 * p_SiPBA - q_SiPBA)
        alpha = alpha_0 * (i + 1) ** (-8 * p_SiPBA - 8 * q_SiPBA)
        rho = 10 * (i + 1) ** p_SiPBA
        sigma = 1e-4 * (i + 1) ** (-q_SiPBA)
        step_start_time = time.time()
        agg_loss = outer_func(y, x) - rho * (inner_func(y, x, b=0.0) - inner_func(z, x, b=0.0)) - sigma * torch.sum(
            y[0] * z[0]) + sigma / 2 * torch.sum(z[0] ** 2)
        for param in y:
            grad = torch.autograd.grad(agg_loss, param, retain_graph=True)[0]
            param.data += beta * grad
        for param in z:
            grad = torch.autograd.grad(agg_loss, param, retain_graph=True)[0]
            param.data -= beta * grad
        agg_loss = outer_func(y, x) - rho * (inner_func(y, x, b=0.0) - inner_func(z, x, b=0.0))
        for param in x:
            grad = torch.autograd.grad(agg_loss, param, retain_graph=True)[0]
            param.data -= alpha * grad
        step_time = time.time() - step_start_time
        total_time += step_time
        running_time.append(total_time)
        test_loss = test_func(y, x)
        test_losses.append(test_loss.item())
        if (i) % 100 == 0:
            print('SiPBA step={} ({:.2e}s) | test loss={} '.format(i, step_time, test_losses[-1]))
    return running_time, test_losses
