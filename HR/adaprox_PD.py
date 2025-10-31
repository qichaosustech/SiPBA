import torch
import time
import numpy as np
def Adaprox_PD(hp0,p0,Xg,yg,Xf,yf,Xv,yv,n,params,K=500):
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
    def reg(x,y,v,x0,y0,v0):
        return (torch.sum((x[0]-x0[0])**2)+torch.sum((y[0]-y0[0])**2)+torch.sum((v-v0)**2))

    def inner_solver(x,y,v,x0,y0,v0,sigma,lr, N=100):
        y_n = [p.detach().clone() for p in y]
        y_n = [p.requires_grad_(True) for p in y_n]
        for _ in range(N):
            loss=inner_func(y_n, x)+sigma*reg(x,y,v,x0,y0,v0)
            d_w=torch.autograd.grad(loss,y_n[0])
            y_n[0].data -= lr*d_w[0]
        return y_n
    def h(x,y,y_N, v,x0,y0,v0, sigma,xi,delta):
        h1 = inner_func(y, x)-inner_func(y_N,x)-xi
        h1 = h1.unsqueeze(0)
        h2 = (-torch.autograd.grad(outer_func(y,x),y[0])[0]+v*torch.autograd.grad(inner_func(y, x),y[0])[0])
        h3 = -h2
        h4 = v*h1
        h5 = -h4
        h_tensor = torch.cat([h1, h2, h3, h4, h5], dim=0)
        h_tensor=h_tensor+sigma*reg(x,y,v,x0,y0,v0)*torch.ones_like(h_tensor)
        return h_tensor-delta*torch.ones_like(h_tensor)

    def lagrangian(x,y,y_N, v,x0,y0,v0,lmd, sigma,xi,delta):
        h_tensor = h(x,y,y_N, v,x0,y0,v0, sigma,xi,delta)
        return outer_func(y,x) +sigma*reg(x,y,v,x0,y0,v0)- torch.sum(lmd * h_tensor)



    d = p0[0].shape[0]
    x = [hp.detach().clone() for hp in hp0]
    x = [hp.requires_grad_(True) for hp in x]
    y = [p.detach().clone() for p in p0]
    y = [p.requires_grad_(True) for p in y]
    v = torch.zeros(1).requires_grad_(True)
    lmd = torch.zeros(2*d+3).requires_grad_(True)
    x0 = [hp.detach().clone() for hp in x]
    x0 = [hp.requires_grad_(False) for hp in x0]
    y0 = [p.detach().clone() for p in y]
    y0 = [p.requires_grad_(False) for p in y0]
    v0 =v.detach().clone().requires_grad_(False)



    total_time, test_losses, running_time, hg_norms = 0, [], [0], []
    test_loss = test_func(p0, hp0)
    test_losses.append(test_loss.item())

    beta = params[0]
    eta_0 = params[1]
    tau_0 = params[2]
    xi = params[3]
    sigma = params[4]
    iter = 0
    if 'moment' in locals() or 'moment' in globals():
        del moment
    for _ in range(K):
        out_x = torch.zeros_like(x0[0])
        out_y = torch.zeros_like(y0[0])
        out_v = torch.zeros_like(v0)
        epsilon = beta / (2 * K)
        T = min(20, int(1 / (epsilon) ** 0.5))
        sum_gamma = 0
        start_time = time.time()
        for t in range(T):
            iter += 1
            '''gamma = (t + 1)
            theta = (t + 2) / (t + 1)
            psi = xi + (_ + 1) / K * beta
            eta = eta_0 / (t + 1)
            tau = tau_0 * (t + 1)'''
            gamma = 1
            theta = 1
            psi = xi
            delta=(_ + 1) / K * beta
            eta = eta_0
            tau = tau_0
            N = min(10, int(np.log(1 / epsilon)))
            start = time.time()
            for param in x:
                if param.grad is not None:
                    param.grad = None
            for param in y:
                if param.grad is not None:
                    param.grad = None
            v.grad=None
            lmd.grad=None
            y_N= inner_solver(x,y,v,x0,y0,v0,sigma,lr=1/tau, N=N)
            loss = lagrangian(x,y,y_N, v,x0,y0,v0,lmd, sigma,psi,delta)
            d_lmd = torch.autograd.grad(loss, lmd)[0]
            if 'moment' in locals() or 'moment' in globals():
                d = (1 + theta) * d_lmd - theta * moment
            else:
                d = d_lmd
            moment = d_lmd
            lmd.data += 1 / eta * d
            lmd = torch.clamp(lmd, 0, None)
            lmd.grad = None
            # Compute gradients
            with torch.enable_grad():
                loss = lagrangian(x,y,y_N, v,x0,y0,v0,lmd, sigma,psi,delta)
                grads = torch.autograd.grad(loss, [x[0], y[0], v], retain_graph=False)
            # Update parameters
            with torch.no_grad():
                x[0] -= 1 / tau * grads[0]
                y[0] -= 1 / tau * grads[1]
                v -= 1 / tau * grads[2]
            out_x = out_x + gamma * x[0]
            out_y = out_y + gamma * y[0]
            out_v = out_v + gamma * v
            sum_gamma += gamma
            end = time.time()
        x0 = [(out_x / sum_gamma).detach().requires_grad_(False)]
        y0 = [(out_y / sum_gamma).detach().requires_grad_(False)]
        v0 = (out_v / sum_gamma).detach().requires_grad_(False)
        x = [hp.detach().clone() for hp in x0]
        x = [hp.requires_grad_(True) for hp in x]
        y = [p.detach().clone() for p in y0]
        y = [p.requires_grad_(True) for p in y]
        v = v0.detach().clone().requires_grad_(True)
        step_time=time.time()-start_time
        total_time += step_time
        running_time.append(total_time)
        test_loss = test_func(y, x)
        test_losses.append(test_loss.item())
        if _ % 10 == 0:
            print('AdaProx_PD step={} ({:.2e}s) | test loss={} '.format(_, step_time, test_losses[-1]))
    return running_time, test_losses