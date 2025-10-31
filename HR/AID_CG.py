import torch
import time
import hypergrad as hg




def AID_CG(hp0, p0, Xg, yg, Xf, yf, Xv, yv, n, alpha, beta,K=300):
    b = 0.1
    eval_interval = 100
    T = 20
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

    def map_func(params, hparams):
        g = inner_func(params, hparams)
        # inner_losses.append(g.item())
        # print(torch.norm(torch.autograd.grad(g, params, create_graph=True)[0]))

        return [params[0] - alpha * torch.autograd.grad(g, params, create_graph=True)[0]]

    def regressor(params, Z, y):
        w = params[0]
        loss = (torch.norm(Z @ w - y)) ** 2
        return loss

    def inner_solver(hparams, steps=100, params0=None, optim=None):
        H = hparams[0]
        Zg = Xg @ H

        # params = [torch.randn(d).requires_grad_(True)]
        params = [p.requires_grad_(True) for p in p0]

        for _ in range(steps):
            loss = 0.5 * regressor(params, Zg, yg) / n + 0.5 * b * (torch.norm(params[0])) ** 2
            params = [params[0] - alpha * torch.autograd.grad(loss, params, create_graph=True)[0]]

        return params

    hparams = [hp.clone() for hp in hp0]
    hparams = [hp.requires_grad_(True) for hp in hparams]

    outer_opt = torch.optim.Adam(lr=beta, params=hparams)

    total_time, test_losses, running_time, hg_norms = 0, [], [0], []
    test_loss = test_func(p0, hp0)
    test_losses.append(test_loss.item())
    for k in range(K):

        step_start_time = time.time()
        inner_losses = []
        params = inner_solver(hparams, steps=T)
        t1 = time.time() - step_start_time  # inner loop time

        outer_opt.zero_grad()
        _, cost = hg.CG(params, hparams, T, map_func, outer_func, set_grad=True)
        t2 = time.time() - step_start_time - t1  # hypergrad estimation time

        outer_opt.step()

        step_time = time.time() - step_start_time
        total_time += step_time
        test_loss = test_func(params, hparams)
        test_losses.append(test_loss.item())
        running_time.append(total_time)
        hg_norms.append(torch.norm(hparams[0].grad))

        if (k+1) % eval_interval == 0 or k == K - 1:
            print('AID-CG outer step={} ({:.2e}s)({:.2e}, {:.2e}) | test loss={} | hypergrad norm = {:.3e}'.format(k+1, step_time,
                                                                                                           t1,
                                                                                                           t2,
                                                                                                           test_losses[
                                                                                                               -1],
                                                                                                           torch.norm(
                                                                                                               hparams[
                                                                                                                   0].grad)))

    return running_time, test_losses