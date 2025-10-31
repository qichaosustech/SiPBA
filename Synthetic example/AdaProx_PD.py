import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
torch.random.manual_seed(123)
num_experiments = 10

K = 100
n = 100

point_error_list_all = []
time_list_all= []
x_list_all = []
y_list_all = []
for exp in range(num_experiments):
    print(f"experiment:{exp}/{num_experiments}")

    x_ini = torch.rand(n,) * 9.9 + 0.1  # Random values between 0.1 and 10.0
    y_ini = torch.rand(n,) * (10-1 / (2 * (n ** 0.5))) + 1 / (2 * (n ** 0.5))  # Random values for y

    e_0 = torch.ones_like(x_ini)
    n = x_ini.shape[0]
    x_star = e_0 / 2
    y_star = e_0 / (2 * n ** 0.5)
    x_prox = x_ini.detach().clone().requires_grad_(True)
    y_prox = y_ini.detach().clone().requires_grad_(True)
    y_T = y_prox.detach().clone().requires_grad_(True)
    w = torch.zeros(1).requires_grad_(True)
    v = torch.zeros_like(y_ini).requires_grad_(True)
    lmd = torch.zeros(5 * n + 3).requires_grad_(True)
    x0 = x_prox.detach().clone().requires_grad_(False)
    y0 = y_prox.detach().clone().requires_grad_(False)
    w0 = w.detach().clone().requires_grad_(False)
    v0 = v.detach().clone().requires_grad_(False)

    time_list = []
    x_list = []
    y_list = []
    point_error_list = []
    time_total = 0
    loss_to_point_0 = 1 / n * torch.norm(x_ini - x_star) ** 2 + 1 / n * torch.norm(y_ini - y_star) ** 2
    def F(x, y):
        e = torch.ones_like(x)
        n = x.shape[0]
        return torch.sum((x - e) ** 2) / n - torch.sum((y - e) ** 2)


    def f(x, y):
        e = torch.ones_like(y)
        return (torch.norm(x) - torch.dot(e, y)) ** 2


    def inner_solver(x, y, T=10, lr=0.01):
        y_t = y.detach().clone().requires_grad_(True)

        for i in range(T):
            if y_t.grad is not None:
                y_t.grad.detach_()
                y_t.grad.zero_()
            loss = f(x, y_t) + 0.01 * torch.sum(y_t ** 2)
            d_y = torch.autograd.grad(loss, y_t, create_graph=False)[0]  # Don't create graph
            with torch.no_grad():
                y_t -= lr * d_y
                y_t.data.clamp_(1 / (2 * n ** 0.5), 10)
        return y_t.detach()


    def reg(x, y, w, v, x0, y0, w0, v0, sigma):
        return sigma * (torch.sum((x - x0) ** 2) + torch.sum((y - y0) ** 2) + torch.sum((w - w0) ** 2) + torch.sum(
            (v - v0) ** 2))


    def h(x, y, y_T, w, v, psi=0.1, sigma=0.001,delta=0.001):
        h1 = f(x, y) - f(x, y_T) - psi
        h1 = h1.unsqueeze(0)
        F_loss = F(x, y)
        f_loss = f(x, y)
        h2 = -torch.autograd.grad(F_loss, y, create_graph=True, retain_graph=True)[0] + w * \
             torch.autograd.grad(f_loss, y, create_graph=True, retain_graph=True)[0]
        h3 = -h2
        h4 = w * (f_loss - f(x, y_T) - psi)
        h5 = -h4
        n = x.shape[0]
        h6 = v * (1 / (2 * n ** 0.5) * torch.ones_like(y) - y - psi * torch.ones_like(y))
        h7 = -h6
        h8 = 1 / (2 * n ** 0.5) * torch.ones_like(y) - y - psi * torch.ones_like(y)
        h_tensor = torch.cat([h1, h2, h3, h4, h5, h6, h7, h8], dim=0)
        if sigma > 0:
            reg_term = reg(x, y, w, v, x0, y0, w0, v0, sigma)
            h_tensor = h_tensor + reg_term * torch.ones_like(h_tensor)
        return h_tensor-delta*torch.ones_like(h_tensor)


    def lagrangian(x, y, y_T, w, v, lmd, psi=0.1, sigma=0.001,delta=0.001):
        h_tensor = h(x, y, y_T, w, v, psi, sigma,delta)
        return F(x, y) + reg(x, y, w, v, x0, y0, w0, v0, sigma) - torch.sum(lmd * h_tensor)


    if 'moment' in locals() or 'moment' in globals():
        del moment
    for _ in range(K):

        out_x = torch.zeros_like(x0)
        out_y = torch.zeros_like(y0)
        out_w = torch.zeros_like(w0)
        out_v = torch.zeros_like(v0)
        for i in range(200):
            if (i) % 50 == 0:
                loss_to_point = 1 / n * torch.norm(x_prox - x_star) ** 2 + 1 / n * torch.norm(y_prox - y_star) ** 2
                print(f"Iters:{_ * 200 + (i)} loss:{loss_to_point.item() / loss_to_point_0.item()}")
                point_error_list.append(loss_to_point.item() / loss_to_point_0.item())
                time_list.append(time_total)
                x_list.append(x_prox.detach().clone().detach().numpy())
                y_list.append(y_prox.detach().clone().detach().numpy())
            lr = 0.001
            lmd_lr = 0.001
            psi = 0.1
            sigma = 0.001
            T = 10
            mu = 1.0
            beta = 0.001
            psi = psi
            delta=(_ + 1) / K * beta
            start = time.time()
            for param in [x_prox, y_prox, y_T, w, v, lmd]:
                if param.grad is not None:
                    param.grad = None
            y_T = inner_solver(x_prox, y_prox, T=T, lr=lr)
            loss = lagrangian(x_prox, y_prox, y_T, w, v, lmd, psi=psi, sigma=sigma,delta=delta)
            d_lmd = torch.autograd.grad(loss, lmd)[0]
            if 'moment' in locals() or 'moment' in globals():
                d = (1 + mu) * d_lmd - mu * moment
            else:
                d = d_lmd
            moment = d_lmd.detach()
            lmd.data += lmd_lr * d
            lmd = torch.clamp(lmd, 0, None)
            lmd.grad = None
            with torch.enable_grad():
                L = lagrangian(x_prox, y_prox, y_T, w, v, lmd, psi=psi, sigma=sigma,delta=delta)
                grads = torch.autograd.grad(L, [x_prox, y_prox, w, v], retain_graph=False)
            with torch.no_grad():
                x_prox -= lr * grads[0]
                y_prox -= lr * grads[1]
                w -= lr * grads[2]
                v -= lr * grads[3]
                x_prox.clamp_(min=0.1, max=10)
                y_prox.clamp_(min=1 / (2 * n ** 0.5), max=10)
                w.clamp_(min=0)
                v.clamp_(min=0)
            out_x = out_x + x_prox
            out_y = out_y + y_prox
            out_w = out_w + w
            out_v = out_v + v
            end = time.time()
            time_total += end - start
        x0 = (out_x / 200).detach().requires_grad_(False)
        y0 = (out_y / 200).detach().requires_grad_(False)
        w0 = (out_w / 200).detach().requires_grad_(False)
        v0 = (out_v / 200).detach().requires_grad_(False)
        x_prox = x0.clone().detach().requires_grad_(True)
        y_prox = y0.clone().detach().requires_grad_(True)
        w = w0.clone().detach().requires_grad_(True)
        v = v0.clone().detach().requires_grad_(True)
        if _==K-1:
            loss_to_point = 1 / n * torch.norm(x_prox - x_star) ** 2 + 1 / n * torch.norm(y_prox - y_star) ** 2
            print(f"Iters:{_ * 200 + i+1} loss:{loss_to_point.item() / loss_to_point_0.item()}")
            point_error_list.append(loss_to_point.item() / loss_to_point_0.item())
            time_list.append(time_total)
            x_list.append(x_prox.detach().clone().detach().numpy())
            y_list.append(y_prox.detach().clone().detach().numpy())
    point_error_list_all.append(point_error_list)
    time_list_all.append(time_list)
    x_list_all.append(x_list)
    y_list_all.append(y_list)


save_data = {
    'point_error_PD_list_all': point_error_list_all,
    'time_list_PD_all ': time_list_all ,
    'x_PD_list_all': x_list_all,
    'y_PD_list_all': y_list_all,
}

torch.save(save_data, 'results_PD.pt')

# Load the data from the files
point_error_mean = np.mean(np.array(point_error_list_all), axis=0)
point_error_min = np.min(np.array(point_error_list_all), axis=0)
point_error_max = np.max(np.array(point_error_list_all), axis=0)

max_iteration = 20000
iteration_steps = range(0, max_iteration, 50)
point_error_mean = point_error_mean[:len(iteration_steps)]
point_error_min = point_error_min[:len(iteration_steps)]
point_error_max = point_error_max[:len(iteration_steps)]



iteration_steps = np.array(range(0, max_iteration, 50))
iteration_steps_k = iteration_steps / 1000.0


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# === Plot 1: Loss vs Iteration ===
line_PD, = ax1.semilogy(iteration_steps_k, point_error_mean, label="AdaProx-PD", color='red', linewidth=5)
ax1.fill_between(iteration_steps_k, point_error_min, point_error_max, color='red', alpha=0.2)

ax1.xaxis.set_major_locator(ticker.MaxNLocator(5))
ax1.set_xlabel(r"Iteration ($\times10^3$)", fontsize=25)
ax1.set_ylabel(r"Relative error ($\epsilon_{rel}$)", fontsize=20)
ax1.grid(True, which='major')
ax1.minorticks_off()
ax1.tick_params(axis='y', labelsize=20)
ax1.tick_params(axis='x', labelsize=20)

# === Plot 2: Loss vs Time ===
time_list_all_avg = np.mean(np.array(time_list_all), axis=0)
time_limit = 20 # Set the time limit

# Filter the data by time
time_list_all_avg_filtered = time_list_all_avg[time_list_all_avg <= time_limit]
point_error_mean_filtered = point_error_mean[:len(time_list_all_avg_filtered)]
point_error_min_filtered = point_error_min[:len(time_list_all_avg_filtered)]
point_error_max_filtered = point_error_max[:len(time_list_all_avg_filtered)]

# Plot the loss to point vs time for each method
ax2.semilogy(time_list_all_avg_filtered, point_error_mean_filtered, label="AdaProx-PD", color='red', linewidth=5)
ax2.fill_between(time_list_all_avg_filtered, point_error_min_filtered, point_error_max_filtered, color='red', alpha=0.2)

ax2.xaxis.set_major_locator(ticker.MaxNLocator(5))
# Set y-axis and x-axis ticks
ax2.yaxis.set_major_locator(ticker.LogLocator(numticks=5))
ax2.set_xlabel('Time (S)', fontsize=25)

# Set grid
ax2.grid(True, which='major')
ax2.minorticks_off()
ax2.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='x', labelsize=20)

# Combine legends from all methods
lines = [line_PD]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=25)

plt.tight_layout(rect=[0, 0, 1, 0.85])
pic_dir = "./pic"
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)
plt.savefig(os.path.join(pic_dir, "AdaProx-PD.png"), dpi=300)

plt.show()
