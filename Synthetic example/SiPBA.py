import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
# Define the function F(x, y)
def F(x, y):
    e = torch.ones_like(x)
    n=x.shape[0]
    return torch.sum((x - e) ** 2) / n - torch.sum((y - e) ** 2)


def f(x, y):
    e = torch.ones_like(y)
    return (torch.norm(x) - torch.dot(e, y)) ** 2


# Define Aggregation function
def aggregation(x, y, z, rho, sigma):
    return F(x, y) - rho * (f(x, y) - f(x, z)) - sigma * torch.sum(y * z) + sigma / 2 * torch.sum(z ** 2)
torch.manual_seed(123)
num_experiments = 10
iterations = 20000
K = 100
n = 100
x_initial_list = []
y_initial_list = []
point_error_list_all = []
time_list_all= []
x_list_all = []
y_list_all = []
for exp in range(num_experiments):
    print(f"experiment:{exp}/{num_experiments}")

    x_ini = torch.rand(n,) * 9.9 + 0.1  # Random values between 0.1 and 10.0
    y_ini = torch.rand(n,) * (10-1 / (2 * (n ** 0.5))) + 1 / (2 * (n ** 0.5))  # Random values for y
    x_initial_list.append(x_ini)
    y_initial_list.append(y_ini)
    n = x_ini.shape[0]
    x = x_ini.detach().clone()
    y = y_ini.detach().clone()
    z = y_ini.detach().clone()
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True
    e_0 = torch.ones_like(x)

    x_star = e_0 / 2
    y_star = e_0 / (2 * n ** 0.5)
    print(F(x_star, y_star))
    n = x.shape[0]
    point_error_list = []
    time_list = []
    x_list = []
    y_list = []
    time_total = 0
    loss_to_point_0 = 1 / n * torch.norm(x - x_star) ** 2 + 1 / n * torch.norm(y - y_star) ** 2
    for i in range(iterations):
        if (i) % 50 == 0 or i == iterations - 1:
            loss_to_point = 1 / n * torch.norm(x - x_star) ** 2 + 1 / n * torch.norm(y - y_star) ** 2
            print(f"Iters:{(i)} loss:{loss_to_point.item() / loss_to_point_0.item()}")
            point_error_list.append(loss_to_point.item() / loss_to_point_0.item())
            time_list.append(time_total)
            x_list.append(x.detach().clone().detach().numpy())
            y_list.append(y.detach().clone().detach().numpy())
        p = 0.001
        q = 0.001
        s = 0.1
        alpha = 0.1 * (i + 1) ** (-s)
        beta = 0.001 * (i + 1) ** (-q - 2 * p)
        rho = 10 * (i + 1) ** p
        sigma = 0.01 * (i + 1) ** (-q)
        start = time.time()
        loss = aggregation(x, y, z, rho, sigma)
        d_y = torch.autograd.grad(loss, y, retain_graph=True)[0]
        d_z = torch.autograd.grad(loss, z)[0]

        y.data += beta * d_y
        y = torch.clamp(y, 1 / (2 * n ** 0.5), None)
        z.data -= beta * d_z
        z = torch.clamp(z, 1 / (2 * n ** 0.5), None)

        loss = aggregation(x, y, z, rho, sigma)
        d_x = torch.autograd.grad(loss, x, retain_graph=True)[0]
        x.data -= alpha * d_x
        x = torch.clamp(x, 0.1, 10)
        end = time.time()
        time_total += end - start


    point_error_list_all.append(point_error_list)
    time_list_all.append(time_list)
    x_list_all.append(x_list)
    y_list_all.append(y_list)
save_data = {
    'point_error_list_all': point_error_list_all,
    'time_list_all ': time_list_all ,
    'x_list_all': x_list_all,
    'y_list_all': y_list_all,
}

torch.save(save_data, 'results.pt')

save_data={'x': x_initial_list, 'y': y_initial_list}
torch.save(save_data, 'initial.pt')

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
line_SPA, = ax1.semilogy(iteration_steps_k, point_error_mean, label="SiPBA", color='red', linewidth=5)
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
ax2.semilogy(time_list_all_avg_filtered, point_error_mean_filtered, label="SiPBA", color='red', linewidth=5)
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
lines = [line_SPA]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=25)

plt.tight_layout(rect=[0, 0, 1, 0.85])
pic_dir = "./pic"
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)
plt.savefig(os.path.join(pic_dir, "SiPBA.png"), dpi=300)

plt.show()
