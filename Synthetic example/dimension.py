import random
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from matplotlib.ticker import MaxNLocator
import os

def F(x, y):
    e = torch.ones_like(x)
    return torch.sum((x - e) ** 2) / n - torch.sum((y - e) ** 2)


def f(x, y):
    e = torch.ones_like(y)
    return (torch.norm(x) - torch.dot(e, y)) ** 2


def aggregation(x, y, z, rho, sigma):
    return F(x, y) - rho * (f(x, y) - f(x, z)) - sigma * torch.sum(y * z) + sigma / 2 * torch.sum(z ** 2)


time_list = []
iterations_list = []
param_list = [200, 400, 600, 800, 1000]
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

'''for n in param_list:
    num_experiments = 10
    iterations = 200000

    time_total = 0
    iter_total = 0
    for exp in range(num_experiments):
        bound = 1 / (2 * n ** 0.5)
        x = torch.rand(n, requires_grad=True) * (10 - 0.1) + 0.1
        y = torch.rand(n, requires_grad=True) * (10 - bound) + bound
        z = y.detach().clone().requires_grad_()
        e =  torch.ones(size=(n,))
        x_star = 0.5 * e
        y_star = e * 1 / (2 * n ** 0.5)
        loss_to_point_0 = 1 / n * torch.norm(x - x_star) ** 2 + 1 / n * torch.norm(y - y_star) ** 2
        for i in range(iterations):
            if i % 50 == 0:
                loss_to_point = 1/n*torch.norm(x - x_star) ** 2 + 1/n*torch.norm(y - y_star) ** 2
                if loss_to_point/loss_to_point_0   <1e-4:
                    iter_total += i
                    break
            p = 0.001
            q = 0.001
            s = 0.1
            alpha = 0.1 * (i + 1) ** (-s)
            beta = 0.001 * (i + 1) ** (-q - 2 * p)
            rho = 10 * (i + 1) ** p
            sigma = 0.01 * (i + 1) ** (-q)

            time_start = time.time()
            loss = aggregation(x, y, z, rho, sigma)
            # Manually compute gradients
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
            time_end = time.time()
            time_total = time_total + time_end - time_start
    time_list.append(time_total/num_experiments)
    iterations_list.append(iter_total/num_experiments)
    print(f"n:{n}, time:{time_total/num_experiments} iter")
save_data = {
    'time_list': time_list,
    'iterations_list': iterations_list
}
torch.save(save_data, 'results_dimension.pt')'''





loaded_data = torch.load('results_dimension.pt')
time_list = loaded_data['time_list']
iterations_list = loaded_data['iterations_list']

# Create a figure and axis
fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)

# Set the bar width
bar_width = 80  # Adjust bar width to avoid overlap

# Compute the x positions for the bars
index = np.array(param_list)

# Plot the first bar chart: Total Time vs Dimension
ax1.bar(index - bar_width / 2, time_list, bar_width, label='Time (s)', color='b')
ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax2 = ax1.twinx()
ax2.bar(index + bar_width / 2, np.array(iterations_list) / 10 ** 4, bar_width, label='Iterations (×10⁴)', color='red',
        alpha=0.6)
ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax1.set_xlabel('Dimension (n)', fontsize=25)
ax1.tick_params(labelsize=25)
ax2.tick_params(labelsize=25)
ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 0.8), fontsize=25)
ax2.legend(loc='upper left', bbox_to_anchor=(0.0, 1), fontsize=25)
plt.tight_layout(rect=[0, 0, 1, 0.85])
ax1.text(-0.15, 1.02, 'Time', transform=ax1.transAxes, fontsize=22, fontweight='bold', ha='left', va='bottom')
ax2.text(1.12, 1.02, 'Iter', transform=ax2.transAxes, fontsize=22, fontweight='bold', ha='right', va='bottom')

pic_dir = "./pic"
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)
plt.savefig(os.path.join(pic_dir, "dimension.png"), dpi=300)
# Display the plot
plt.show()