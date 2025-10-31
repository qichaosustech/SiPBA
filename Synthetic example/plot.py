import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
# Load the data from the files
loaded_data = torch.load('results_PD.pt')
point_error_PD_list_all = loaded_data['point_error_PD_list_all']
time_list_PD_all  = loaded_data['time_list_PD_all ']
x_PD_list_all = loaded_data['x_PD_list_all']
y_PD_list_all = loaded_data['y_PD_list_all']

loaded_data = torch.load('results_SG.pt')
point_error_SG_list_all = loaded_data['point_error_list_all']
time_list_SG_all  = loaded_data['time_list_all ']
x_SG_list_all = loaded_data['x_list_all']
y_SG_list_all = loaded_data['y_list_all']

loaded_data = torch.load('results.pt')
point_error_list_all = loaded_data['point_error_list_all']
time_list_all = loaded_data['time_list_all ']
x_list_all = loaded_data['x_list_all']
y_list_all = loaded_data['y_list_all']

# Compute mean before taking the log
point_error_mean = np.mean(np.array(point_error_list_all), axis=0)
point_error_PD_mean = np.mean(np.array(point_error_PD_list_all), axis=0)
point_error_SG_mean = np.mean(np.array(point_error_SG_list_all), axis=0)

# Compute min and max for each
point_error_min = np.min(np.array(point_error_list_all), axis=0)
point_error_max = np.max(np.array(point_error_list_all), axis=0)

point_error_PD_min = np.min(np.array(point_error_PD_list_all), axis=0)
point_error_PD_max = np.max(np.array(point_error_PD_list_all), axis=0)

point_error_SG_min = np.min(np.array(point_error_SG_list_all), axis=0)
point_error_SG_max = np.max(np.array(point_error_SG_list_all), axis=0)

max_iteration = 20000
iteration_steps = range(0, max_iteration, 50)

# Slice for iteration steps
point_error_mean = point_error_mean[:len(iteration_steps)]
point_error_min = point_error_min[:len(iteration_steps)]
point_error_max = point_error_max[:len(iteration_steps)]

point_error_PD_mean = point_error_PD_mean[:len(iteration_steps)]
point_error_PD_min = point_error_PD_min[:len(iteration_steps)]
point_error_PD_max = point_error_PD_max[:len(iteration_steps)]

point_error_SG_mean = point_error_SG_mean[:len(iteration_steps)]
point_error_SG_min = point_error_SG_min[:len(iteration_steps)]
point_error_SG_max = point_error_SG_max[:len(iteration_steps)]

iteration_steps = np.array(range(0, max_iteration, 50))
iteration_steps_k = iteration_steps / 1000.0

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# === Plot 1: Loss vs Iteration ===
line_SPA, = ax1.semilogy(iteration_steps_k, point_error_mean, label="SiPBA", color='red', linewidth=5)
ax1.fill_between(iteration_steps_k, point_error_min, point_error_max, color='red', alpha=0.2)

line_PD, = ax1.semilogy(iteration_steps_k, point_error_PD_mean, label="AdaProx-PD", color='blue', linewidth=5)
ax1.fill_between(iteration_steps_k, point_error_PD_min, point_error_PD_max, color='blue', alpha=0.2)

line_SG, = ax1.semilogy(iteration_steps_k, point_error_SG_mean, label="AdaProx-SG", color='green', linewidth=5)
ax1.fill_between(iteration_steps_k, point_error_SG_min, point_error_SG_max, color='green', alpha=0.2)

ax1.xaxis.set_major_locator(ticker.MaxNLocator(5))
ax1.set_xlabel(r"Iteration ($\times10^3$)", fontsize=25)
ax1.set_ylabel(r"Relative error ($\epsilon_{rel}$)", fontsize=20)
ax1.grid(True, which='major')
ax1.minorticks_off()
ax1.tick_params(axis='y', labelsize=20)
ax1.tick_params(axis='x', labelsize=20)

# === Plot 2: Loss vs Time ===
time_list_all_avg = np.mean(np.array(time_list_all), axis=0)
time_list_PD_all_avg = np.mean(np.array(time_list_PD_all), axis=0)
time_list_SG_all_avg = np.mean(np.array(time_list_SG_all), axis=0)

# === Filter data to only show time <= 25 seconds ===
time_limit = 20  # Set the time limit

# Filter the data by time
time_list_all_avg_filtered = time_list_all_avg[time_list_all_avg <= time_limit]
point_error_mean_filtered = point_error_mean[:len(time_list_all_avg_filtered)]
point_error_min_filtered = point_error_min[:len(time_list_all_avg_filtered)]
point_error_max_filtered = point_error_max[:len(time_list_all_avg_filtered)]

time_list_PD_all_avg_filtered = time_list_PD_all_avg[time_list_PD_all_avg <= time_limit]
point_error_PD_mean_filtered = point_error_PD_mean[:len(time_list_PD_all_avg_filtered)]
point_error_PD_min_filtered = point_error_PD_min[:len(time_list_PD_all_avg_filtered)]
point_error_PD_max_filtered = point_error_PD_max[:len(time_list_PD_all_avg_filtered)]

time_list_SG_all_avg_filtered = time_list_SG_all_avg[time_list_SG_all_avg <= time_limit]
point_error_SG_mean_filtered = point_error_SG_mean[:len(time_list_SG_all_avg_filtered)]
point_error_SG_min_filtered = point_error_SG_min[:len(time_list_SG_all_avg_filtered)]
point_error_SG_max_filtered = point_error_SG_max[:len(time_list_SG_all_avg_filtered)]

# Plot the loss to point vs time for each method
ax2.semilogy(time_list_all_avg_filtered, point_error_mean_filtered, label="SiPBA", color='red', linewidth=5)
ax2.fill_between(time_list_all_avg_filtered, point_error_min_filtered, point_error_max_filtered, color='red', alpha=0.2)

ax2.semilogy(time_list_PD_all_avg_filtered, point_error_PD_mean_filtered, label="AdaProx-PD", color='blue', linewidth=5)
ax2.fill_between(time_list_PD_all_avg_filtered, point_error_PD_min_filtered, point_error_PD_max_filtered, color='blue', alpha=0.2)

ax2.semilogy(time_list_SG_all_avg_filtered, point_error_SG_mean_filtered, label="AdaProx-SG", color='green', linewidth=5)
ax2.fill_between(time_list_SG_all_avg_filtered, point_error_SG_min_filtered, point_error_SG_max_filtered, color='green', alpha=0.2)

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
lines = [line_SPA, line_PD, line_SG]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=25)

plt.tight_layout(rect=[0, 0, 1, 0.85])
pic_dir = "./pic"
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)
plt.savefig(os.path.join(pic_dir, "convergence.png"), dpi=300)
plt.show()
