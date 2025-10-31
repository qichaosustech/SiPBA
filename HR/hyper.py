import os
import numpy as np
from SIPBA import *
from PZOBO import *
from AID_FP import *
from AID_CG import *
from adaprox_PD import *
from adaprox_SG import *
def run_experiment(p,d,n,m):
    for seed in range(10):
        print(f"\n=== Running Experiment {seed + 1}/10 ===")
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        H_true = torch.randn((p, d))
        w_true = torch.randn(d)
        Xg = torch.randn((n, p)) + m*torch.randn((n, p))
        yg = Xg @ H_true @ w_true + m * torch.randn(n)
        Xf = torch.randn((n, p))+ m*torch.randn((n, p))
        yf = Xf @ H_true @ w_true + m * torch.randn(n)
        Xv = torch.randn((n, p))
        yv = Xv @ H_true @ w_true
        p0 = [torch.randn(d)]
        hp0 = [torch.randn((p, d))]




        



        #SiPBA
        if n==500 and m==0.1:
            alpha_0=0.0005
            beta_0=0.0005
        else:
            alpha_0=0.0001
            beta_0=0.0001
        K=1000

        running_time, test_losses= SiPBA(hp0,p0,Xg,yg,Xf,yf,Xv,yv,n,alpha_0,beta_0,K)
        run_sipba = running_time
        test_sipba = test_losses



        #Adaprox_PD
        K = 100
        beta = 0.001
        xi = 0.001
        sigma = 0.001
        if n == 500 and m == 0.1:
            eta_0 = 10000
            tau_0 = 10000
        elif n == 1000 and m == 0.1:
            eta_0 = 20000
            tau_0 = 20000
        elif n == 500 and m == 1:
            eta_0 = 20000
            tau_0 = 20000
        elif n == 1000 and m == 1:
            eta_0 = 50000
            tau_0 = 50000


        params = [beta, eta_0, tau_0, xi, sigma]
        running_time, test_losses = Adaprox_PD(hp0, p0, Xg, yg, Xf, yf, Xv, yv, n, params, K)
        run_pd = running_time
        test_pd = test_losses

        # AdaProx-SG
        beta = 0.001
        gamma_0 = 1000
        xi = 0.001
        sigma = 0.001
        K = 100
        params = [beta, gamma_0, xi, sigma]
        running_time, test_losses = Adaprox_SG(hp0, p0, Xg, yg, Xf, yf, Xv, yv, n, params, K)
        run_sg = running_time
        test_sg = test_losses



        #PZOBO
        alpha=0.0001
        beta=0.05
        mu=0.01
        K=300
        running_time, test_losses = PZOBO(hp0, p0, Xg, yg, Xf, yf, Xv, yv, n,alpha,beta,mu,K)
        test_pzobo = test_losses
        run_pzobo = running_time

        # AID-CG
        alpha = 0.0001
        if n == 500:
            beta = 0.05
        else:
            beta = 0.01
        K = 300
        running_time, test_losses = AID_CG(hp0, p0, Xg, yg, Xf, yf, Xv, yv, n, alpha, beta, K)
        test_cg = test_losses
        run_cg = running_time

        # AID-FP
        alpha = 0.0001
        beta = 0.05
        K = 300
        running_time, test_losses = AID_FP(hp0, p0, Xg, yg, Xf, yf, Xv, yv, n, alpha, beta, K)
        test_fp = test_losses
        run_fp = running_time


        os.makedirs('result', exist_ok=True)
        np.savez(
            f"result/experiment_results_n_{n}_m_{m}_run_{seed}.npz",
            run_sipba = np.array(run_sipba),
            test_sipba = np.array(test_sipba),
            run_pd = np.array(run_pd),
            test_pd = np.array(test_pd),
            run_sg = np.array(run_sg),
            test_sg = np.array(test_sg),
            run_pzobo = np.array(run_pzobo),
            test_pzobo = np.array(test_pzobo),
            run_cg = np.array(run_cg),
            test_cg = np.array(test_cg),
            run_fp = np.array(run_fp),
            test_fp = np.array(test_fp)
        )

        print("All result lists saved to experiment_results.npz")
def plot_results():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    n_list = [500,1000]
    m_list = [0.1,1]
    nm_list = [[n, m] for m in m_list for n in n_list]
    algorithms = ["sipba", "pzobo", "cg", "fp","pd",'sg']
    labels = ["SiPBA", "PZOBO", "AID-CG", "AID-FP","AdaProx-PD",'AdaProx-SG']
    linewidth = 3
    max_time = 100
    num_runs = 10
    # Define custom colors
    line_colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"] # Blue, Orange, Green, Red
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    axes = axes.flatten()

    for idx, (n, m) in enumerate(nm_list):
        ax = axes[idx]
        test_dict = {key: [] for key in algorithms}
        run_dict = {key: [] for key in algorithms}

        for run_id in range(num_runs):
            path = f"result/experiment_results_n_{n}_m_{m}_run_{run_id}.npz"
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")
            data = np.load(path)

            for key in algorithms:
                run_dict[key].append(data[f"run_{key}"])
                test_dict[key].append(data[f"test_{key}"])

        # Check if any values are non-positive
        for key in algorithms:
            for run_data in test_dict[key]:
                if np.any(run_data <= 0):
                    test_dict[key] = [np.maximum(v, 1e-10) for v in test_dict[key]]  # Replace non-positive values

        # Interpolate to a common time axis
        x_common = np.linspace(0, max_time, 10000)
        for key, label, color in zip(algorithms, labels, line_colors):
            test_interp = []
            for r, v in zip(run_dict[key], test_dict[key]):
                if len(r) < 2:
                    continue
                r = np.array(r)
                v = np.array(v)
                log_v = np.log10(v)
                test_interp.append(np.interp(x_common, r, log_v))

            test_interp = np.array(test_interp)
            mean_log_tests = np.mean(test_interp, axis=0)
            std_log_tests = np.std(test_interp, axis=0)

            mean_tests = 10 ** mean_log_tests
            lower = 10 ** (mean_log_tests - std_log_tests)
            upper = 10 ** (mean_log_tests + std_log_tests)

            ax.semilogy(x_common, mean_tests, label=label, linewidth=linewidth, color=color)
            ax.fill_between(x_common, lower, upper, alpha=0.2, color=color)

        # Set axis ticks
        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=5))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))

        if n == 500 and m == 0.1:
            ax.set_title(r"$m=500,n=64$" + "\n" + r"$p=128,a=0.1$", fontsize=25)
            ax.set_xlim(0, 10)
            ax.set_ylim(1e-5, 1e5)
        elif n == 1000 and m == 0.1:
            ax.set_title(r"$m=1000,n=128$" + "\n" + r"$p=256,a=0.1$", fontsize=25)
            ax.set_xlim(0, 10)
            ax.set_ylim(1e-5, 1e5)
        elif n == 500 and m == 1:
            ax.set_title(r"$m=500,n=64$" + "\n" + r"$p=128,a=1$", fontsize=25)
            ax.set_xlim(0, 10)
            ax.set_ylim(1e-2, 1e5)
        elif n == 1000 and m == 1:
            ax.set_title(r"$m=1000,n=128$" + "\n" + r"$p=256,a=1$", fontsize=25)
            ax.set_xlim(0, 10)
            ax.set_ylim(1e-2, 1e5)
        ax.set_xlabel("Time (s)", fontsize=25)
        ax.grid(True, which="major", ls="--")
        ax.tick_params(axis='both', which='minor', length=0, labelsize=25)
        ax.tick_params(axis='y', labelsize=25)
        ax.tick_params(axis='x', labelsize=25)
        if idx == 0:
            ax.set_ylabel("Test loss", fontsize=25)

    # Legend and layout adjustments
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6, fontsize=25)
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    pic_dir = "./pic"
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    plt.savefig(os.path.join(pic_dir, "convergence.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    #p,d,n,m corresponds to n,p,m,a in the paper
    '''run_experiment(p=64, d=128, n =500, m=0.1)
    run_experiment(p=64, d=128, n=500, m=1)
    run_experiment(p=128, d=256, n=1000, m=0.1)
    run_experiment(p=128, d=256, n=1000, m=1)'''
    plot_results()

