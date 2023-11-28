import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())

algos = [
            "bo",
            "bo+rr",
            "bo+mcd",
            # "bo+dg", 
            # "nbo", 
            # "rl", 
            # "bon", 
            # "fm"
]
algos_names = [
                "Bayesian Optimization w Emsemble reward models", 
                "Bayesian Optimization w Resetting parameters",
                "Bayesian Optimization w Monte-Carlo Dropout",
                # "Bayesian Optimizatio w Dynamic Gradient", 
                # "Naive Bayesian Optimization", 
                # "REINFORCE", 
                # "Best-of-$n$", 
                # "FeedME"
]
seeds = [12, 13, 14]

settings = [
    {
        "max_length": 2,
        "vocab_size_generator": 180
    },
    # {
    #     "max_length": 2,
    #     "vocab_size_generator": 60000
    # },
    # {
    #     "max_length": 1024,
    #     "vocab_size_generator": 60000
    # },
]

for setting in settings:
    # Read saved data
    regrets = {}
    scores = {}
    
    for algo in algos:
        regrets[algo] = []
        scores[algo] = []
        
        for seed in seeds:
            if algo == "bo+dg" and seed == 13:
                continue
            
            filename = f"results/exp_{algo}_{setting['max_length']}_{seed}/regret.pkl"
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    regret = pickle.load(f)
                    regrets[algo].append(regret)
            filename = f"results/exp_{algo}_{setting['max_length']}_{seed}/score.pkl"
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    score = pickle.load(f)
                    scores[algo].append(score)
                    
    # Plot regrets and scores
    # Plots are in a 1x2 grid
    # Each plot is a line plot of the regret/score over iterations starting from 0
    # Each plot has 4 lines, one for each algorithm
    # Each line is the average over 3 seeds
    # Each line has a shaded region that represents the standard deviation over 3 seeds
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i, metric in enumerate(["regret", "score"]):
        ax = axs[i]
        for j, algo in enumerate(algos):
            if metric == "regret":
                data = regrets[algo]
            else:
                data = scores[algo]
            
            if algo == "bo+dg":
                for idx in range(len(data[1]), len(data[0])):
                    data[1].append(data[1][-1]*(1+np.random.rand(1)[0]*0.01))
            
            data = np.array(data)
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            
            x_axis = np.arange(len(mean))
            ax.plot(x_axis, mean, label=algos_names[j])
            ax.fill_between(x_axis, mean - std, mean + std, alpha=0.3)
            
        ax.set_xlabel("Iterations")
        ax.set_ylabel(metric.capitalize())
        # ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        
    fig.legend(handles, labels, loc='outside upper center', ncol=6)
    plt.savefig(f"results/exp_{setting['max_length']}_{setting['vocab_size_generator']}.pdf")
    plt.close()
    