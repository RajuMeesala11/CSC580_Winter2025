"""
CSC 580 HW#4 "experiments.py" -- Q-learning for the Snake Game
Name : Om Prakash Gunja
Assignment : HW4
Student ID: 2131025
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to prevent GUI rendering
import matplotlib.pyplot as plt
import pandas as pd
import QLearning as ql

def plot_metrics(results, labels, save_path):
    """Generate visualizations for performance metrics."""
    metrics = ['Mean Returns', 'Mean # of Apples', 'Mean # of Stops', 'Mean # of Good Steps', 'Mean # of States Visited']
    colors = ['green', 'red', 'blue', 'blue', 'purple']
    x_values = range(len(labels))  # Labels for comparison
    
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i + 1)
        plt.plot(x_values, results[:, i], marker='o', color=colors[i], label=metric)
        plt.xticks(x_values, labels, rotation=45, ha='right')
        plt.title(metric)
        plt.xlabel('Experiment')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')  # Close all figures to prevent memory overflow
    
# Parameters
num_runs = 10
num_steps = 1000
eq_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Experiment with different epsilon values
params = dict()
all_results = []

# Step 1: Run using init Q-table and plot metrics
params['gamma'] = 0.95
params['alpha'] = 0.7
params['epsilon'] = 0.5  # Default epsilon for first run
params['epsilon_min'] = .01
params['epsilon_decay'] = .995

results_list = ql.run_ql(num_runs, num_steps, params, "init_qtable_2025_.csv", display=False, train=True)
init_results = np.array(results_list)
plot_metrics(np.mean(init_results, axis=0, keepdims=True), ["Initial"], 'init_qtable.png')

# Step 2: Tune epsilon and plot metrics using best epsilon
all_results = []
for epsilon in eq_values:
    params['epsilon'] = epsilon
    results_list = ql.run_ql(num_runs, num_steps, params, "init_qtable_2025_.csv", display=False, train=True)
    results = np.array(results_list)
    all_results.append(np.mean(results, axis=0))

all_results = np.array(all_results)
plot_metrics(all_results, [f"ε={e}" for e in eq_values], 'best_epsilon_qtable.png')

# Determine best epsilon
best_index = np.argmax(all_results[:, 0] + all_results[:, 1] - all_results[:, 2])
best_epsilon = eq_values[best_index]
print(f"Best epsilon based on performance metrics: {best_epsilon}")
params['epsilon'] = best_epsilon
best_qtable_file = "best_epsilon_qtable.csv"
best_epsilon_results = ql.run_ql(num_runs, num_steps, params, best_qtable_file, display=False, train=True)
best_epsilon_results = np.array(best_epsilon_results)
print(f"Best epsilon Q-table generated from updated Q-values of init_qtable_2025_.csv and saved as {best_qtable_file}")

# Step 3: Compare initial vs best epsilon
comparison_results = np.array([
    np.mean(init_results, axis=0),
    np.mean(best_epsilon_results, axis=0)
])
plot_metrics(comparison_results, ["Initial", f"Best ε={best_epsilon}"], 'comparison_epsilon.png')

# Step 4: Tune alpha and gamma and plot metrics using best values
gamma_values = [0.85, 0.95, 0.99]
alpha_values = [0.5, 0.7, 0.9]
hyperparam_results = []
hyperparam_labels = []

tuned_qtable_file = "tuned_qtable.csv"
for alpha in alpha_values:
    for gamma in gamma_values:
        params['alpha'] = alpha
        params['gamma'] = gamma
        results_list = ql.run_ql(num_runs, num_steps, params, best_qtable_file, display=False, train=True)
        results = np.array(results_list)
        hyperparam_results.append(np.mean(results, axis=0))
        hyperparam_labels.append(f"α={alpha}, γ={gamma}")

hyperparam_results = np.array(hyperparam_results)
plot_metrics(hyperparam_results, hyperparam_labels, 'best_alpha_gamma_qtable.png')

# Determine best alpha and gamma
best_hyperparam_index = np.argmax(hyperparam_results[:, 0] + hyperparam_results[:, 1] - hyperparam_results[:, 2])
best_alpha, best_gamma = [(a, g) for a in alpha_values for g in gamma_values][best_hyperparam_index]
print(f"Best alpha and gamma: α={best_alpha}, γ={best_gamma}")
params['alpha'] = best_alpha
params['gamma'] = best_gamma
tuned_results = ql.run_ql(num_runs, num_steps, params, tuned_qtable_file, display=False, train=True)
tuned_results = np.array(tuned_results)

# Step 5: Compare all three Q-tables
final_comparison_results = np.vstack([
    np.mean(init_results, axis=0),
    np.mean(best_epsilon_results, axis=0),
    np.mean(tuned_results, axis=0)
])
final_labels = ["Initial", f"Best ε={best_epsilon}", f"Best α={best_alpha}, γ={best_gamma}"]
plot_metrics(final_comparison_results, final_labels, 'comparison_all.png')

print("Full hyperparameter tuning and comparison completed. Results saved in respective comparison images.")
