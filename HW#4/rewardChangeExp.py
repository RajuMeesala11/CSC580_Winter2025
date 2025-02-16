"""
CSC 580 HW#4 "rewardChangeExp" -- Q-learning for the Snake Game
Name : Om Prakash Gunja
Assignment : HW4
Student ID: 2131025
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent GUI rendering issues
import matplotlib.pyplot as plt
import QLearning as ql

# Parameters
num_runs = 10
num_steps = 1000

# Use the final tuned Q-table from previous best hyperparameters
tuned_qtable = "tuned_qtable.csv"
new_qtable = "reward_adjusted_qtable.csv"

params = {
    'gamma': 0.85,  # Best gamma from previous tuning
    'alpha': 0.7,   # Best alpha from previous tuning
    'epsilon': 0.7,  # Controlled exploration level
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995
}

# Step 1: Run Q-learning using the existing best-tuned Q-table
print("Running Q-learning with the current best-tuned Q-table...")
initial_results = ql.run_ql(num_runs, num_steps, params, tuned_qtable, display=False, train=False)
initial_results = np.array(initial_results)

# Step 2: Modify reward function in SnakeEnv.py manually before proceeding
print("Modify SnakeEnv.py to increase penalty for self-collision, then re-run this script.")
input("Press Enter after modifying the reward function...")

# Step 3: Run Q-learning again with the updated reward function
print("Running Q-learning after reward adjustment...")
updated_results = ql.run_ql(num_runs, num_steps, params, new_qtable, display=False, train=True)
updated_results = np.array(updated_results)

# Step 4: Compare results
def plot_comparison(before_results, after_results, labels, save_path):
    metrics = ['Mean Returns', 'Mean # of Apples', 'Mean # of Stops', 'Mean # of Good Steps', 'Mean # of States Visited']
    colors = ['green', 'red', 'blue', 'blue', 'purple']
    x_values = range(len(labels))
    
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i + 1)
        plt.plot(x_values, [np.mean(before_results[:, i]), np.mean(after_results[:, i])], marker='o', color=colors[i], label=metric)
        plt.xticks(x_values, labels, rotation=45, ha='right')
        plt.title(metric)
        plt.xlabel('Experiment')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

# Save the comparison plot
plot_comparison(initial_results, updated_results, ["Before Reward Change", "After Reward Change"], "reward_change_comparison.png")

print("Comparison complete. Results saved to 'reward_change_comparison.png'.")
