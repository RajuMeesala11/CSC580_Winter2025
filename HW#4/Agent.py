"""
CSC 580 HW#4 "Agent.py" -- Q-learning for the Snake Game
Name : Om Prakash Gunja
Assignment : HW4
Student ID: 2131025
"""

import random
import numpy as np
import pandas as pd

class Agent:
    """
    An AI agent that performs Q-learning for Snake.
    """
    def __init__(self, env, params):
        self.env = env
        self.action_space = env.action_space  # 4 actions for SnakeGame
        self.state_space = env.state_space    # 12 features for SnakeGame
        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.Q = {}  # Q-table as a dictionary of dictionaries
        self.visited = set()  # Keep track of visited states

    @staticmethod
    def state_to_int(state_list):
        """ Convert a binary state list to an integer. """
        return int("".join(str(x) for x in state_list), 2)

    def init_state(self, state):
        """ Ensure a state is initialized in the Q-table. """
        state_int = self.state_to_int(state)
        if state_int not in self.Q:
            self.Q[state_int] = {a: 0.0 for a in range(self.action_space)}

    def select_action(self, state):
        """ Epsilon-greedy action selection. """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)  # Explore
        else:
            return self.select_greedy(state)  # Exploit

    def select_greedy(self, state):
        """ Select the best action based on Q-values (exploitation). """
        state_int = self.state_to_int(state)
        if state_int not in self.Q:
            self.init_state(state)
            return np.random.choice(self.action_space)
        
        return max(self.Q[state_int], key=self.Q[state_int].get)

    def update_Qtable(self, state, action, reward, next_state, done):
        """ Update the Q-table using the Q-learning algorithm. """
        state_int = self.state_to_int(state)
        next_state_int = self.state_to_int(next_state)
        self.visited.add(state_int)

        # Ensure states exist in Q-table
        self.init_state(state)
        self.init_state(next_state)

        # Terminal state handling
        max_next_q = 0 if done else max(self.Q[next_state_int].values())

        # Q-learning update rule
        self.Q[state_int][action] += self.alpha * (
            reward + self.gamma * max_next_q - self.Q[state_int][action]
        )

        # Decay epsilon
        self.adjust_epsilon()

    def num_states_visited(self):
        """ Returns the number of unique states visited. """
        return len(self.visited)

    def write_qtable(self, filepath):
        """ Save Q-table to CSV. """
        data = [[s, a, q] for s, actions in self.Q.items() for a, q in actions.items()]
        pd.DataFrame(data, columns=['state', 'action', 'q_value']).to_csv(filepath, index=False)
        print(f"Q-table saved to {filepath}")

    def read_qtable(self, filepath):
        """ Load Q-table from CSV. """
        try:
            df = pd.read_csv(filepath)
            self.Q = {}
            for _, row in df.iterrows():
                state, action, q_value = int(row['state']), int(row['action']), float(row['q_value'])
                if state not in self.Q:
                    self.Q[state] = {}
                self.Q[state][action] = q_value
            print(f"Q-table loaded from {filepath}")
        except Exception as e:
            print(f"Error reading Q-table: {e}")

    def adjust_epsilon(self):
        """ Implements epsilon decay. """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
