"""
CSC 580 HW#4 "Agent.py" -- Class Agent, which performs Temporal Difference (TD) Q-Learning.

"""
import random
import numpy as np
import pandas as pd
class Agent:
    """ 
    An AI agent which controls the snake's movements.
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
        ## TO-DO: Choose your data structure to hold the Q table and initialize it
        self.Q = {}  # Q-table: a dictionary of dictionaries
        self.visited = set() # Keep track of visited states
        

    @staticmethod
    def state_to_int(state_list):
        """ Map state as a list of binary digits, e.g. [0,1,0,0,1,1,1] to an integer."""
        return int("".join(str(x) for x in state_list), 2)
    
    @staticmethod
    def state_to_str(state_list):
        """ Map state as a list of binary digits, e.g. [0,1,0,0,1,1,1], to a string e.g. '0100111'. """
        return "".join(str(x) for x in state_list)

    @staticmethod
    def binstr_to_int(state_str):
        """ Map a state binary string, e.g. '0100111', to an integer."""
        return int(state_str, 2)

    # (A) 
    def init_state(self, state):
        """ Initialize the state's entry in state_table and Q, if anything needed at all."""
        state = self.state_to_str(state)
        if state not in self.Q:
            self.Q[state] = {
                a: np.random.uniform(-1, 1) for a in self.action_space
            }
        
    # (A)
    def select_action(self, state):
        """
        Do the epsilon-greedy action selection. Note: 'state' is an original list of binary digits.
        It should call the function select_greedy() for the greedy case.
        """
        state = self.state_to_str(state)
        if state not in self.Q:
            self.init_state(state)
        
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return self.select_greedy(state)
        #
        #return np.random.choice(self.action_space) # for now

    # (A)
    def select_greedy(self, state):
        """ 
        Greedy choice of action based on the Q-table. 
        """
        state = self.state_to_str(state)
        max_q = max(self.Q[state].values())
        best_actions = [a for a, q in self.Q[state].items() if q == max_q]  # Get all actions with max Q-value
        return random.choice(best_actions)
        #return np.random.choice(self.action_space) # for now
    
    # (A)
    def update_Qtable(self, state, action, reward, next_state):
        """
        Update the Q-table (and anything else necessary) after an action is taken.
        Note that both 'state' and 'next_state' are an original list of binary digits.
        """
        state = self.state_to_str(state)
        next_state = self.state_to_str(next_state)
        self.visited.add(state)
        
        if state not in self.Q:
            self.init_state(state)
        if next_state not in self.Q:
            self.init_state(next_state)
        
        # Q-learning update
        current_q = self.Q[state][action]
        if next_state not in self.Q:
            max_q = 0
        else:
            max_q = max(self.Q[next_state].values())
        new_q = current_q + self.alpha * (reward + self.gamma * max_q - current_q)
        self.Q[state][action] = new_q
        
        # update the epsilon at the end
        self.adjust_epsilon()
        
    # (A)
    def num_states_visited(self):
        """ Returns the number of unique states visited. Obtain from the Q table."""
        #
        #
        return len(self.visited) # for now
    
    # (A)
    def write_qtable(self, filepath):
        """ Write the content of the Q-table to an output file. """
        df = pd.DataFrame.from_dict(self.Q)
        df.to_csv(filepath)
        print(f"Q-table saved to {filepath}")

    # (A)
    def read_qtable(self, filepath):
        """ Read in the Q table saved in a csv file. """
        try:
            df = pd.read_csv(filepath, index_col=0)
            self.Q = df.to_dict()
            print(f"Q-table read from {filepath}")
        except:
            print(f"Error reading Q-table from {filepath}")


    def adjust_epsilon(self):
        """ Implements the epsilon decay. """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
