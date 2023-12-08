import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from simulator import TennisSimulator
from utils import HIT_TYPES

class DQN(nn.Module):
    def __init__(self, n_dim_state = 12, n_dim_actions = 10, hidden_dim=64):
        """
        n_dim_state: dimension of state space (player x, player y, ball x, ball y, ball direction)
        n_dim_actions: dimension of action space (player x, player y, hit type * 8)
        hidden_dim: dimension of hidden layer
        """
        super().__init__()
        self.fc1 = nn.Linear(n_dim_state, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_dim_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def run_simulation():
    """
    Runs a simulation and saves the results to tmp.csv
    """
    simulator = TennisSimulator()
    winner, _ = simulator.simulate_match()

    print(f"Player {winner} won the match!")
    print(f"===== Score Board =====")
    print(*simulator.score_board[0], sep=' | ')
    print(*simulator.score_board[1], sep=' | ')
    print(f"=======================")
    
    simulator.export_history("tmp.csv")

class DQNAgent:
    def __init__(self, state_dim, action_dim, discount, learning_rate, TAU=0.01):
        """
        state_dim: dimension of state space (player x, player y, ball x, ball y, ball hit type)
        action_dim: dimension of action space (player x, player y, hit type * 8)
        discount: discount factor
        learning_rate: learning rate
        """
        self.q_network = DQN(state_dim, action_dim).double()
        self.target_q_network = DQN(state_dim, action_dim).double()
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = discount
        self.TAU = TAU

    def update(self, s, a, r, s_prime):
        # TODO: This is really really weird

        # Supposed behavior: # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These are the actions which would've been taken for each batch state according to policy_net
        # But currently returning a continuous value for each column of action
        q_values = self.q_network(s)

        # Supposed behavior: Compute V(s_{t+1}) for all next states. Expected values of actions for non_final_next_states are computed based on the "older" target_net; selecting their best reward with max(1).values
        # But currently also returning a continuous value for each column of action
        next_q_values = self.target_q_network(s_prime).detach()

        # Compute the expected Q values
        target_q_values = r + self.gamma * next_q_values

        loss = F.smooth_l1_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        target_net_state_dict = self.target_q_network.state_dict()
        policy_net_state_dict = self.q_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_q_network.load_state_dict(target_net_state_dict)

    def train(self, num_iter):
        total_loss = []
        for iter in range(num_iter):
            run_simulation()

            # load data
            df = pd.read_csv("tmp.csv")
            cur_loss = []
            for i in range(len(df)):
                receiver = df.loc[i, "receiver"]
                if receiver == 0:
                    continue

                player_pos = convert_str_list(df.loc[i, "player_1_pos"])
                ball_pos = convert_str_list(df.loc[i, "ball_pos"])
                ball_dir = df.loc[i, "ball_dir"]
                hitter_list = convert_type_to_encoding(df.loc[i, "hitter_hit_type"])

                receiver_pos = convert_str_list(df.loc[i, "receiver_movement"])
                receiver_list = convert_type_to_encoding(df.loc[i, "receiver_hit_type"])

                s = torch.tensor(player_pos + ball_pos + [ball_dir] + hitter_list)
                a = torch.tensor(receiver_pos + receiver_list)
                r = torch.tensor([df.loc[i, "player_1_reward"]])

                next_state_idx = -1
                for j in range(i+1, len(df)):
                    if df.loc[j, "receiver"] == receiver:
                        next_state_idx = j
                        break
                # end of game
                if next_state_idx == -1:
                    # TODO: Figure out what to do when the game ends
                    break
                else:
                    player_pos = convert_str_list(df.loc[next_state_idx, "player_1_pos"])
                    ball_pos = convert_str_list(df.loc[next_state_idx, "ball_pos"])
                    ball_dir = df.loc[next_state_idx, "ball_dir"]
                    hitter_list = convert_type_to_encoding(df.loc[next_state_idx, "hitter_hit_type"])

                    sp = torch.tensor(player_pos + ball_pos + [ball_dir] + hitter_list)

                loss = self.update(s, a, r, sp)
                cur_loss.append(loss)
                self.update_target_network()
            print("Simulation: ", iter)
            print("Loss: ", loss)
            total_loss.append(np.mean(cur_loss))
        plt.plot(total_loss)
        plt.savefig("loss.png")
        

def convert_type_to_encoding(hit_type):
    # Create a one-hot vector based on the hit type
    hit_type_idx = HIT_TYPES.index(hit_type)
    hitter_list = [0] * len(HIT_TYPES)
    hitter_list[hit_type_idx] = 1
    return hitter_list

def convert_str_list(str_list):
    return [float(s) for s in str_list.replace("[", "").replace("]", "").split()]

if __name__ == '__main__':
    # Example usage:
    learning_rate = 0.5
    discount_factor = 1.0
    agent = DQNAgent(13, 10, discount_factor, learning_rate)
    agent.train(100)