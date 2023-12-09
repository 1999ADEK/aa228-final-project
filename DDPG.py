import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from simulator import TennisSimulator
from utils import HIT_TYPES

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

class Actor(nn.Module):
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
    
class Critic(nn.Module):
    def __init__(self, n_dim_state = 12, n_dim_actions = 10, hidden_dim=64):
        """
        n_dim_state: dimension of state space (player x, player y, ball x, ball y, ball direction)
        n_dim_actions: dimension of action space (player x, player y, hit type * 8)
        hidden_dim: dimension of hidden layer
        """
        super().__init__()
        self.fc1 = nn.Linear(n_dim_state + n_dim_actions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def add(self, s, a, r, s_prime):
        self.buffer.append((s, a, r, s_prime))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.buffer), size=batch_size)
        state, next_state, action, reward = [], [], [], []

        for i in ind:
            s, a, r, s_prime = self.buffer[i]
            state.append(np.array(s, copy=False))
            next_state.append(np.array(s_prime, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))

        return np.array(state), np.array(action), np.array(reward), np.array(next_state)

    

class DDPGAgent:
    def __init__(self, state_dim, action_dim, discount, TAU=0.001, capacity=1000):
        """
        state_dim: dimension of state space (player x, player y, ball x, ball y, ball hit type)
        action_dim: dimension of action space (player x, player y, hit type * 8)
        discount: discount factor
        learning_rate: learning rate
        """
        self.replay_buffer = ReplayBuffer(capacity)

        self.actor = Actor(state_dim, action_dim).double()
        self.critic = Critic(state_dim, action_dim).double()
        self.target_actor = Actor(state_dim, action_dim).double()
        self.target_critic = Critic(state_dim, action_dim).double()

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=2e-3)
        self.gamma = discount
        self.TAU = TAU

    def update(self, batch_size=64):
        # Sample a batch of transitions from replay buffer
        s, a, r, s_prime = self.replay_buffer.sample(batch_size)
        s = torch.tensor(s)
        a = torch.tensor(a)
        r = torch.tensor(r)
        s_prime = torch.tensor(s_prime)

        # Compute the Q values
        q_values = self.critic(torch.cat((s, a), dim=1))
        target_q_values = self.target_critic(torch.cat((s_prime, self.target_actor(s_prime)), dim=1))
        target = r + self.gamma * target_q_values

        critic_loss = F.mse_loss(q_values, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss as the negative mean Q value using the critic network and the actor network
        actor_loss = -self.critic(torch.cat((s, self.actor(s)), dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()


    def update_target_network(self, network, target_network):
        target_net_state_dict = target_network.state_dict()
        cur_net_state_dict = network.state_dict()
        for key in cur_net_state_dict:
            target_net_state_dict[key] = cur_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        target_network.load_state_dict(target_net_state_dict)
        return target_network
    
    def save_model(self):
        torch.save(self.actor.state_dict(), "model/actor.pt")
        torch.save(self.critic.state_dict(), "model/critic.pt")

    def load_model(self):
        self.actor.load_state_dict(torch.load("model/actor.pt"))
        self.critic.load_state_dict(torch.load("model/critic.pt"))



    def train(self, num_iter):
        total_act_loss = []
        total_crit_loss = []
        for iter in range(num_iter):
            run_simulation()

            # load data
            df = pd.read_csv("tmp.csv")
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

                self.replay_buffer.add(s, a, r, sp)

            actor_loss, critic_loss = self.update()
            self.target_actor = self.update_target_network(self.actor, self.target_actor)
            self.target_critic = self.update_target_network(self.critic, self.target_critic)

            print("Simulation: ", iter)
            print("Actor Loss: ", actor_loss)
            print("Critic Loss: ", critic_loss)
            total_act_loss.append(actor_loss)
            total_crit_loss.append(critic_loss)

            if iter % 100 == 0:
                self.save_model()
                print("Model saved")

        plt.plot(total_crit_loss, label="critic loss")
        plt.plot(total_act_loss, label="actor loss")
        plt.legend()
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
    learning_rate = 0.0001
    discount_factor = 0.99
    agent = DDPGAgent(13, 10, discount_factor, learning_rate)
    agent.train(1000)