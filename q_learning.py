import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor

from utils import HIT_TYPES, POSITIONS, State, Action
from simulator import TennisSimulator

import pickle

import warnings
warnings.filterwarnings("ignore")


def run_simulation():
    simulator = TennisSimulator()
    winner, scores = simulator.simulate_match()

    print(f"Player {winner} won the match!")
    print(f"===== Score Board =====")
    print(*simulator.score_board[0], sep=' | ')
    print(*simulator.score_board[1], sep=' | ')
    print(f"=======================")
    
    simulator.export_history("tmp.csv")


def get_state_index(player_pos, ball_pos, hit_type):
    player_pos_idx = POSITIONS.index(player_pos)
    ball_pos_idx = POSITIONS.index(ball_pos)
    hit_type_idx = HIT_TYPES.index(hit_type)
    return player_pos_idx * len(POSITIONS) * len(HIT_TYPES) + ball_pos_idx * len(HIT_TYPES) + hit_type_idx



def get_action_index(receive_pos, receive_type):
    pos_idx = POSITIONS.index(receive_pos)
    hit_type_idx = HIT_TYPES.index(receive_type)
    return pos_idx * len(HIT_TYPES) + hit_type_idx



def q_learning(num_states, num_actions, lr=0.5, gamma=0.99):

    Q = np.zeros((num_states, num_actions))


    max_diff = float('-Inf')  # track updates

    iter = 0
    while iter < 1000:
        run_simulation()

        # load data
        df = pd.read_csv("tmp.csv")

        for i in range(len(df)):
            receiver = df.loc[i, "receiver"]
            if receiver == 0:
                continue

            hit_type = df.loc[i, "hitter_hit_type"]
            # if hit_type == "forehand_serve":
            #     continue


            # s, a, r, sp = df[i]
            pos1, ball_pos = df.loc[i, "player_1_pos"], df.loc[i, "ball_pos"]
            receive_type = df.loc[i, "receiver_hit_type"]
            receive_pos = df.loc[i, "receiver_movement"]
            r1 = df.loc[i, "player_1_reward"]

            r1_next = r1
            if i < len(df)-1:
                r1_next = df.loc[i+1, "player_1_reward"]

        
            # player pos, ball pos, hit type
            s = get_state_index(pos1, ball_pos, hit_type)
            # receive pos, receive type
            a = get_action_index(receive_pos, receive_type)

            r = r1_next - r1 + 10.1

            next_state_idx = -1
            for j in range(i+1, len(df)):
                if df.loc[j, "receiver"] == receiver:
                    next_state_idx = j
                    break
            
            # end of game
            if next_state_idx == -1:
                sp = num_states - 1
            else:
                pos1 = df.loc[next_state_idx, "player_1_pos"]
                ball_pos = df.loc[i, "ball_pos"]
                hit_type = df.loc[i, "hitter_hit_type"]
                sp = get_state_index(pos1, ball_pos, hit_type)

            # print(s, a, r, sp)
            delta = r + gamma * np.max(Q[sp, :]) - Q[s, a]
            Q[s, a] += lr * delta

            max_diff = max(max_diff, delta)

        iter += 1
        print(iter, max_diff)
        print(np.count_nonzero(Q))
    
    return Q


def get_best_actions(Q_table):
    return np.argmax(Q_table, axis=1)


def approx_q_values(Q_table, num_states, num_actions):
    Q = np.zeros_like(Q_table)

    for a in range(num_actions):
        receive_type = a % len(HIT_TYPES)
        if HIT_TYPES[receive_type] == "forehand_serve":
            continue

        matrix = Q_table[:num_states-1, a].reshape((len(POSITIONS), len(POSITIONS), len(HIT_TYPES)))
        
        h, w, c = np.nonzero(matrix)
        X = np.concatenate((h[:, np.newaxis], w[:, np.newaxis], c[:, np.newaxis]), axis=1).astype(float)
        y = matrix[np.nonzero(matrix)].reshape((-1, 1))

        h_fill, w_fill, c_fill = np.nonzero(matrix == 0)
        X_fill = np.concatenate((h_fill[:, np.newaxis], w_fill[:, np.newaxis], c_fill[:, np.newaxis]), axis=1).astype(float)

        # dists = scipy.spatial.distance.cdist(X_fill, X)
        # weights = scipy.special.softmax(dists, axis=1)
        # y_fill = weights @ y
        # w = np.argmax(dists, axis=1)
        # y_fill = y[w]

        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit(X, y)
        y_fill = knn.predict(X_fill)

    
        matrix = matrix.flatten()
        indices = h_fill[:, np.newaxis] * len(POSITIONS) * len(HIT_TYPES) + w_fill[:, np.newaxis] * len(HIT_TYPES) + c_fill[:, np.newaxis]

        np.put(matrix, indices, y_fill)

        Q[:num_states-1, a] = matrix

    return Q


if __name__ == '__main__':
    # (player's position, ball position, opponent hit type) + one state for end of game
    num_states = len(POSITIONS) * len(POSITIONS) * len(HIT_TYPES) + 1

    # (player's new position, ball hit type)
    num_actions = len(POSITIONS) * len(HIT_TYPES)

    print(num_states, num_actions)

    Q_table = q_learning(num_states, num_actions)

    Q_table = approx_q_values(Q_table, num_states, num_actions)
    
    policy = get_best_actions(Q_table)
    policy = policy.tolist()

    with open("q_learning.policy", 'w') as f:
        for i in range(num_states):
            f.write(f'{policy[i]}\n')

    state_to_action = dict()
    with open("q_learning.policy", 'r') as f:
        idx = 0
        for line in f.readlines():
            val = int(line)
            player_pos_idx = idx // (len(POSITIONS) * len(HIT_TYPES))
            ball_pos_idx = (idx % (len(POSITIONS) * len(HIT_TYPES))) // len(HIT_TYPES)
            hit_type_idx = (idx % (len(POSITIONS) * len(HIT_TYPES))) % len(HIT_TYPES)

            receive_pos = val // len(HIT_TYPES)
            receive_type = val % len(HIT_TYPES)

            state_to_action[(POSITIONS[player_pos_idx], POSITIONS[ball_pos_idx], HIT_TYPES[hit_type_idx])] = (POSITIONS[receive_pos], HIT_TYPES[receive_type])

            idx += 1
            if idx == num_states-1:
                break

    with open('q_learning.pkl', 'wb') as f:
        pickle.dump(state_to_action, f)



