import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from utils import HIT_TYPES, POSITIONS, State, Action
from q_learning_utils import get_state_index, get_action_index, NUM_ACTIONS, NUM_STATES
from simulator import TennisSimulator

import pickle

import warnings
warnings.filterwarnings("ignore")


def save_policy(policy):
    policy = policy.tolist()

    with open("online_q_learning.policy", 'w') as f:
        for i in range(NUM_STATES):
            f.write(f'{policy[i]}\n')

    state_to_action = dict()
    with open("online_q_learning.policy", 'r') as f:
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
            if idx == NUM_STATES-1:
                break

    with open('online_q_learning.pkl', 'wb') as f:
        pickle.dump(state_to_action, f)



if __name__ == "__main__":
    simulator = TennisSimulator()

    ITERS = 1000
    wins = [0]
    best = 0

    for i in range(ITERS):
        print("Iter ", i)
        # train
        print(simulator.players[1].is_train)
        winner, scores = simulator.simulate_match()

        print(f"Player {winner} won the match!")
        print(f"===== Score Board =====")
        print(*simulator.score_board[0], sep=' | ')
        print(*simulator.score_board[1], sep=' | ')
        print(f"=======================")

        simulator.reset()

        # test
        if (i+1) % 50 == 0:
            simulator.players[1].is_train = False

            count_win = 0
            for _ in range(20):
                winner, scores = simulator.simulate_match()
                # print(f"Player {winner} won the match!")
                # print(f"===== Score Board =====")
                # print(*simulator.score_board[0], sep=' | ')
                # print(*simulator.score_board[1], sep=' | ')
                # print(f"=======================")

                if winner == 1:
                    count_win += 1
                simulator.reset()
            
            win_ratio = count_win * 1.0 / 20
            wins.append(win_ratio)

            # resume policy updates
            simulator.players[1].is_train = True

            if win_ratio > best:
                best = win_ratio
                policy = simulator.players[1].get_best_policy()
                save_policy(policy)
    
    x = np.arange(0, ITERS+1, 50)
    plt.plot(x, wins)
    plt.show()
    




