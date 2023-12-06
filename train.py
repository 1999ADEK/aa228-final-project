import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


from utils import HIT_TYPES, POSITIONS, State, Action
from q_learning_utils import NUM_ACTIONS, NUM_STATES
from simulator import TennisSimulator, position_lookup_table
from players import DefaultPlayer, QLearningPlayer, OnlineQLearningPlayer

import pickle

import warnings
warnings.filterwarnings("ignore")


# import random
# random.seed(42)
# np.random.seed(42)


def save_policy(policy, prefix="online_q_learning"):
    policy = policy.tolist()

    with open(f"{prefix}.policy", 'w') as f:
        for i in range(NUM_STATES):
            f.write(f'{policy[i]}\n')

    state_to_action = dict()
    with open(f"{prefix}.policy", 'r') as f:
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

    with open(f"{prefix}.pkl", 'wb') as f:
        pickle.dump(state_to_action, f)



def train_single_agent():
    players = [
        DefaultPlayer(
            player_id=0,
            first_serve_success_rate=0.6,
            second_serve_success_rate=0.8,
            position_lookup_table=position_lookup_table,
        ),
        OnlineQLearningPlayer(
            player_id=1,
            first_serve_success_rate=0.6,
            second_serve_success_rate=0.8,
            position_lookup_table=position_lookup_table,
            q_learning_policy="",
            is_train=True,
        )
    ]
    simulator = TennisSimulator(players=players)

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


def train_multi_agents(args):
    players = [
        OnlineQLearningPlayer(
            player_id=0,
            first_serve_success_rate=0.6,
            second_serve_success_rate=0.8,
            position_lookup_table=position_lookup_table,
            q_learning_policy="",
            is_train=True,
        ),
        OnlineQLearningPlayer(
            player_id=1,
            first_serve_success_rate=0.6,
            second_serve_success_rate=0.8,
            position_lookup_table=position_lookup_table,
            q_learning_policy="",
            is_train=True,
        )
    ]

    simulator = TennisSimulator(players=players)

    player_0_wins = [0.5]
    player_1_wins = [0.5]

    for i in range(args.iters):
        print("Iter ", i)
        # train
        winner, scores = simulator.simulate_match()

        print(f"Player {winner} won the match!")
        print(f"===== Score Board =====")
        print(*simulator.score_board[0], sep=' | ')
        print(*simulator.score_board[1], sep=' | ')
        print(f"=======================")

        simulator.reset()

        # test
        if (i+1) % args.log_freq == 0:
            for p in range(2):
                simulator.players[p].is_train = False

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
            player_1_wins.append(win_ratio)
            player_0_wins.append(1 - win_ratio)

            # resume policy updates
            for p in range(2):
                simulator.players[p].is_train = True

    for p in range(2):
        policy = simulator.players[p].get_best_policy()
        save_policy(policy, prefix=f"q_learning_player_{p}")
    
    x = np.arange(0, args.iters+1, args.log_freq)
    plt.plot(x, player_0_wins, label="Player 0")
    plt.plot(x, player_1_wins, label="Player 1")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-multi', action='store_true', help="training multi agent vs single agent")
    parser.add_argument('-iters', default=1000)
    parser.add_argument('-log_freq', default=50, help="validation and logging frequency")
    args = parser.parse_args()

    if not args.multi:
        train_single_agent()
    else:
        train_multi_agents(args)
    