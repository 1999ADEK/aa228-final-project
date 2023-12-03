import argparse

import matplotlib.pyplot as plt
from tqdm import tqdm

from players import DefaultPlayer, QLearningPlayer
from simulator import TennisSimulator


# TODO(@tchang): Definitely move this to another place. (Maybe utils.py?)
position_lookup_table = {
    "TL": {"TL": 0.93, "TR": 0.03, "BL": 0.03, "BR": 0.01},
    "TR": {"TL": 0.03, "TR": 0.93, "BL": 0.01, "BR": 0.03},
    "BL": {"TL": 0.03, "TR": 0.01, "BL": 0.93, "BR": 0.03},
    "BR": {"TL": 0.01, "TR": 0.03, "BL": 0.03, "BR": 0.93},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tennis simulation and compute metrics."
    )

    parser.add_argument(
        "-p0",
        "--player_0",
        type=str,
        default="default",
        choices=["default", "q_learning"],
        help="Specify the first player.",
    )
    parser.add_argument(
        "-p1",
        "--player_1",
        type=str,
        default="default",
        choices=["default", "q_learning"],
        help="Specify the second player.",
    )
    parser.add_argument(
        "-n",
        "--num_matches",
        type=int,
        default=100,
        help="Number of matches to simulate.",
    )

    args = parser.parse_args()
    return args


def main(args):
    # Initialize the players
    players = []
    for player_id, player_type in enumerate([args.player_0, args.player_1]):
        if player_type == "default":
            players.append(
                DefaultPlayer(
                    player_id=player_id,
                    first_serve_success_rate=0.6,
                    second_serve_success_rate=0.8,
                    position_lookup_table=position_lookup_table,
                )
            )
        elif player_type == "q_learning":
            players.append(
                QLearningPlayer(
                    player_id=player_id,
                    first_serve_success_rate=0.6,
                    second_serve_success_rate=0.8,
                    position_lookup_table=position_lookup_table,
                    q_learning_policy="model/q_learning.pkl",
                )
            )
    # Initialize the simulator
    simulator = TennisSimulator(players=players)

    # Initialize the metrics
    winning_rates = [[] for _ in range(2)]
    mean_rewards = [[] for _ in range(2)]
    num_matches_won = [0, 0]
    total_rewards = [0, 0]

    print(f"Start simulating {args.num_matches} matches!")
    for match_idx in tqdm(range(args.num_matches)):
        # Simulate a match
        simulator.reset()
        winner_id, _ = simulator.simulate_match()
        num_matches_won[winner_id] += 1
        total_rewards[0] += simulator.reward[0]
        total_rewards[1] += simulator.reward[1]

        # Update metrics
        for player_id in range(2):
            winning_rates[player_id].append(
                num_matches_won[player_id] / (match_idx + 1)
            )
            mean_rewards[player_id].append(
                total_rewards[player_id] / (match_idx + 1)
            )
    print("Simulation done!")

    # Plot metrics
    print("Plotting the metrics.")

    plt.title("Winning Rate")
    plt.plot(winning_rates[0], label=f"p0: {args.player_0}")
    plt.plot(winning_rates[1], label=f"p1: {args.player_1}")
    plt.legend()
    plt.savefig("winning_rate.png")

    plt.clf()
    plt.title("Mean Reward")
    plt.plot(mean_rewards[0], label=f"p0: {args.player_0}")
    plt.plot(mean_rewards[1], label=f"p1: {args.player_1}")
    plt.legend()
    plt.savefig("mean_reward.png")
    print("Plots saved as winning_rate.png and mean_reward.png")

if __name__ == "__main__":
    args = parse_args()
    main(args)