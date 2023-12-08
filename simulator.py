from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import (
    Action,
    State,
    POS_CENTER_SERVICE_LINE,
    POS_BASELINE,
    DISTANCE_LOOKUP_TABLE_DJOKOVIC,
    DIR_CHANGE_LOOKUP_TABLE_DJOKOVIC,
)
from players import DefaultPlayer#, QLearningPlayer, OnlineQLearningPlayer
import copy

import warnings
warnings.filterwarnings("ignore")


def check_hit_success(player_position: np.ndarray, ball_position: np.ndarray):
    """Checks whether the player can hit the ball based on the positions.
        The assumption here is that a player can hit a ball that's within
        1-meter range.

        Parameters
        ----------
        player_position: np.ndarray of shape (2,)
            The position of the player.
        ball_position: np.ndarray of shape (2,)
            The position of the tennis ball.
    """
    distance = np.linalg.norm(player_position - ball_position)
    return distance < 1.0


class TennisSimulator(object):
    """A class that simulates a tennis match."""


    def __init__(self, players: Optional[List] = None):
        """Initializes the simulator.

        Parameters
        ----------
        players: Optional[List]
            The two players in the match.
        """
        if players is not None:
            assert len(players) == 2
            self.players = players
        else:
            self.players = [
                DefaultPlayer(
                    player_id=0,
                    first_serve_success_rate=0.6,
                    second_serve_success_rate=0.8,
                    distance_lookup_table=DISTANCE_LOOKUP_TABLE_DJOKOVIC,
                    dir_change_lookup_table=DIR_CHANGE_LOOKUP_TABLE_DJOKOVIC,
                ),
                DefaultPlayer(
                    player_id=1,
                    first_serve_success_rate=0.6,
                    second_serve_success_rate=0.8,
                    distance_lookup_table=DISTANCE_LOOKUP_TABLE_DJOKOVIC,
                    dir_change_lookup_table=DIR_CHANGE_LOOKUP_TABLE_DJOKOVIC,
                ),
            ]

        self.history = defaultdict(list)
        self.score_board = [[] for _ in range(2)]
        self.reward = [0, 0]

    def reset(self):
        """Resets the score and reward before a new match."""

        self.history = defaultdict(list)
        self.score_board = [[] for _ in range(2)]
        self.reward = [0, 0]

    def update_history(
        self,
        player_id: int,
        state: State,
        action: Action,
    ) -> None:
        """Updates the log.

        Parameters
        ----------
        player_id: int
            The id of the current receiver. Should be either 0 or 1.
        state: State
            The state of the tennis court.
        action: Action
            The action the receiver takes.
        """
        
        self.history["player_0_pos"].append(state.player_positions[0])
        self.history["player_1_pos"].append(state.player_positions[1])
        self.history["hitter"].append(1 - player_id)
        self.history["receiver"].append(player_id)
        self.history["ball_pos"].append(state.ball_position)
        self.history["ball_dir"].append(state.ball_direction)
        self.history["hitter_hit_type"].append(state.hitter_hit_type)
        self.history["receiver_hit_type"].append(action.hit_type)
        self.history["receiver_movement"].append(action.player_movement)
        self.history["player_0_reward"].append(self.reward[0])
        self.history["player_1_reward"].append(self.reward[1])

    def simulate_point(
        self,
        serve_id: int,
        serve_position: str,
    ) -> int:
        """Simulates a point.

        Parameters
        ----------
        serve_id: int
            The id of the server. Should be either 0 or 1.
        serve_position: str
            The position at which the server serves the ball.
            Should be "BR" or "BL".

        Returns
        -------
        winner_id: int
            The winner of the point.
        """

        server = self.players[serve_id]
        # First serve
        if not server.check_serve_success(is_first_serve=True):
            # Second serve
            if not server.check_serve_success(is_first_serve=False):
                # Double Fault, the opponent wins the point
                return 1 - serve_id

        # Initial state
        ball_direction = server.get_serve_direction(serve_position)
        # The serve_position is hard coded for now
        serve_position = np.array([
            POS_CENTER_SERVICE_LINE + (1.0 if serve_position == "BR" else -1.0),
            POS_BASELINE - 0.1,
        ])
        state = State(
            player_positions=[serve_position, serve_position],
            hitter_hit_type="forehand_serve",
            ball_position=serve_position,
            ball_direction=ball_direction,
        )

        player_id = serve_id
        # Simulate until the ball dies
        while check_hit_success(
            state.player_positions[player_id],
            state.ball_position,
        ):
            player_id = 1 - player_id
            # current_state = copy.deepcopy(state)

            action = self.players[player_id].choose_action(state)
            self.update_history(player_id, state, action)
            state = self.players[player_id].update_state(state, action)

            # collect s, a, r for the current player
            # s (player's position, ball position, opponent hit type)
            # a (player's new position, ball hit type)
            # self.players[player_id].update_policy(current_state, copy.deepcopy(action), self.reward[player_id], None)

        # When a player fails to get to the ball position,
        # The opponent wins the point
        winner_id = 1 - player_id
        return winner_id

    def simulate_game(self, serve_id: int) -> Tuple[int, Tuple[int, int]]:
        """Simulates a game.

        Parameters
        ----------
        serve_id: int
            The id of the server. Should be either 0 or 1.

        Returns
        -------
        winner_id: int
            The winner of the game.
        scores: Tuple[int, int]
            How many points each player wins in the game.
        """

        scores = [0, 0]
        # Player who wins 4 points and has 2-point lead wins the game
        while max(scores) < 4 or abs(scores[0] - scores[1]) < 2:
            # Change serve_position every two points
            if sum(scores) % 2 == 0:
                serve_position = "BR"
            else:
                serve_position = "BL"
            winner_id = self.simulate_point(serve_id, serve_position)
            scores[winner_id] += 1
            self.reward[winner_id] += 10
            self.reward[1 - winner_id] -= 10

        winner_id = 0 if scores[0] > scores[1] else 1
        return winner_id, scores  

    def simulate_tiebreak(self, serve_id: int) -> Tuple[int, Tuple[int, int]]:
        """Simulates a tiebreak.

        Parameters
        ----------
        serve_id: int
            The id of the server. Should be either 0 or 1.

        Returns
        -------
        winner_id: int
            The winner of the tiebreak.
        scores: Tuple[int, int]
            How many points each player wins in the tiebreak.
        """

        scores = [0, 0]
        # Player who wins 7 points and has 2-point lead wins the tiebreak
        while max(scores) < 7 or abs(scores[0] - scores[1]) < 2:
            if sum(scores) % 2 == 0:
                serve_position = "BR"
            else:
                serve_position = "BL"
            winner_id = self.simulate_point(
                abs(serve_id - (sum(scores) // 2) % 2), # Rule for serve_id is different in tiebreak
                serve_position,
            )
            scores[winner_id] += 1

        winner_id = 0 if scores[0] > scores[1] else 1
        return winner_id, scores

    def simulate_set(self, serve_id: int) -> Tuple[int, Tuple[int, int]]:
        """Simulates a set.

        Parameters
        ----------
        serve_id: int
            The id of the server. Should be either 0 or 1.

        Returns
        -------
        winner_id: int
            The winner of the set.
        scores: Tuple[int, int]
            How many games each player wins in the set.
        """
        scores = [0, 0]
        # Player who wins 6 games wins the set
        while max(scores) < 6 or sum(scores) == 11:
            winner_id, statistics = self.simulate_game(serve_id)
            scores[winner_id] += 1
            # The other player serves the next game
            serve_id = 1 - serve_id

        # Tiebreak
        if scores[0] == scores[1] == 6:
            winner_id, statistics = self.simulate_tiebreak(serve_id)
            scores[winner_id] += 1
            # Store the tiebreak results in scores as well
            scores[0] = (scores[0], statistics[0])
            scores[1] = (scores[1], statistics[1])
        
        winner_id = 0 if scores[0] > scores[1] else 1
        return winner_id, scores

    def simulate_match(self) -> Tuple[int, Tuple[int, int]]:
        """Simulates a match.

        Returns
        -------
        winner_id: int
            The winner of the match.
        scores: Tuple[int, int]
            How many sets each player wins in the match.
        """
        scores = [0, 0]
        serve_id = 0
        # Player who wins two sets wins the match
        while max(scores) < 2:
            winner_id, statistics = self.simulate_set(serve_id)
            scores[winner_id] += 1
            # Update the score board
            self.score_board[0].append(statistics[0])
            self.score_board[1].append(statistics[1])
            # The other player serves the first game in the next set
            serve_id = 1 - serve_id

        winner_id = 0 if scores[0] > scores[1] else 1
        return winner_id, scores

    def export_history(self, filename: str = "history.csv") -> None:
        """Exports the log as a CSV file.

        Parameters
        ----------
        filename: str
            The filename of the exported log.
        """
        df_history = pd.DataFrame(self.history)
        df_history.to_csv(filename)


if __name__ == "__main__":
    simulator = TennisSimulator()
    winner, scores = simulator.simulate_match()

    print(f"Player {winner} won the match!")
    print(f"===== Score Board =====")
    print(*simulator.score_board[0], sep=' | ')
    print(*simulator.score_board[1], sep=' | ')
    print(f"=======================")
    
    simulator.export_history("tmp.csv")