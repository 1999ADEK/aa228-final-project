import random

import numpy as np
import pandas as pd


STROKE_TYPES = [
    'forehand',
    'backhand',
]
BALL_HIT_TYPES = [
    'serve',
    'slice',
    'topspin',
    'return',
    'volley',
    'stop',
    'smash',
    'lob',
]
POSITIONS = [
    "TL",
    "TR",
    "BL",
    "BR",
]
SPEED = [0, 1]


class Player(object):
    def __init__(self, transition=None, action_probs=None):
        self.transition = transition
        self.action_probs = action_probs

    def update_state(self, state, action):
        # =========== TODO =========== #
        position, speed = state
        stroke, shot = action
        return (random.choice(POSITIONS), random.choice(SPEED))
        # =========== TODO =========== #

    def choose_action(self, state=None):
        # =========== TODO =========== #
        return (random.choice(STROKE_TYPES), random.choice(BALL_HIT_TYPES))
        # =========== TODO =========== #

class TennisSimulator(object):
    def __init__(self):
        self.players = [Player() for _ in range(2)]
        self.history = {
            "player": [],
            "position": [],
            "speed": [],
            "stroke": [],
            "ball_hit": [],
        }
        self.score_board = [[] for i in range(2)]

    def update_history(self, player_id, state, action):
        self.history["player"].append(player_id)
        self.history["position"].append(state[0])
        self.history["speed"].append(state[1])
        self.history["stroke"].append(action[0])
        self.history["ball_hit"].append(action[1])

    def simulate_point(self, serve_id, serve_position):
        # First serve
        state = self.players[serve_id].update_state(
            (serve_position, 0),
            ("forehand", "serve"),
        )
        self.update_history(
            serve_id,
            (serve_position, 0),
            ("forehand", "serve"),
        )
        if state[1] == 0:
            # Second serve
            state = self.players[serve_id].update_state(
                (serve_position, 0),
                ("forehand", "serve"),
            )
            self.update_history(
                serve_id,
                (serve_position, 0),
                ("forehand", "serve"),
            )

        player_id = serve_id
        while state[1] != 0:
            player_id = 1 - player_id
            action = self.players[player_id].choose_action()
            self.update_history(player_id, state, action)
            state = self.players[player_id].update_state(state, action)

        return 1 - player_id

    def simulate_game(self, serve_id):
        scores = [0, 0]
        # Player who wins 4 points and has 2-point lead wins the game
        while max(scores) < 4 or abs(scores[0] - scores[1]) < 2:
            if sum(scores) % 2 == 0:
                serve_position = "BR"
            else:
                serve_position = "BL"
            winner_id = self.simulate_point(serve_id, serve_position)
            scores[winner_id] += 1

        winner_id = 0 if scores[0] > scores[1] else 1
        return winner_id, scores  

    def simulate_tiebreak(self, serve_id):
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

    def simulate_set(self, serve_id):
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

    def simulate_match(self):
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

    def export_history(self, filename="history.csv"):
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
    simulator.export_history("/content/tmp.csv")