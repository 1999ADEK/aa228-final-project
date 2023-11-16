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

speed_lookup_table = {
    # (stroke, ball_hit):    (avg, std)
    ("forehand", "serve"):   (30.87, 3.85),
    ("forehand", "slice"):   (14.20, 2.04),
    ("forehand", "topspin"): (21.86, 2.92),
    ("forehand", "return"):  (20.26, 4.96),
    ("backhand", "slice"):   (13.81, 3.41),
    ("backhand", "topspin"): (21.42, 3.60),
    ("backhand", "return"):  (19.44, 3.58),
    ("backhand", "volley"):  (6.88,  1.60),
    ("backhand", "stop"):    (7.54,  0.87),
}

action_success_rate = {
    # (stroke, ball_hit):    prob
    ("forehand", "serve"):   0.7,
    ("forehand", "slice"):   0.7,
    ("forehand", "topspin"): 0.7,
    ("forehand", "return"):  0.7,
    ("backhand", "slice"):   0.7,
    ("backhand", "topspin"): 0.7,
    ("backhand", "return"):  0.7,
    ("backhand", "volley"):  0.7,
    ("backhand", "stop"):    0.7,
}

position_lookup_table = {
    "TL": {"TL": 0.28, "TR": 0.12, "BL": 0.36, "BR": 0.24},
    "TR": {"TL": 0.12, "TR": 0.28, "BL": 0.24, "BR": 0.36},
    "BL": {"TL": 0.21, "TR": 0.09, "BL": 0.49, "BR": 0.21},
    "BR": {"TL": 0.09, "TR": 0.21, "BL": 0.21, "BR": 0.49},
}

action_probs = {
    # (stroke, ball_hit):    prob
    ("forehand", "serve"):   0.0, # Set to zero since it only happens at the beginning
    ("forehand", "slice"):   0.125,
    ("forehand", "topspin"): 0.125,
    ("forehand", "return"):  0.125,
    ("backhand", "slice"):   0.125,
    ("backhand", "topspin"): 0.125,
    ("backhand", "return"):  0.125,
    ("backhand", "volley"):  0.125,
    ("backhand", "stop"):    0.125,
}

action_sets = list(action_probs.keys())

class Player(object):
    def __init__(self):
        self.speed_lookup_table = speed_lookup_table
        self.position_lookup_table = position_lookup_table
        self.action_probs = action_probs
        self.action_success_rate = action_success_rate

    def update_state(self, state, action):
        """
        - Each state is represented as a tuple (position, speed),
            which refers to the status of the ball.
            (speed == 0) means the player misses the ball or the ball goes
            outside the court. In this case, the opponent get one point.
        - Each action is represented as a tuple (stroke, shot).
        """

        # Unpack current state and action
        position, speed = state
        stroke, shot = action

        # Determine the speed of the next state
        if np.random.uniform() > self.action_success_rate.get(action, 0):
            # Attempt fails
            next_speed = 0
        else:
            mean, std = self.speed_lookup_table.get(action, (0, 0))
            next_speed = np.random.normal(mean, std)
        
        # Determine the position of the next state
        next_state = np.random.choice(
            POSITIONS,
            p=list(self.position_lookup_table[position].values())
        )

        return (next_state, next_speed)

    def choose_action(self, state=None):
        action_idx = np.random.choice(
            len(action_sets),
            p=list(self.action_probs.values())
        )
        return action_sets[action_idx]


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
        self.score_board = [[] for _ in range(2)]

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
        # Simulate untill the ball dies
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
    
    simulator.export_history("tmp.csv")