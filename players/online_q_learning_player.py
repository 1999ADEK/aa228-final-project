import sys
from typing import Dict

import numpy as np
import pickle

sys.path.append("../")
from .base_player import BasePlayer
from utils import Action, State, HIT_TYPES, POSITIONS

class OnlineQLearningPlayer(BasePlayer):
    """A class used to represent the default tennis player."""

    def __init__(self,
        player_id: int,
        first_serve_success_rate: float,
        second_serve_success_rate: float,
        position_lookup_table: Dict[str, Dict[str, float]],
        q_learning_policy,
        is_train: bool,
    ):
        super().__init__(
            player_id,
            first_serve_success_rate,
            second_serve_success_rate,
        )

        self.position_lookup_table = position_lookup_table
        self.q_learning_policy = pickle.load(open(q_learning_policy, 'rb'))

    
    def update_policy(self, current_state, action, reward, next_state):
        # update policy only when training
        if self.is_train:
            pass


    def update_state(self, current_state: State, action: Action) -> State:
        player_positions = current_state.player_positions
        player_movement = action.player_movement
        # Stochastic outcome of player_movement
        player_positions[self.player_id] = np.random.choice(
            POSITIONS,
            p=list(self.position_lookup_table[player_movement].values())
        )

        # Determine ball_position
        hitter_hit_type = current_state.hitter_hit_type
        ball_position = current_state.ball_position
        # A serve can only be in a certain position
        if hitter_hit_type == "forehand_serve":
            ball_position = 'T' + ball_position[1]
        # A volley results in a slower ball
        if "volley" in hitter_hit_type:
            ball_position = 'T' + ball_position[1]
        # Stochastic outcome of return
        if "return" in hitter_hit_type:
            ball_position = ball_position[0] + np.random.choice(['L', 'R'])
        # Apply overall stochasticity
        if hitter_hit_type != "forehand_serve":
            ball_position = np.random.choice(
                POSITIONS,
                p=list(self.position_lookup_table[ball_position].values())
            )

        next_state = State(
            player_positions=player_positions,
            hitter_hit_type=action.hit_type,
            ball_position=ball_position,
        )
        return next_state

    def choose_action(self, state: State) -> Action:
        """Chooses an action based on the current state."""
        
        move, hit_type = self.q_learning_policy[(state.player_positions[self.player_id], state.ball_position, state.hitter_hit_type)]

        # When the ball to return is a serve, it goes to a specific location.
        # For example, if the server serves the ball at "BL", the ball has to
        # go "TL" at the other side of the court. So the only reasonable
        # movement to pick here for the player is "TL".
        if state.hitter_hit_type == "forehand_serve":
            if state.ball_position == "BL":
                player_movement = "TL"
            elif state.ball_position == "BR":
                player_movement = "TR"
            else:
                player_movement = move
        else:
            player_movement = move
        action = Action(hit_type=hit_type, player_movement=player_movement)
        return action
