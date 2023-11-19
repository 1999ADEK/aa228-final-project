import sys
from typing import Dict

import numpy as np

sys.path.append("../")
from .base_player import BasePlayer
from utils import Action, State, HIT_TYPES, POSITIONS

class DefaultPlayer(BasePlayer):
    """A class used to represent the default tennis player."""

    def __init__(self,
        player_id: int,
        first_serve_success_rate: float,
        second_serve_success_rate: float,
        position_lookup_table: Dict[str, Dict[str, float]],
    ):
        super().__init__(
            player_id,
            first_serve_success_rate,
            second_serve_success_rate,
        )
        self.position_lookup_table = position_lookup_table

    def update_state(self, current_state: State, action: Action) -> State:
        player_positions = current_state.player_positions
        player_movement = action.player_movement
        # Stochastic outcome of player_movement
        player_positions[self.player_id] = np.random.choice(
            POSITIONS,
            p=list(self.position_lookup_table[player_movement].values())
        )

        # TODO(@rqwang): Determine ball_position based on the existing dataset.
        ball_position = np.random.choice(POSITIONS)

        next_state = State(
            player_positions=player_positions,
            hitter_hit_type=action.hit_type,
            ball_position=ball_position,
        )
        return next_state

    def choose_action(self, state: State) -> Action:
        # TODO: Maybe pick the action based on our existing dataset as well?
        hit_type = np.random.choice(HIT_TYPES)
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
            player_movement = np.random.choice(POSITIONS)
        action = Action(hit_type=hit_type, player_movement=player_movement)
        return action