import sys
from typing import Dict, Tuple

import numpy as np
import pickle

sys.path.append("../")
from .base_player import BasePlayer
from utils import (
    Action, State, RECEIVE_HIT_TYPES, COURT_BBOX, POS_CENTER_SERVICE_LINE
)
from .action_chooser import position_table

class DefaultPlayer(BasePlayer):
    """A class used to represent the default tennis player."""

    def __init__(self,
        player_id: int,
        first_serve_success_rate: float,
        second_serve_success_rate: float,
        distance_lookup_table: Dict[str, Tuple[float, float]],
        dir_change_lookup_table: Dict[str, Tuple[float, float]],
    ):
        super().__init__(
            player_id,
            first_serve_success_rate,
            second_serve_success_rate,
        )
        self.distance_lookup_table = distance_lookup_table
        self.dir_change_lookup_table = dir_change_lookup_table

    def update_state(self, current_state: State, action: Action) -> State:
        # ======== Determine player_positions ======== #
        player_positions = current_state.player_positions
        player_movement = action.player_movement
        # Stochastic outcome of player_movement
        player_positions[self.player_id] = np.random.normal(
            loc=player_movement,
            # Assign a larger std if the targeted position is far away
            # from the current position
            scale=np.abs(
                player_movement - player_positions[self.player_id]
            ) / 3,
        )

        # ======== Determine ball_position/direction ======== #
        hitter_hit_type = current_state.hitter_hit_type
        ball_position = current_state.ball_position
        ball_direction = current_state.ball_direction

        distance = np.random.normal(
            *self.distance_lookup_table[hitter_hit_type]
        )
        # Ball directions will be different when served at BL and BR,
        # so we need to use different distributions in the lookup table
        if hitter_hit_type == "forehand_serve":
            if ball_position[0] < POS_CENTER_SERVICE_LINE:
                hitter_hit_type += "_BL"
            else:
                hitter_hit_type += "_BR"
        dir_change = np.random.normal(
            *self.dir_change_lookup_table[hitter_hit_type]
        )
        # Apply the change of direction
        # When the incoming ball is in this direction: ↘
        if ball_direction > 270 or ball_direction == 0:
            ball_direction = ball_direction + dir_change
        # When the incoming ball is in this direction: ↙
        else:
            ball_direction = ball_direction - dir_change
        # Apply the displacement, and flip the coordinate
        theta = np.deg2rad(ball_direction)
        displacement = distance * np.array([np.cos(theta), np.sin(theta)])
        ball_position = COURT_BBOX - (ball_position + displacement)
        # Flip the direction coordinate and wrap the angle
        ball_direction += 180
        if ball_direction >= 360:
            ball_direction -= 360
        elif ball_direction < 0:
            ball_direction += 360
        
        next_state = State(
            player_positions=player_positions,
            hitter_hit_type=action.hit_type,
            ball_position=ball_position,
            ball_direction=ball_direction,
        )
        return next_state

    def choose_action(self, state: State) -> Action:
        """Chooses an action based on the current state."""
        # Randomly pick a hit type
        hit_type = np.random.choice(RECEIVE_HIT_TYPES)
        if state.hitter_hit_type == "forehand_serve":
            player_movement = np.copy(state.player_positions[self.player_id])
        else:
            # Brainlessly mirror the current position according to
            # the center service line
            player_movement = np.array([
                COURT_BBOX[0] - state.player_positions[self.player_id][0],
                state.player_positions[self.player_id][1],
            ])
        action = Action(hit_type=hit_type, player_movement=player_movement)
        return action