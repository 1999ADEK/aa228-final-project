import sys
from typing import Dict, Tuple

import numpy as np
import pickle

sys.path.append("../")
from .base_player import BasePlayer
from utils import (
    Action, State, RECEIVE_HIT_TYPES, COURT_BBOX, POS_CENTER_SERVICE_LINE, COURT_WIDTH, POS_NET
)


def wrap_angle(theta):
    """Wraps angle to 0 <= theta < 360 degree."""
    if theta >= 360:
        theta -= 360
    elif theta < 0:
        theta += 360
    return theta

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

    def get_serve_direction(self, serve_position) -> float:
        """Determines the ball direction of a serve."""

        # Ball directions will be different when served at BL and BR,
        # so we need to use different distributions in the lookup table
        dir_change = np.random.normal(
            *self.dir_change_lookup_table[f"forehand_serve_{serve_position}"]
        )
        return dir_change
    
    def update_state(self, current_state: State, action: Action) -> State:
        """Updates the state based on current state and action."""

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

        # ======== Determine ball_position ======== #
        hitter_hit_type = current_state.hitter_hit_type
        ball_position = current_state.ball_position
        ball_direction = current_state.ball_direction

        distance = np.random.normal(
            *self.distance_lookup_table[hitter_hit_type]
        )
        # Apply the displacement, and flip the coordinate
        theta = np.deg2rad(ball_direction)
        displacement = distance * np.array([np.cos(theta), np.sin(theta)])
        ball_position = COURT_BBOX - (ball_position + displacement)

        # ======== Determine ball_position ======== #
        dir_change = np.random.normal(
            *self.dir_change_lookup_table[action.hit_type]
        )
        # Flip the direction coordinate
        ball_direction = wrap_angle(ball_direction + 180)
        
        # Apply the change of direction
        # When the incoming ball is in this direction: ↘
        if ball_direction > 270 or ball_direction == 0:
            ball_direction = ball_direction + dir_change
        # When the incoming ball is in this direction: ↙
        else:
            ball_direction = ball_direction - dir_change
        ball_direction = wrap_angle(ball_direction)
        
        next_state = State(
            player_positions=player_positions,
            hitter_hit_type=action.hit_type,
            ball_position=ball_position,
            ball_direction=ball_direction,
        )
        return next_state

    def choose_action(self, state: State) -> Action:
        """Chooses an action based on the current state."""
        with open('model/ordinal_encoder.pkl', 'rb') as encoder_file:
            loaded_ordinal_encoder = pickle.load(encoder_file)

        # Load the encoder from the file using pickle
        with open('model/label_encoder.pkl', 'rb') as encoder_file:
            loaded_label_encoder = pickle.load(encoder_file)

        # Load the kNN model from the file using pickle
        with open('model/knn_model.pkl', 'rb') as model_file:
            loaded_knn_model = pickle.load(model_file)
        with open('model/knn_model_action_x.pkl', 'rb') as model_file:
            loaded_knn_model_action_x = pickle.load(model_file)
        with open('model/knn_model_action_y.pkl', 'rb') as model_file:
            loaded_knn_model_action_y = pickle.load(model_file)

        # Flip the ball position to the other side of the court
        input_ball_x = COURT_WIDTH - state.ball_position[0]
        input_ball_y = POS_NET + (POS_NET - state.ball_position[1])

        # Construct input based on the flipped ball position
        state_input = np.array([input_ball_x,input_ball_y]  + list(state.player_positions[self.player_id])+ list(loaded_ordinal_encoder.transform([[state.hitter_hit_type]])[0])).reshape(1, -1)

        hit_type = loaded_label_encoder.inverse_transform(loaded_knn_model.predict(state_input))[0]
        # Not sure if we want to change this
        if state.hitter_hit_type == "forehand_serve":
            player_movement = np.copy(state.player_positions[self.player_id])
        else:
            # Select player position based on the output from the kNN model
            player_x = loaded_knn_model_action_x.predict(state_input)[0]
            player_y = loaded_knn_model_action_y.predict(state_input)[0]

            player_movement = np.array([
                player_x,
                player_y,
            ])
        action = Action(hit_type=hit_type, player_movement=player_movement)
        return action