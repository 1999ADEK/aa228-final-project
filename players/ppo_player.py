import numpy as np

from typing import Dict, Tuple
from .base_player import BasePlayer
from utils import State, Action, COURT_BBOX, HIT_TYPES, POS_CENTER_SERVICE_LINE
from .default_player import wrap_angle
from stable_baselines3 import PPO

def check_forehand_serve(player_x):
    if player_x > POS_CENTER_SERVICE_LINE:
        lookup_hit_type = "forehand_serve_BR"
    else:
        lookup_hit_type = "forehand_serve_BL"
    return lookup_hit_type

def convert_type_to_encoding(hit_type):
    # Create a one-hot vector based on the hit type
    hit_type_idx = HIT_TYPES.index(hit_type)
    hitter_list = [0] * len(HIT_TYPES)
    hitter_list[hit_type_idx] = 1
    return hitter_list

def convert_state_to_output(state, player_id):
    # Convert state to a vector
    player_pos = state.player_positions
    ball_dir = state.ball_direction
    hit_type = state.hitter_hit_type
    return np.array([player_pos[player_id][0], player_pos[player_id][1], player_pos[1-player_id][0], player_pos[1-player_id][1], ball_dir] + convert_type_to_encoding(hit_type))

class PPOPlayer(BasePlayer):
    """A class used to represent the default tennis player."""

    def __init__(self,
        player_id: int,
        first_serve_success_rate: float,
        second_serve_success_rate: float,
        distance_lookup_table: Dict[str, Tuple[float, float]],
        dir_change_lookup_table: Dict[str, Tuple[float, float]],
        ppo_model_path: str,
    ):
        super().__init__(
            player_id,
            first_serve_success_rate,
            second_serve_success_rate,
        )
        self.dir_change_lookup_table = dir_change_lookup_table
        self.distance_lookup_table = distance_lookup_table
        if ppo_model_path != "":
            self.model = PPO.load(ppo_model_path)

    def get_serve_direction(self, serve_position) -> float:
        """Determines the ball direction of a serve."""

        # Ball directions will be different when served at BL and BR,
        # so we need to use different distributions in the lookup table
        dir_change = np.random.normal(
            *self.dir_change_lookup_table[f"forehand_serve_{serve_position}"]
        )
        return dir_change

    def update_state(self, current_state: State, action: Action) -> State:
        """Updates the state of the player based on the action."""
        import pdb; pdb.set_trace()
        player_positions = current_state.player_positions
        player_movement = action.player_movement
        # Stochastic outcome of player_movement
        player_positions[self.player_id] = np.random.normal(
            loc=player_movement,
            # Assign a larger std if the targeted position is far away
            # from the current position
            scale=np.abs(
                player_movement - player_positions[self.player_id]
            ) / 10,
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

        # ======== Determine ball_direction ======== #
        cur_hit_type = action.hit_type
        if cur_hit_type == "forehand_serve":
            lookup_hit_type = check_forehand_serve(player_positions[self.player_id][0])
        else:
            lookup_hit_type = cur_hit_type
        dir_change = np.random.normal(
            *self.dir_change_lookup_table[lookup_hit_type]
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
        cur_state = convert_state_to_output(state, self.player_id)
        action, _ = self.model.predict(cur_state, deterministic=True)
        hit_type = HIT_TYPES[np.argmax(action[2:])]
        player_movement = action[:2]
        return Action(hit_type, player_movement)