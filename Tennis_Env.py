from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulator import TennisSimulator, check_hit_success
from players import DefaultPlayer, PPOPlayer
from utils import DISTANCE_LOOKUP_TABLE_DJOKOVIC, DIR_CHANGE_LOOKUP_TABLE_DJOKOVIC, POS_CENTER_SERVICE_LINE, POS_BASELINE, HIT_TYPES, COURT_BBOX, State, DISTANCE_LOOKUP_TABLE_NADAL, DIR_CHANGE_LOOKUP_TABLE_NADAL, POS_NET

from stable_baselines3.common.env_checker import check_env

def wrap_angle(theta):
    """Wraps angle to 0 <= theta < 360 degree."""
    if theta >= 360:
        theta -= 360
    elif theta < 0:
        theta += 360
    return theta

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

def check_forehand_serve(player_x):
    if player_x > POS_CENTER_SERVICE_LINE:
        lookup_hit_type = "forehand_serve_BR"
    else:
        lookup_hit_type = "forehand_serve_BL"

    return lookup_hit_type

class TennisEnv(gym.Env):

    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(TennisEnv, self).__init__()
        # (player x, player y, ball x, ball y, ball direction, ball hit type)
        self.observation_space = spaces.Box(low = np.array([-2, -3, -2, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0]), high = np.array([13, POS_NET, 13, POS_NET, 180, 1, 1, 1, 1, 1, 1, 1, 1]), dtype = np.float64)
        # (player x, player y, hit type * 8)
        self.action_space = spaces.Box(low = np.array([-2, -3, 0, 0, 0, 0, 0, 0, 0, 0]), high = np.array([13, POS_NET, 1, 1, 1, 1, 1, 1, 1, 1]), dtype = np.float64)
        self.opponent_id = 0
        self.player_id = 1

    def reset(self, seed=None, options=None):
        """
        called at the beginning of an episode, it returns an observation and a dictionary with additional info (defaults to an empty dict)
        """
        super().reset(seed=seed, options=options)
        # Initialize a new game
        players = []
        players.append(
                DefaultPlayer(
                    player_id=0,
                    first_serve_success_rate=0.6,
                    second_serve_success_rate=0.8,
                    distance_lookup_table=DISTANCE_LOOKUP_TABLE_DJOKOVIC,
                    dir_change_lookup_table=DIR_CHANGE_LOOKUP_TABLE_DJOKOVIC,
                )
            )
        players.append(
                PPOPlayer(
                    player_id=1,
                    first_serve_success_rate=0.6,
                    second_serve_success_rate=0.8,
                    distance_lookup_table=DISTANCE_LOOKUP_TABLE_NADAL,
                    dir_change_lookup_table=DIR_CHANGE_LOOKUP_TABLE_NADAL,
                    ppo_model_path="",
                )
            )
        self.simulator = TennisSimulator(players=players)
        self.players = players

        # Let the server be the default player
        server = self.players[self.opponent_id]
        serve_position = "BR"
        ball_direction = server.get_serve_direction(serve_position)
        serve_position = np.array([
            POS_CENTER_SERVICE_LINE + (1.0 if serve_position == "BR" else -1.0),
            POS_BASELINE - 0.1,
        ])
        
        state = np.array([serve_position, serve_position] + [ball_direction] + convert_type_to_encoding("forehand_serve"))

        state = State(
            player_positions=[serve_position, serve_position],
            hitter_hit_type="forehand_serve",
            ball_position=serve_position,
            ball_direction=ball_direction,
        )

        output = convert_state_to_output(state, self.player_id)
        self.last_state = state

        return output, {}

    def step(self, action: Any):
        """ 
        called to take an action with the environment, it returns the 1. next observation, 2. the immediate reward 3. whether new state is a terminal state (episode is finished), 4. whether the max number of timesteps is reached (episode is artificially finished), and additional information

        action: Action vector coming from player 1 
        """
        reward = 1
        debug = False
        done = False
        truncated = False
        # Convert action to a hit type
        hit_type = HIT_TYPES[np.argmax(action[2:])]
        # ======== Determine player_positions ======== #
        player_positions = self.last_state.player_positions
        player_movement = action[0:2]
        if debug:
            player_movement = [8.40567547,0.9273905]

        if hit_type == "forehand_serve":
            lookup_hit_type = check_forehand_serve(player_positions[self.player_id][0])
        else:
            lookup_hit_type = hit_type

        # Stochastic outcome of player_movement
        player_movement = np.random.normal(
            loc=player_movement,
            # Assign a larger std if the targeted position is far away
            # from the current position
            scale=np.abs(
                player_movement - player_positions[self.player_id]
            ) / 10,
        )

        player_positions[self.player_id] = [min(max(-2, player_movement[0]), 14), min(max(-3, player_movement[1]), POS_NET)]
        # ======== Determine ball_position ======== #
        hitter_hit_type = self.last_state.hitter_hit_type
        ball_position = self.last_state.ball_position
        ball_direction = self.last_state.ball_direction
        # Not modifing the distance since it's most likely 
        distance = np.random.normal(
            *self.players[self.player_id].distance_lookup_table[hitter_hit_type]
        )

        # Apply the displacement, and flip the coordinate
        theta = np.deg2rad(ball_direction)
        displacement = distance * np.array([np.cos(theta), np.sin(theta)])
        ball_position = COURT_BBOX - (ball_position + displacement)

        # ======== Determine ball_direction ======== #
        dir_change = np.random.normal(
            *self.players[self.player_id].dir_change_lookup_table[lookup_hit_type]
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

        # Check if the returning hit is successful
        player_hit_success = check_hit_success(player_position=player_positions[self.player_id], ball_position=ball_position
        )
        #print(f"Player hit success: {player_hit_success}")
        #print(f"Player hit type: {hit_type}")
        #print(f"Player hit position: {player_positions[self.player_id]}")
        #print(f"Ball position: {ball_position}")

        next_state = State(
            player_positions=player_positions,
            hitter_hit_type=hit_type,
            ball_position=ball_position,
            ball_direction=ball_direction,
        )

        if not player_hit_success:
            # The opponent wins the point
            reward = -10
            done = True

        # Get action from opponent
        opoonent_action = self.players[self.opponent_id].choose_action(next_state)
        # ======== Determine opponent_positions ======== #
        player_positions = next_state.player_positions
        opponent_movement = opoonent_action.player_movement
        opponent_hit_type = opoonent_action.hit_type

        # Stochastic outcome of player_movement
        opponent_movement = np.random.normal(
            loc=opponent_movement,
            # Assign a larger std if the targeted position is far away
            # from the current position
            scale=np.abs(
                opponent_movement - player_positions[self.opponent_id]
            ) / 10,
        )

        player_positions[self.opponent_id] = [min(max(-2, opponent_movement[0]), 14), min(max(-3, opponent_movement[1]), POS_NET)]

        # ======== Determine ball_position ======== #
        player_hit_type = next_state.hitter_hit_type
        player_ball_position = next_state.ball_position
        player_ball_direction = next_state.ball_direction

        player_distance = np.random.normal(
            *self.players[self.opponent_id].distance_lookup_table[player_hit_type]
        )

        # Apply the displacement, and flip the coordinate
        player_theta = np.deg2rad(player_ball_direction)
        player_displacement = player_distance * np.array([np.cos(player_theta), np.sin(player_theta)])
        player_ball_position = COURT_BBOX - (player_ball_position + player_displacement)

        # ======== Determine ball_direction ======== #
        if opponent_hit_type == "forehand_serve":
            opponent_lookup_hit_type = check_forehand_serve(player_positions[self.opponent_id][0])
        else:
            opponent_lookup_hit_type = opponent_hit_type
        player_dir_change = np.random.normal(
            *self.players[self.opponent_id].dir_change_lookup_table[opponent_lookup_hit_type]
        )
        # Flip the direction coordinate
        player_ball_direction = wrap_angle(player_ball_direction + 180)
        
        # Apply the change of direction
        # When the incoming ball is in this direction: ↘
        if player_ball_direction > 270 or player_ball_direction == 0:
            player_ball_direction = player_ball_direction + player_dir_change
        # When the incoming ball is in this direction: ↙
        else:
            player_ball_direction = player_ball_direction - player_dir_change
        player_ball_direction = wrap_angle(player_ball_direction)

        # Check if the returning hit is successful
        opponent_hit_success = check_hit_success(player_position=player_positions[self.opponent_id], ball_position=player_ball_position
        )
        #print(f"Opponent hit success: {opponent_hit_success}")
        #print(f"Opponent hit type: {opponent_hit_type}")
        #print(f"Opponent hit position: {player_positions[self.opponent_id]}")
        #print(f"Ball position: {player_ball_position}")

        player_next_state = State(
            player_positions=player_positions,
            hitter_hit_type=hit_type,
            ball_position=ball_position,
            ball_direction=ball_direction,
        )

        if not opponent_hit_success and player_hit_success:
            # The opponent loses the point
            reward = 10
            done = True
        #print("Reward: ", reward)
        self.last_state = player_next_state
        return convert_state_to_output(player_next_state, self.player_id), reward, done, truncated, {}
    
    def render(self):
        """ 
        called to render the environment, it returns nothing
        """
        pass

    def close(self):
        """ 
        called to close the environment, it returns nothing
        """
        pass



if __name__ == "__main__":
    env = TennisEnv()
    check_env(env, warn=True)

    obs, _ = env.reset()

    for step in range(100):
        print(f"Step {step+1}")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"action: {action}")
        print(f"obs: {obs}", f"reward: {reward}", f"terminated: {terminated}", sep="\n")

        if terminated or truncated:
            print("Done")
            break
