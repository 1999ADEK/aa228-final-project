import sys
from typing import Dict

import numpy as np
import pickle

sys.path.append("../")
from .base_player import BasePlayer
from utils import Action, State, HIT_TYPES, POSITIONS
from q_learning_utils import get_state_index, get_action_index, action_idx_to_action, NUM_ACTIONS, NUM_STATES


class OnlineQLearningPlayer(BasePlayer):
    """A class used to represent the default tennis player."""

    def __init__(self,
        player_id: int,
        first_serve_success_rate: float,
        second_serve_success_rate: float,
        position_lookup_table: Dict[str, Dict[str, float]],
        q_learning_policy,
        is_train: bool,
        learning_rate = 0.3,
        gamma = 0.99,
        c = 1.0,
    ):
        super().__init__(
            player_id,
            first_serve_success_rate,
            second_serve_success_rate,
        )

        self.position_lookup_table = position_lookup_table
        self.q_learning_policy = pickle.load(open(q_learning_policy, 'rb')) if q_learning_policy != "" else None
        self.is_train = is_train

        self.Q_table = np.random.rand(NUM_STATES, NUM_ACTIONS)
        self.T = np.ones((NUM_STATES, NUM_ACTIONS, NUM_STATES))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.c = c
        # self.epsilon = 0.99

        self.current_state = None
        self.action = None
        self.reward = 0
    
    def update_policy(self, state0, action, reward, state1):
        # update policy only when training
        if not self.is_train:
            return
        
        # if current_state is not None:
        #     self.current_state = current_state
        #     self.action = action
        #     self.reward += reward
        
        # elif next_state is not None and self.current_state is not None:

        if self.current_state is None:
            self.current_state = state0
            self.action = action
            self.reward = reward
        # elif state0 is None:
        #     self.reward += reward
        else:
            next_state = state0
            # player_pos, ball_pos, hit_type
            current_state_idx = get_state_index(
                self.current_state.player_positions[self.player_id],
                self.current_state.ball_position,
                self.current_state.hitter_hit_type,
            )

            # receive_pos, receive_type
            current_action_idx = get_action_index(
                self.action.player_movement,
                self.action.hit_type,
            )

            next_state_idx = get_state_index(
                next_state.player_positions[self.player_id],
                next_state.ball_position,
                next_state.hitter_hit_type,
            )

            # print(self.current_state, self.action, next_state, reward)
            cur_val = self.Q_table[current_state_idx, current_action_idx] 
            update = (reward - self.reward) + 5 + self.gamma * np.max(self.Q_table[next_state_idx, :]) - cur_val
            self.Q_table[current_state_idx, current_action_idx] += self.learning_rate * update

            self.T[current_state_idx, current_action_idx, next_state_idx] += 1

            self.current_state = state0
            self.action = action
            self.reward = reward

            # print(update)
            # print(current_state_idx, current_action_idx, self.Q_table[current_state_idx, current_action_idx])


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

        if self.q_learning_policy is not None:
        
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
        
        else:
            state_idx = get_state_index(
                state.player_positions[self.player_id],
                state.ball_position,
                state.hitter_hit_type,
            )

            # UCB1
            Na = np.sum(self.T[state_idx], axis=1)
            logN = np.log(np.sum(Na))
            action_idx = np.argmax(self.Q_table[state_idx] + self.c * np.sqrt(logN / Na))

            # if np.random.rand() < self.epsilon:
            #     action_idx = np.random.randint(0, NUM_ACTIONS)
            #     self.epsilon *= 0.99
            # else:
            #     print(self.Q_table[state_idx])
            #     action_idx = np.argmax(self.Q_table[state_idx])


            return action_idx_to_action(action_idx)
    

    def get_best_policy(self):
        Na = np.sum(self.T, axis=2) # (num_states, num_actions)
        logN = np.log(np.sum(Na, axis=1)) # num_states

        policy =  np.argmax(self.Q_table + self.c * np.sqrt(logN.reshape((-1, 1)) / Na), axis=1)

        return policy
