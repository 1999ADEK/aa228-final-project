from utils import HIT_TYPES, POSITIONS, State, Action


# (player's position, ball position, opponent hit type) + one state for end of game
NUM_STATES = len(POSITIONS) * len(POSITIONS) * len(HIT_TYPES) + 1

# (player's new position, ball hit type)
NUM_ACTIONS = len(POSITIONS) * len(HIT_TYPES)

def get_state_index(player_pos, ball_pos, hit_type):
    player_pos_idx = POSITIONS.index(player_pos)
    ball_pos_idx = POSITIONS.index(ball_pos)
    hit_type_idx = HIT_TYPES.index(hit_type)
    return player_pos_idx * len(POSITIONS) * len(HIT_TYPES) + ball_pos_idx * len(HIT_TYPES) + hit_type_idx



def get_action_index(receive_pos, receive_type):
    pos_idx = POSITIONS.index(receive_pos)
    hit_type_idx = HIT_TYPES.index(receive_type)
    return pos_idx * len(HIT_TYPES) + hit_type_idx


def action_idx_to_action(idx):
    receive_pos = idx // len(HIT_TYPES)
    receive_type = idx % len(HIT_TYPES)

    return Action(
        hit_type=HIT_TYPES[receive_type], 
        player_movement=POSITIONS[receive_pos]
    )
