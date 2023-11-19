from typing import List, NamedTuple


HIT_TYPES = [
    "forehand_slice",
    "forehand_topspin",
    "forehand_return",
    "backhand_slice",
    "backhand_topspin",
    "backhand_return",
    "backhand_volley",
]


POSITIONS = [
    "TL",
    "TR",
    "BL",
    "BR",
]


class State(NamedTuple):
    """A class used to represent a state of the tennis court.

    player_positions: List[str]
        The positions of the two players at the current timestep.
    hitter_hit_type: str
        How the hitter hits the ball. Note that at each timestep,
        the player that will take action is the receiver, not the
        player.
    ball_position: str
        The position of the tennis ball.
    """
    player_positions: List[str]
    hitter_hit_type: str
    ball_position: str

class Action(NamedTuple):
    """A class used to represent an action.
    
    hit_type: str
        How the player hits the ball.
    player_movement: str
        The destination the player attempts to go to.
    """
    hit_type: str
    player_movement: str