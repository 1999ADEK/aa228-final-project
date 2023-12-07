from typing import List, NamedTuple

import numpy as np


HIT_TYPES = [
    "forehand_slice",
    "forehand_topspin",
    "forehand_return",
    "forehand_serve",
    "backhand_slice",
    "backhand_topspin",
    "backhand_return",
    "backhand_volley",
]

RECEIVE_HIT_TYPES = [
    "forehand_slice",
    "forehand_topspin",
    "forehand_return",
    "backhand_slice",
    "backhand_topspin",
    "backhand_return",
    "backhand_volley",
]


# Define tennis court dimensions
# (Below is half of a tennis court only)
#
#   Center Service Line
#           ↓
# --------------------- ← Net
# | |       |       | |
# | |       |       | |
# | |       |       | |
# | |       |       | |
# --------------------- ← Service Line
# | |               | |
# | |               | |
# | |               | |
# | |               | |
# --------------------- ← Baseline
#   ↑               ↑
# Sideline        Sideline

# Vertical lines
POS_CENTER_SERVICE_LINE = 5.485
POS_SINGLES_SIDELINE_LEFT = 1.37
POS_SINGLES_SIDELINE_RIGHT = 9.6
# Horizontal lines
POS_NET = 11.885
POS_SERVICE_LINE = 5.485
POS_BASELINE = 0
# Bounding box
COURT_LENGTH = 23.77
COURT_WIDTH = 10.97
COURT_BBOX = np.array([COURT_WIDTH, COURT_LENGTH])


# Store the stats of the distance (instead of speed) of each ball hit type
# Each tuple is (mean, std) of distance
DISTANCE_LOOKUP_TABLE_DJOKOVIC = {
    "forehand_serve":   (26.43914459254912,  2.0325763325429778),
    "forehand_slice":   (26.6167899956595,   2.0670607942100996),
    "forehand_topspin": (26.695330586864483, 2.0294712788253),
    "forehand_return":  (23.70826155243442,  2.847862777005586),
    "backhand_slice":   (21.14247842289603,  2.9087178396518665),
    "backhand_topspin": (25.400779051022457, 3.166209529518272),
    "backhand_return":  (24.758324729862796, 1.3601504336812549),
    "backhand_volley":  (14.506932725183782, 4.928644462763079),
}

DISTANCE_LOOKUP_TABLE_NADAL = {
    "forehand_serve":   (24.958712054744662, 2.1907641682315617),
    "forehand_slice":   (22.904350469116547, 5.502197705270183),
    "forehand_topspin": (26.4056299600696,   3.136237119762917),
    "forehand_return":  (25.54441490028956,  1.6368834417468072),
    "backhand_slice":   (23.580386079617423, 4.384463582653366),
    "backhand_topspin": (26.022891940740163, 2.968785994081775),
    "backhand_return":  (25.510577733652777, 2.6426229088003055),
    "backhand_volley":  (12.944648933558359, 2.620575125220877),
}


# Store the stats of the change of direction of each ball hit type
# Each tuple is (mean, std) of theta
DIR_CHANGE_LOOKUP_TABLE_DJOKOVIC = {
    "forehand_serve_BL": (84.7609521921382,   3.6728556716820324),
    "forehand_serve_BR": (102.80568333381865, 3.9411038960983955),
    "forehand_slice":    (160.1336132600564,  10.17331111384927),
    "forehand_topspin":  (162.98446269675492, 10.936658778350614),
    "forehand_return":   (164.38970185833824, 6.0839754820840115),
    "backhand_slice":    (168.94215504380162, 7.459211851649383),
    "backhand_topspin":  (163.25766571689394, 8.984303687582587),
    "backhand_return":   (161.52502237036225, 8.079587616336017),
    "backhand_volley":   (170.70022910033347, 5.9040864208681185),
}

DIR_CHANGE_LOOKUP_TABLE_NADAL = {
    "forehand_serve_BL": (97.32311570444048,  3.388529183939932),
    "forehand_serve_BR": (89.02934760971407,  9.932171255137026),
    "forehand_slice":    (169.0625103963013,  7.264315009141505),
    "forehand_topspin":  (163.45810701642745, 9.068513875551627),
    "forehand_return":   (159.6367988303698,  3.4090026603328747),
    "backhand_slice":    (165.30476329660706, 10.89368954172109),
    "backhand_topspin":  (164.22586112480013, 11.066443253826453),
    "backhand_return":   (161.44634334320196, 12.996293405236504),
    "backhand_volley":   (162.48532228728465, 11.524124143140957),
}


class State(NamedTuple):
    """A class used to represent a state of the tennis court.

    player_positions: np.ndarray of shape (2, 2)
        The positions of the two players at the current timestep.
        player_positions[i] is the position of player i.
    hitter_hit_type: str
        How the hitter hits the ball. Note that at each timestep,
        the player that will take action is the receiver, not the
        player.
    ball_position: np.ndarray of shape (2,)
        The position of the tennis ball.
    ball_direction: float
        The direction of the tennis ball, represented by the direction
        angle.
        Note that this is the direction of the ball BEFORE the hitter
        hits the ball.
    """
    player_positions: np.ndarray
    hitter_hit_type: str
    ball_position: np.ndarray
    ball_direction: float

class Action(NamedTuple):
    """A class used to represent an action.
    
    hit_type: str
        How the player hits the ball.
    player_movement: np.ndarray of shape (2,)
        The destination the player attempts to go to.
    """
    hit_type: str
    player_movement: np.ndarray