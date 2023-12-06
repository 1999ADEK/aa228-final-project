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
# TODO: model the distributions of returns differently
DIR_CHANGE_LOOKUP_TABLE_DJOKOVIC = {
    "forehand_serve":   (94.2420821140381,   9.785480383292212),
    "forehand_slice":   (164.4665546322629,  16.02752807290332),
    "forehand_topspin": (169.93025509790778, 24.009030617197666),
    "forehand_return":  (71.8612373555223,   86.40281254294233),
    "backhand_slice":   (184.9273043652371,  12.395057410735312),
    "backhand_topspin": (180.36870683620108, 44.63281044368173),
    "backhand_return":  (139.70907346396416, 84.628870006565),
    "backhand_volley":  (189.29977089966653, 5.9040864208681185),
}

DIR_CHANGE_LOOKUP_TABLE_NADAL = {
    "forehand_serve":   (89.02934760971407,  9.932171255137026),
    "forehand_slice":   (175.08623175454207, 12.175953063846165),
    "forehand_topspin": (180.85918357811315, 37.62348309739396),
    "forehand_return":  (91.84638291788094,  93.1248672008286),
    "backhand_slice":   (169.74230324483685, 15.146026240544616),
    "backhand_topspin": (166.64443659556468, 26.554311944724848),
    "backhand_return":  (106.09981261159317, 80.2898965037309),
    "backhand_volley":  (168.47587585685906, 17.514677712715383),
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