import sys

import numpy as np

sys.path.append("../")
from utils import Action, State


class BasePlayer(object):
    """A class used to represent a tennis player."""

    def __init__(self,
        player_id: int,
        first_serve_success_rate: float,
        second_serve_success_rate: float,
    ):
        self.player_id = player_id
        self.first_serve_success_rate = first_serve_success_rate
        self.second_serve_success_rate = second_serve_success_rate

    def check_serve_success(self, is_first_serve: bool = True) -> bool:
        """Determine if a serve is successful.

        Parameters
        ----------
        is_first_serve: bool
            Whether this is the first or second serve.
        """
        if is_first_serve:
            success_rate = self.first_serve_success_rate
        else:
            success_rate = self.second_serve_success_rate
        return np.random.uniform() < success_rate

    def update_state(self, current_state: State, action: Action) -> State:
        """Updates the state based on current state and action.

        Parameters
        ----------
        current_state: State
            The current state of the tennis court.
        action: Action
            The action the player takes.

        Returns
        -------
        next_state: State
            The next state of the tennis court.
        """
        raise NotImplementedError

    def choose_action(self, state: State) -> Action:
        """Chooses an action for the player.

        Parameters
        ----------
        state: State
            The current state of the tennis court.

        Returns
        -------
        action: Action
            The action the player takes.
        """
        raise NotImplementedError