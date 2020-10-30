"""
Created on 26 oct. 12:02 2020

@author: HaroldKS
"""
from core import BoardEnv
from core import Board
from seega import SeegaAction
from seega import SeegaState
from seega import SeegaRules


class SeegaEnv(BoardEnv):

    def __init__(self, board_shape, players, allowed_time, first_player=-1, boring_limit=200): # TODO: Remove player object
        self.players = players
        self.board_shape = board_shape
        self.allowed_time = allowed_time
        self.game_phases = [1, 2]
        self.rewarding_move = False
        self.done = False
        self.first_player = first_player
        self.just_stop = boring_limit
        self._reset()

    def reset(self):
        self._reset()

    def _reset(self):
        self.board = Board(self.board_shape, max_per_cell=1)
        self.phase = self.game_phases[0]
        self.state = SeegaState(board=self.board, next_player=self.first_player, boring_limit=self.just_stop,
                                game_phase=self.phase)
        self.current_player = self.first_player

    def step(self, action):
        """Plays one step of the game. Takes an action and perform in the environment.

        Args:
            action (Action): An action containing the move from a player.

        Returns:
            bool: Dependent on the validity of the action will return True if the was was performed False if not.
        """
        assert isinstance(action, SeegaAction), "action has to be an Action class object"
        result = SeegaRules.act(self.state, action, self.current_player)
        if isinstance(result, bool):
            return False
        else:
            self.state, self.done = result
            self.current_player = self.state.get_next_player()
            return True

    def render(self):
        """Gives the current state of the environnement

        Returns:
            (state, done): The state and the game status
        """
        return self.state, self.done

    def get_player_info(self, player):
        return self.state.get_player_info(player)

    def get_state(self):
        return self.state

    def is_end_game(self):
        return self.done