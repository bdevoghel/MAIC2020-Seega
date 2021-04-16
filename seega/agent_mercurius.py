from core.player import Player
from core.board import Board
from seega.seega_rules import SeegaRules
from seega.seega_actions import SeegaAction, SeegaActionType
from seega_state import SeegaState

from copy import deepcopy
from time import time

import numpy as np

inf = float("inf")
absolute_max_depth = 30  # TODO remove (was for debugging)


# TODO
""" NOTES
 - if several deepening iterations return same action, immediatly commit and do not go deeper
 - if action results in sufficient evaluation score gain, immediatly commit
 - if winning state is encountered, immediatly commit 
 - remove prints
 - add map from state to final evaluation
 
 cache :     
    from functools import lru_cache
    @lru_cache(maxsize=int(1e6))  # TODO determine best size
    successors.cache_info()

"""


class State(SeegaState):
    def __repr__(self):
        return str(self)
        # return f"<State({self.phase}, {self.get_next_player()}, {self.board.get_board_state()})>"

    def __str__(self):  # TODO optimize ?
        def display_black(string):
            string += f"   o : {(self.in_hand[-1] if self.phase == 1 else self.score[-1]):>2}"
            return string

        def display_green(string):
            string += f"   + : {(self.in_hand[1] if self.phase == 1 else self.score[1]):>2}"
            return string

        def display_next(string):
            string += f"     {'v' if self.get_next_player() == 1 else '^'}"
            return string

        def display_board_with_info(board=self.board.get_json_board()):
            colorize_prev_move = True
            prev_move = self.get_latest_move()['action'].values() if self.get_latest_move() else []
            b = f"p{self.phase}-----"
            for i, l in enumerate(board[::-1]):
                b += '\n|'
                for j, c in enumerate(l):
                    if colorize_prev_move and (4-i, j) in prev_move:
                        b += '\033[44m'  # color background
                    b += ' ' if c == 'empty' else ('o' if c == 'black' else '+')
                    if colorize_prev_move and (4-i, j) in prev_move:
                        b += '\033[0m'
                b += '|'
                if i == int(len(board) / 2) -1:
                    b = display_black(b)
                elif i == int(len(board) / 2):
                    b = display_next(b)
                elif i == int(len(board) / 2) +1:
                    b = display_green(b)
            b += '\n-------'
            return b
        string = f"<State : {hash(self)} (after {self.get_latest_move()})\n{display_board_with_info()}>"
        return string

    def __eq__(self, other):
        return isinstance(other, SeegaState) \
               and self.phase == other.phase \
               and self.get_next_player() == other.get_next_player() \
               and np.array_equal(self.board.get_board_state(), other.board.get_board_state())

    def __hash__(self):
        return hash((self.phase, self.get_next_player(), str(self.board.get_board_state())))  # TODO efficient to get str of np.array ?


class AI(Player):

    max_nb_moves = 100  # empirical

    in_hand = 12
    score = 0
    name = "Mercurius"

    def __init__(self, color):
        super(AI, self).__init__(color)
        self.ME = color.value
        self.OTHER = -color.value

        self.max_time = None
        self.remaining_time = None
        self.typical_time = None

        self.move_nb = 0
        self.max_depth = 1

        self.repeat_boring_moves = False
        self.last_action = None

        self.cache_successors = {'hits': 0, 'misses': 0}

    def play(self, state, remaining_time):
        self.move_nb += 1
        state.__class__ = State
        print(f"\nPlayer {self.ME} is playing with {remaining_time} seconds remaining for move #{self.move_nb}")
        print(f"CacheInfo : "
              f"hits={self.cache_successors['hits']}, "
              f"misses={self.cache_successors['misses']}, "
              f"currsize={len(self.cache_successors) - 2}")
        print(f"{state} evaluation={self.evaluate(state):.2f}\n")

        # TODO remove obsolete since stuck player fix
        # if self.repeat_boring_moves:  # fast-forward to save time
        #     assert state.get_latest_player() == self.ME, \
        #         " - ERROR : May not repeat boring moves, latest player isn't self"
        #     print(" - PLAYING BOREDOM")
        #     return self.reverse_last_move(state)

        if self.max_time is None:
            self.max_time = remaining_time
            self.typical_time = remaining_time / self.max_nb_moves
        self.remaining_time = remaining_time

        possible_actions = SeegaRules.get_player_actions(state, self.color.value)
        if len(possible_actions) == 1:
            best_action = possible_actions[0]
        elif state.phase == 1:
            best_action = SeegaRules.random_play(state, self.ME)  # TODO play smart during phase 1
        else:  # phase == 2
            # TODO remove obsolete since stuck player fix
            # if self.can_start_self_play(state):
            #     best_action = self.make_self_play_move(state, fallback_function=self.iterative_deepening)
            best_action = self.iterative_deepening(state)

        print(f" - SELECTED ACTION : {best_action}")
        self.last_action = best_action
        return best_action

    def successors(self, state: SeegaState):
        """
        The successors function must return (or yield) a list of
        pairs (a, s) in which a is the action played to reach the state s.
        """
        if state in self.cache_successors:
            self.cache_successors['hits'] += 1
            return self.cache_successors[state]

        next_player = state.get_next_player()
        possible_actions = SeegaRules.get_player_actions(state, next_player)
        succ = []
        for action in possible_actions:
            next_state, done = SeegaRules.make_move(deepcopy(state), action, next_player)
            succ.append((action, next_state))

        self.cache_successors['misses'] += 1
        self.cache_successors[state] = succ
        return succ

    def sort_successors(self, succ, maximize):
        return succ
        # TODO ordering : from best to worst for the player wh'os turn it is at the position
        # TODO follow principal variation https://youtu.be/zj3WsRyjkYM?t=815

    def sorted_successors(self, state, maximize=True):
        return self.sort_successors(self.successors(state), maximize)

    def cutoff(self, state, depth):
        """
        The cutoff function returns true if the alpha-beta/minimax search has to stop and false otherwise.
        """
        game_over = SeegaRules.is_end_game(state)
        max_depth = depth == self.max_depth or depth == absolute_max_depth
        cutoff = game_over or max_depth
        return cutoff
        # TODO cut states that are too bad

    # TODO to memoize
    def evaluate(self, state, details=False):
        """
        The evaluate function returns a value representing the utility function of the board.
        """
        # TODO necessity to make eval fucnction symmetric ??
        is_end = SeegaRules.is_end_game(state)
        captured = state.score[self.ME] - state.score[self.OTHER]
        other_is_stuck = state.phase == 2 and SeegaRules.is_player_stuck(state, self.OTHER)
        control_center = state.board.get_cell_color((2, 2)) == self.color

        # weights of liner interpolation of heuristics
        w_end = 100 * (-1 if captured < 0 else (0 if captured == 0 else 1))
        w_capt = 1
        w_stuck = 1
        w_center = 0.6

        random = .001 * np.random.random()  # random is to avoid always taking the first move when there is a draw
        if not details:
            return w_capt * captured + \
                   w_stuck * (1 if other_is_stuck else 0) + \
                   w_end * (1 if is_end else 0) + \
                   w_center * (1 if control_center else 0) + \
                   random
        else:
            return {'captured':         captured,
                    'other_is_stuck':   other_is_stuck,
                    'is_end':           is_end,
                    'control_center':   control_center}

    def compute_time_for_move(self):
        """
        Returns number of seconds allowed for the next move
        """
        return self.typical_time * .8  # TODO compute smart time

    def iterative_deepening(self, state):
        self.max_depth = 1  # reset
        time_for_move = self.compute_time_for_move()
        start_time = time()
        best_action = None

        while time() - start_time < time_for_move:
            print(f" - EXPLORING DEPTH {self.max_depth} ... | Current best action : {best_action}")
            action = minimax_search(state, self)

            if action is not None:
                best_action = action
            else:
                break
            self.max_depth += 1
            if self.max_depth > absolute_max_depth:
                break
        return best_action

    def can_start_self_play(self, state):
        """
        If state is winnable by repeating boring moves, set repeat_boring_moves to True and return True
        """
        # TODO do not do self_play if game can be won by exploring whole search tree till end (and win)
        other_actions = SeegaRules.get_player_all_cases_actions(state, self.OTHER)
        pieces_captured = self.evaluate(state, details=True)['captured']  # TODO optimize (no need to compute whole eval)
        if pieces_captured == state.MAX_SCORE - 1:  # other has only one piece left
            print("OTHER HAS ONLY ONE PIECE LEFT")
            self.repeat_boring_moves = True
            return True

        if len(other_actions) == 0 and pieces_captured > 0:  # opponent is blocked and has less captured pieces
            print("OTHER IS BLOCKED AND SELF HAS ADVANTAGE")
            self.repeat_boring_moves = True
            return True

        return False

    def make_self_play_move(self, state, fallback_function):
        for action, s in self.successors(state):
            if SeegaRules.is_player_stuck(s, self.OTHER):
                print(" - SELF PLAY MOVE FOUND")
                return action

        print(" - NO SELF PLAY MOVE FOUND, CONTINUE")
        self.repeat_boring_moves = False
        return fallback_function(state)

    def reverse_last_move(self, state):
        """
        Returns the move resulting in the previous state, allowing for (boring) self-play
        """
        # TODO use self.last_action instead of state last action (in case last move was performed by opponent)
        last_move = state.get_latest_move()
        next_move = SeegaAction(action_type=SeegaActionType.MOVE,
                                at=last_move['action']['to'],
                                to=last_move['action']['at'])
        return next_move

    """
    Specific methods for a Seega player (do not modify)
    """
    def set_score(self, new_score):
        self.score = new_score

    def update_player_infos(self, infos):
        self.in_hand = infos['in_hand']
        self.score = infos['score']

    def reset_player_informations(self):
        self.in_hand = 12
        self.score = 0
        # TODO to complete


def minimax_search(state, player):
    """
    Perform a MiniMax/AlphaBeta search and return the best action using Alpha-Beta pruning.

    Arguments:
    state -- initial state
    player -- a concrete instance of class AI implementing an Alpha-Beta player

    MiniMax and AlphaBeta algorithms.
    Adapted from:
        Author: Cyrille Dejemeppe <cyrille.dejemeppe@uclouvain.be>
        Copyright (C) 2014, Universite catholique de Louvain
        GNU General Public License <http://www.gnu.org/licenses/>
    """

    def max_value(state, alpha, beta, depth):
        if player.cutoff(state, depth):
            return player.evaluate(state), None
        max_eval = -inf
        action = None

        for a, s in player.sorted_successors(state, maximize=True):
            if s.get_latest_player() == s.get_next_player():  # next turn is for the same player
                val, _ = max_value(s, alpha, beta, depth + 1)
            else:                                             # next turn is for the other one
                val, _ = min_value(s, alpha, beta, depth + 1)

            if val > max_eval:
                max_eval = val
                action = a
                if beta <= val:
                    return val, a
                alpha = max(alpha, val)
        return max_eval, action

    def min_value(state, alpha, beta, depth):
        if player.cutoff(state, depth):
            return player.evaluate(state), None
        min_eval = inf
        action = None

        for a, s in player.sorted_successors(state, maximize=False):
            if s.get_latest_player() == s.get_next_player():  # next turn is for the same player
                val, _ = min_value(s, alpha, beta, depth + 1)
            else:                                             # next turn is for the other one
                val, _ = max_value(s, alpha, beta, depth + 1)

            if val < min_eval:
                min_eval = val
                action = a
                if val <= alpha:
                    return val, a
                beta = min(beta, val)
        return min_eval, action

    _, action = max_value(state, -inf, inf, 0)
    return action
