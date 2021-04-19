from core.player import Player
from core import Color
from seega.seega_rules import SeegaRules
from seega.seega_actions import SeegaAction, SeegaActionType
from seega_state import SeegaState

from functools import lru_cache
from copy import deepcopy
from time import time

import numpy as np

inf = float("inf")


# TODO
""" NOTES
 - if several deepening iterations return same action, immediatly commit and do not go deeper
 - if action results in sufficient evaluation score gain, immediatly commit
 - remove prints
"""


@lru_cache(maxsize=None)
def get_possible_actions(state, player_id):
    return SeegaRules.get_player_all_cases_actions(state, player_id)


@lru_cache(maxsize=None)
def is_end_game(state):
    return SeegaRules.is_end_game(state)


@lru_cache(maxsize=None)
def is_player_stuck(state, player_id):
    return SeegaRules.is_player_stuck(state, player_id)


@lru_cache(maxsize=None)
def get_opponent_neighbours(board, cell, player_id):
    return SeegaRules._get_opponent_neighbours(board, cell, player_id)


class State(SeegaState):
    def __repr__(self):
        return f"<State({self.phase}, {self.get_next_player()}, {self.board.get_board_state()})>"

    def __str__(self):
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
        return hash((self.phase, self.get_next_player(), self.board.get_board_state().tobytes()))
        # NOTE : board.tobytes() is the fastest way to get the content to hash for a np array
        #        timeit.timeit(lambda: hash(str(self.board.get_board_state())), number=int(1e4)))
        #        timeit.timeit(lambda: hash(self.board.get_board_state().tobytes()), number=int(1e4)))

    def get_symmetries(self, position):
        def horizontal_sym(pos):
            return pos[0], 4 - pos[1]

        def vertical_sym(pos):
            return 4 - pos[0], pos[1]

        def pure_sym(pos):
            return vertical_sym(horizontal_sym(pos))

        return pure_sym(position), horizontal_sym(position), vertical_sym(position)


class AI(Player):

    max_nb_moves = 100  # empirical
    winning_value = 1000

    in_hand = 12
    score = 0
    name = "Venus"

    def __init__(self, color):
        super(AI, self).__init__(color)
        self.ME = color.value
        self.ME_color = color
        self.OTHER = -color.value
        self.OTHER_color = Color(-color.value)

        self.max_time = None
        self.remaining_time = None
        self.typical_time = None

        self.move_nb = 0
        self.exploring_depth_limit = 1
        self.absolute_max_depth = 1000

        self.playing_barrier = False
        self.reached_win = False
        self.last_action = None
        self.explored_nodes = 0

        self.state_evaluations = dict()  # {state: (static_eval, dynamic_eval, dynamic_eval_depth)
        self.current_eval = 0

        self.board_preferences = np.array([[60, 50, 99, 50, 60],
                                           [50, 31,  1, 33, 50],
                                           [99,  1, -1,  1, 99],
                                           [50, 11,  1, 13, 50],
                                           [60, 50, 99, 50, 60]])[::-1]
        self.highest_values_indices = [np.unravel_index(i, self.board_preferences.shape)
                                       for i in np.argsort(self.board_preferences, axis=None)][::-1]

    def play(self, state, remaining_time):
        self.move_nb += 1
        state.__class__ = State
        print(f"\nPlayer {self.ME} is playing with {remaining_time} seconds remaining for move #{self.move_nb}")
        print(f"- Cache successors : {self.successors.cache_info()}")
        print(f"- Cache evaluate   : {self.static_evaluation.cache_info()}")
        print(f"{state} evaluation={self.evaluate(state):.2f}\n")

        if self.playing_barrier:  # fast-forward to save time
            print("PLAYING BARRIER - Reversing last move")
            return self.reverse_last_action()

        if self.max_time is None:
            self.max_time = remaining_time
            self.typical_time = remaining_time / self.max_nb_moves
        self.remaining_time = remaining_time
        self.current_eval = self.evaluate(state)

        possible_actions = get_possible_actions(state, self.ME)
        if len(possible_actions) == 1:
            best_action = possible_actions[0]
        elif state.phase == 1:
            # TODO determine which is best starting policy
            # best_action = SeegaRules.random_play(state, self.ME)
            # best_action = self.find_best_placement(state)
            best_action = self.find_symmetry_placement(state)
        else:  # phase == 2
            if self.barrier_exists(state):
                best_action = self.reverse_last_action()
            else:
                best_action = self.iterative_deepening(state)

        print(f"SELECTED ACTION : {best_action}")
        self.last_action = best_action
        return best_action

    @lru_cache(maxsize=None)
    def successors(self, state: SeegaState):
        """
        The successors function must return (or yield) a list of
        pairs (a, s) in which a is the action played to reach the state s.
        """
        next_player = state.get_next_player()
        possible_actions = get_possible_actions(state, next_player)
        succ = []
        for action in possible_actions:
            next_state, done = SeegaRules.make_move(deepcopy(state), action, next_player)
            succ.append((action, next_state))

        return succ

    def sort_successors(self, successors, maximize):
        sorted_successors = sorted(successors,
                                   key=lambda succ: self.evaluate(succ[1], static=False)
                                                    + (1 if maximize else -1)
                                                        * (self.winning_value if self.is_pv_move(succ) else 0),
                                   reverse=maximize)
        return sorted_successors

    def is_pv_move(self, successor):
        """
        Returns True if successor is on the Principal Variation
        (see https://en.wikipedia.org/wiki/Principal_variation_search), False otherwise
        """
        return False  # TODO , see https://youtu.be/zj3WsRyjkYM?t=815

    def sorted_successors(self, state, maximize):
        return self.sort_successors(self.successors(state), maximize)

    def cutoff(self, state, depth):
        """
        The cutoff function returns true if the alpha-beta/minimax search has to stop and false otherwise.
        """
        max_depth = depth == self.exploring_depth_limit or depth == self.absolute_max_depth
        cutoff = max_depth \
                 or is_end_game(state) \
                 or self.evaluate(state) < self.current_eval - 3  # too bad # TODO determine appropriate value
        return cutoff

    def evaluate(self, state, static=True, value=None, depth=None):
        """
        The evaluate function returns a value representing the utility function of the board.
        """
        if state not in self.state_evaluations:
            static = self.static_evaluation(state)
            self.state_evaluations[state] = (static, static if value is None else value, 0 if depth is None else depth)

        return self.state_evaluations[state][0 if static else 1]

    def set_deeper_evatuation(self, state, value, depth):  # TODO set in code to be able to use deeper evaluations
        if state not in self.state_evaluations:
            self.evaluate(state, value=value, depth=depth)

        current = self.state_evaluations[state]
        self.state_evaluations[state] = (current[0], value, depth)

    @lru_cache(maxsize=None)  # should not be used since evaluate() uses self.state_evaluations
    def static_evaluation(self, state, details=False):
        # boolean indicating if state is winnable by either player, regarding score (other winning possibility: boredom)
        is_win = max(state.score.values()) >= state.MAX_SCORE - 1
        # difference in number of captured pieces
        sum_captured = state.score[self.ME] - state.score[self.OTHER]
        # integer indicating which player is stuck (has no possible moves)
        other_is_stuck = 0
        if state.phase == 2:
            if is_player_stuck(state, self.OTHER):
                other_is_stuck = 1
            elif is_player_stuck(state, self.ME):
                other_is_stuck = -1
        # integer indicating which player controls the center
        control_center = state.board.get_cell_color((2, 2)).value * self.ME
        # ratio of possible moved for each player
        mobility = (1+len(get_possible_actions(state, self.ME))) \
                   / (1+len(get_possible_actions(state, self.OTHER)))
        # factor of density of the players' pieces
        # (1 is relatively dense, -1 is relatively spread out, 0 is similar density for both players)
        density = 0  # TODO
        # factor of how much risk has to be taken to gain advantage (e.g. to avoir losing to boring moves)
        risk = 0  # TODO
        # TODO add depth penalty (want good score early)

        # weights for liner interpolation of heuristics
        w_win = self.winning_value * np.sign(sum_captured)
        w_capt = 1
        w_stuck = 0.1
        w_center = 0.001
        w_mobility = -0.01
        w_density = 0
        w_risk = 0
        # TODO determine appropriate weights

        if not details:
            score = (w_win if is_win else 0) + \
                    w_capt * sum_captured + \
                    w_stuck * other_is_stuck + \
                    w_center * control_center + \
                    w_mobility * mobility + \
                    w_density * density + \
                    w_risk * risk
            if score > self.winning_value:
                self.reached_win = True
            return score
        else:
            return {'is_win':         is_win,
                    'sum_captured':   sum_captured,
                    'other_is_stuck': other_is_stuck,
                    'control_center': control_center,
                    'mobility':       mobility,
                    'density':        density,
                    'risk':           risk}

    def compute_time_for_move(self):
        """
        Returns number of seconds allowed for the next move
        NOTE : depth 3 takes around .3s
        """
        if self.remaining_time < self.max_time * 0.05:
            print(f"NOT MUCH TIME LEFT - Limitting to depth 3")
            self.absolute_max_depth = 3
            return .2

        moves_left = self.max_nb_moves - self.move_nb
        time_for_move = self.remaining_time / moves_left
        adjusted_time_for_move = time_for_move * (time_for_move/self.typical_time)
        print('\033[44m' + f"Move nb : {self.move_nb:<5} - Remaining time : {self.remaining_time:<8.4f} - Typical time : {self.typical_time:<8} - Time for move : {time_for_move:<8.4f} - Adjusted time for move : {adjusted_time_for_move:<8.4f}" + '\033[0m')
        return adjusted_time_for_move
        # TODO compute smart time : if repeating move : allow more in-depth search for longer vision

    def find_symmetry_placement(self, state):
        opponent_pieces = state.board.get_player_pieces_on_board(self.OTHER_color)
        print(opponent_pieces)
        symmetry_positions = [pos for symmetries in [state.get_symmetries(pos) for pos in opponent_pieces]
                              for pos in symmetries]
        # reorder symmetries (prefer pure symmetries)
        symmetry_positions = symmetry_positions[::3] + symmetry_positions[1::3] + symmetry_positions[2::3]

        return self.find_best_placement(state, from_indices=symmetry_positions)

    def find_best_placement(self, state, from_indices=[]):
        actions = get_possible_actions(state, self.ME)
        for index in (from_indices + self.highest_values_indices):  # explicit indices preference then heatmap
            for i, action in enumerate(actions):
                if index == action.action['to']:
                    return actions[i]

        assert False, f"NO VALID ACTION FOUND"

    def iterative_deepening(self, state):
        self.exploring_depth_limit = 1  # reset
        time_for_move = self.compute_time_for_move()
        start_time = time()
        best_action = None

        while time() - start_time < time_for_move and self.exploring_depth_limit <= self.absolute_max_depth:
            print(f" - EXPLORING DEPTH {self.exploring_depth_limit} ...",
                  end=(' ' if self.exploring_depth_limit != 1 else '\n'))
            self.explored_nodes = 0
            action = minimax_search(state, self)
            print(f"{(' ' * 25) if self.exploring_depth_limit == 1 else ''}"
                  f"| Explored {self.explored_nodes:>5} nodes "
                  f"| Used {time() - start_time:>.4f}s", end=' ')

            if action is not None:
                best_action = action
                print(f"| New best action : {best_action}")
            else:
                print(f"| No new action")
                break
            self.exploring_depth_limit += 1
            # if self.reached_win or self.exploring_depth_limit > absolute_max_depth:
            #     print(f" - END SEACH EARLY")
            #     break
        return best_action

    # def can_start_self_play(self, state):
    #     """
    #     If state is winnable by repeating boring moves, set repeat_boring_moves to True and return True
    #     """
    #     # TODO do not do self_play if game can be won by exploring whole search tree till end (and win)
    #     other_actions = get_possible_actions(state, self.OTHER)
    #     pieces_captured = self.static_evaluation(state, details=True)['captured']  # TODO optimize (no need to compute whole eval)
    #     if pieces_captured == state.MAX_SCORE - 1:  # other has only one piece left
    #         print("OTHER HAS ONLY ONE PIECE LEFT")
    #         self.repeat_boring_moves = True
    #         return True
    #
    #     if len(other_actions) == 0 and pieces_captured > 0:  # opponent is blocked and has less captured pieces
    #         print("OTHER IS BLOCKED AND SELF HAS ADVANTAGE")
    #         self.repeat_boring_moves = True
    #         return True
    #
    #     return False

    # def make_self_play_move(self, state, fallback_function):
    #     for action, s in self.successors(state):
    #         if is_player_stuck(s, self.OTHER):
    #             print(" - SELF PLAY MOVE FOUND")
    #             return action
    #
    #     print(" - NO SELF PLAY MOVE FOUND, CONTINUE")
    #     self.repeat_boring_moves = False
    #     return fallback_function(state)

    def barrier_exists(self, state):
        favorable_score = state.score[self.ME] - state.score[self.OTHER] >= 0
        if favorable_score:
            has_free_moving_piece = False
            if has_free_moving_piece:
                self.playing_barrier = True
                return True

        return False

    def reverse_last_action(self):
        """
        Returns the move that was played last time
        """
        last_action = self.last_action
        reversed_action = SeegaAction(action_type=SeegaActionType.MOVE,
                                      at=last_action['action']['to'],
                                      to=last_action['action']['at'])
        return reversed_action

    """
    Specific methods for a Seega player (do not modify)
    """
    def set_score(self, new_score):
        self.score = new_score

    def update_player_infos(self, infos):
        self.in_hand = infos['in_hand']
        self.score = infos['score']
        # NOTE : called outside timer loop

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
            player.explored_nodes += 1
            if player.exploring_depth_limit == 1:
                print(f"   - {str(a):<65} : eval={player.evaluate(s):.2f}")
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
            player.explored_nodes += 1
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
