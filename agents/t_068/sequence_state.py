import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from Sequence.sequence_model import SequenceGameRule, COORDS
from Sequence.sequence_utils import EMPTY

class SequenceState:
    def __init__(self):
        self.rule = SequenceGameRule(2)
        self.state = self.rule.initialGameState()
        self.current_player = 0

    def clone(self):
        new = SequenceState()
        new.state = self.rule.copyState(self.state)
        new.current_player = self.current_player
        return new

    def game_over(self):
        return self.rule.gameEnds()

    def get_winner(self):
        scores = [self.rule.calScore(self.state, i) for i in range(2)]
        return int(np.argmax(scores))

    def get_legal_actions(self):
        return self.rule.getLegalActions(self.state, self.current_player)

    def apply_action(self, action):
        self.state = self.rule.generateSuccessor(self.state, action, self.current_player)
        self.current_player = 1 - self.current_player

    def encode(self):
        board = np.array(self.state.board.chips)  # Convert to numpy array
        player_0_colour = self.state.agents[0].colour
        player_1_colour = self.state.agents[1].colour

        p0 = (board == player_0_colour).astype(int).flatten()
        p1 = (board == player_1_colour).astype(int).flatten()
        empty = (board == EMPTY).astype(int).flatten()
        return np.concatenate([p0, p1, empty])

    def action_to_index(self, action):
        if action['type'] != 'place' or action['coords'] is None:
            return 0
        r, c = action['coords']
        return r * 10 + c

    def index_to_action(self, index):
        r, c = index // 10, index % 10
        for card, coords in COORDS.items():
            if (r, c) in coords:
                return {'type': 'place', 'coords': (r, c), 'play_card': card, 'draft_card': card}
        return {'type': 'place', 'coords': (r, c), 'play_card': 'jd', 'draft_card': 'jd'}

    @staticmethod
    def get_action_space_size():
        return 100
