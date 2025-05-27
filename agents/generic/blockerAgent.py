from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule, COORDS
from Sequence.sequence_utils import EMPTY
from copy import deepcopy

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)

    def SelectAction(self, actions, state):
        board = state.board.chips
        my_colour = state.agents[self.id].colour
        opp_colour = state.agents[self.id].opp_colour

        # === Step 1: Use one-eyed jack to remove opponent 4-in-a-row ===
        one_eye_jacks = [card for card in state.agents[self.id].hand if card in ['jh', 'js']]
        if one_eye_jacks:
            for r in range(10):
                for c in range(10):
                    if board[r][c] == opp_colour:
                        for draft in state.board.draft:
                            action = {
                                'type': 'remove',
                                'coords': (r, c),
                                'play_card': one_eye_jacks[0],
                                'draft_card': draft
                            }
                            if action in actions:
                                return action

        # === Step 2: Block opponent 3–4 alignment if possible ===
        for action in actions:
            if action['type'] != 'place' or not action['coords']:
                continue
            r, c = action['coords']
            sim_board = deepcopy(board)
            sim_board[r][c] = my_colour  # simulate placement

            # Check if this interrupts a likely 4-chip alignment
            if self.blocks_opponent(sim_board, r, c, opp_colour):
                return action

        # === Step 3: Fallback: center-first or random ===
        center = [(4, 4), (4, 5), (5, 4), (5, 5)]
        for a in actions:
            if a.get('coords') in center:
                return a

        return actions[0]  # fallback

    def blocks_opponent(self, board, r, c, opp_colour):
        # Count aligned opponent chips in 4 directions
        dirs = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in dirs:
            count = 1
            for step in range(1, 5):
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < 10 and 0 <= nc < 10 and board[nr][nc] == opp_colour:
                    count += 1
                else:
                    break
            for step in range(1, 5):
                nr, nc = r - dr * step, c - dc * step
                if 0 <= nr < 10 and 0 <= nc < 10 and board[nr][nc] == opp_colour:
                    count += 1
                else:
                    break
            if count >= 4:
                return True
        return False
