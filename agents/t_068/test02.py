from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule
import time, random
from copy import deepcopy
from Sequence.sequence_model import COORDS
from Sequence.sequence_utils import EMPTY, RED, BLU, RED_SEQ, BLU_SEQ

MAX_THINK_TIME = 0.95
SAFETY_BUFFER = 0.02
CENTER_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)

    def SelectAction(self, actions, game_state):
        winning_move = self.findImmediateWin(actions, game_state, self.id)
        if winning_move:
            return winning_move

        opponent_colour = BLU if game_state.agents[self.id].colour == RED else RED
        threat_coords, threat_level = self.FindThreatToBlock(game_state.board.chips, opponent_colour)
        if threat_coords and threat_level >= 4:
            for a in actions:
                if a['type'] == 'place' and a['coords'] == threat_coords:
                    return a

        return self.TwoStepLookaheadSearch(actions, game_state)

    def findImmediateWin(self, actions, state, agent_id):
        for action in actions:
            if self.isWinningMove(state, action, agent_id):
                return action
        return None

    def isWinningMove(self, state, action, agent_id):
        if action['type'] != 'place' or action['coords'] is None:
            return False
        r, c = action['coords']
        temp_state = deepcopy(state)
        board = temp_state.board.chips
        agent = temp_state.agents[agent_id]
        board[r][c] = agent.colour
        if all(board[x][y] == agent.colour for x, y in CENTER_COORDS):
            return True
        result, _ = self.rule.checkSeq(board, agent, (r, c))
        return result and result['num_seq'] >= 1

    def TwoStepLookaheadSearch(self, actions, state):
        start_time = time.perf_counter()
        best_score = float('-inf')
        best_actions = []
        agent_id = self.id
        hand = state.agents[agent_id].hand
        draft = state.board.draft
        full_deck = [r + s for r in '23456789tjqka' for s in 'dchs'] * 2

        for a1 in actions:
            if time.perf_counter() - start_time > MAX_THINK_TIME - SAFETY_BUFFER:
                return random.choice(actions)
            if a1['type'] != 'place' or a1['coords'] is None:
                continue

            board1 = self.SimulatedBoard(state, a1, agent_id)
            score1 = self.HeuristicBoard(board1, a1['coords'], state, agent_id)

            new_hand = hand.copy()
            if a1['play_card'] in new_hand:
                new_hand.remove(a1['play_card'])
            new_hand.append(a1['draft_card'])

            new_draft = draft.copy()
            if a1['draft_card'] in new_draft:
                new_draft.remove(a1['draft_card'])

            seen_cards = set(hand + draft)
            available = [c for c in full_deck if c not in seen_cards]
            if available:
                new_draft.append(random.choice(available))

            second_actions = self.get_place_actions_on_board(board1, new_hand, new_draft)
            best_future = max(
                (self.HeuristicBoard(board1, a2['coords'], state, agent_id) for a2 in second_actions if a2['coords']),
                default=0
            )

            total_score = score1 + best_future
            if total_score > best_score:
                best_score = total_score
                best_actions = [a1]
            elif total_score == best_score:
                best_actions.append(a1)

        return random.choice(best_actions) if best_actions else random.choice(actions)

    def SimulatedBoard(self, state, action, agent_id):
        board = [row[:] for row in state.board.chips]
        if action['coords'] and action['type'] == 'place':
            r, c = action['coords']
            board[r][c] = state.agents[agent_id].colour
        return board

    def get_place_actions_on_board(self, board, hand, draft):
        actions = []
        fallback_limit = 12
        player_colour = RED if self.id % 2 == 0 else BLU
        opponent_colour = BLU if player_colour == RED else RED

        card_targets = set()
        for card in hand:
            if card not in ['jd', 'jc', 'jh', 'js']:
                for pos in COORDS.get(card, []):
                    if board[pos[0]][pos[1]] == EMPTY:
                        card_targets.add(pos)

        for card in hand:
            if card in ['jd', 'jc']:
                jack_moves = [
                    pos for pos in CENTER_COORDS
                    if board[pos[0]][pos[1]] == EMPTY and pos not in card_targets
                ]
                if not jack_moves:
                    jack_moves = [
                        (r, c) for r in range(10) for c in range(10)
                        if board[r][c] == EMPTY and (r, c) not in card_targets
                    ]
                    jack_moves.sort(key=lambda pos: abs(pos[0] - 4.5) + abs(pos[1] - 4.5))
                for (r, c) in jack_moves[:fallback_limit]:
                    for d in draft:
                        actions.append({'type': 'place', 'coords': (r, c), 'play_card': card, 'draft_card': d})
            elif card in ['jh', 'js']:
                threat_coords, threat_len = self.FindThreatToBlock(board, opponent_colour)
                if threat_coords and threat_len >= 3:
                    for d in draft:
                        actions.append({'type': 'remove', 'coords': threat_coords, 'play_card': card, 'draft_card': d})
            else:
                added = set()
                for r, c in COORDS.get(card, []):
                    if board[r][c] == EMPTY and (r, c) not in added:
                        added.add((r, c))
                        for d in draft:
                            actions.append({'type': 'place', 'coords': (r, c), 'play_card': card, 'draft_card': d})
        return actions

    def CountAlignedChips(self, board, row, col, d_row, d_col, player_colour):
        count, open_ends = 1, 0
        for step in range(1, 5):
            r, c = row + d_row * step, col + d_col * step
            if 0 <= r < 10 and 0 <= c < 10:
                if board[r][c] == player_colour:
                    count += 1
                elif board[r][c] == EMPTY:
                    open_ends += 1
                    break
                else:
                    break
        for step in range(1, 5):
            r, c = row - d_row * step, col - d_col * step
            if 0 <= r < 10 and 0 <= c < 10:
                if board[r][c] == player_colour:
                    count += 1
                elif board[r][c] == EMPTY:
                    open_ends += 1
                    break
                else:
                    break
        return count, open_ends

    def HeuristicBoard(self, board, coords, state, agent_id):
        if coords is None:
            return 0
        r, c = coords
        player_colour = state.agents[agent_id].colour
        opponent_colour = BLU if player_colour == RED else RED
        score = max(0, 5 - abs(r - 4.5) - abs(c - 4.5)) * 1.5
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for d_row, d_col in directions:
            aligned, open_ends = self.CountAlignedChips(board, r, c, d_row, d_col, player_colour)
            if aligned >= 5: score += 200
            elif aligned == 4 and open_ends >= 1: score += 90
            elif aligned == 3 and open_ends == 2: score += 50
            elif aligned == 2 and open_ends == 2: score += 20
            else: score += 2

        for d_row, d_col in directions:
            aligned, open_ends = self.CountAlignedChips(board, r, c, d_row, d_col, opponent_colour)
            if aligned >= 5: score += 300
            elif aligned == 4 and open_ends >= 1: score += 200
            elif aligned == 3 and open_ends == 2: score += 100
            elif aligned == 2 and open_ends == 2: score += 25
        return score

    def FindThreatToBlock(self, board, opponent_colour):
        max_threat = 0
        threat_coords = None
        for r in range(10):
            for c in range(10):
                if board[r][c] == EMPTY:
                    for d_row, d_col in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        aligned, open_ends = self.CountAlignedChips(board, r, c, d_row, d_col, opponent_colour)
                        if aligned >= 3 and open_ends > 0 and aligned > max_threat:
                            max_threat = aligned
                            threat_coords = (r, c)
        return threat_coords, max_threat
