
# Complete, integrated Sequence agent with:
# - Immediate win detection
# - Two-step GBFS
# - 6-in-a-row bonus
# - Fork detection
# - Smart discard logic (dead cards, low-value discard)
# - Two-dead-card trade simulation if 2+ dead cards found

from template import Agent
from Sequence.sequence_model import SequenceGameRule as GameRule
import time, random
from copy import deepcopy
from Sequence.sequence_model import COORDS
from Sequence.sequence_utils import EMPTY

MAX_THINK_TIME = 0.95
SAFETY_BUFFER = 0.02
CENTER_COORDS = [(4, 4), (4, 5), (5, 4), (5, 5)]

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.rule = GameRule(2)

    def isWinningMove(self, state, action, agent_id):
        if action['type'] != 'place' or action['coords'] is None:
            return False
        r, c = action['coords']
        temp_state = deepcopy(state)
        board = temp_state.board.chips
        agent = temp_state.agents[agent_id]
        board[r][c] = state.agents[agent_id].colour
        if all(board[x][y] == agent.colour for x, y in CENTER_COORDS):
            return True
        result, _ = self.rule.checkSeq(board, agent, (r, c))
        return result and result['num_seq'] >= 1

    def FindImmediateWin(self, actions, state, agent_id):
        for action in actions:
            if self.isWinningMove(state, action, agent_id):
                return action
        return None

    def SelectAction(self, actions, game_state):
        # Two-dead-card trade rule
        dead_cards = [card for card in game_state.agents[self.id].hand
                      if self.is_dead_card(card, game_state.board.chips,
                      game_state.agents[self.id].colour)]
        if len(dead_cards) >= 2:
            # Simulate trade
            traded_card = random.choice(
                [c for c in self.rule.full_deck if c not in game_state.agents[self.id].hand])
            hand = game_state.agents[self.id].hand.copy()
            hand.remove(dead_cards[0])
            hand.append(traded_card)
            updated_actions = self.GeneratePlacingActions(game_state.board.chips, hand,
                                                          game_state.board.draft, game_state)
            return self.TwoStepLookaheadSearch(updated_actions, game_state)

        winning_move = self.FindImmediateWin(actions, game_state, self.id)
        if winning_move:
            return winning_move
        return self.TwoStepLookaheadSearch(actions, game_state)

    def SimulatedBoard(self, state, action, agent_id):
        board = deepcopy(state.board.chips)
        if action['type'] == 'place' and action['coords']:
            r, c = action['coords']
            board[r][c] = state.agents[agent_id].colour
        return board

    def CountAlignedChips(self, board, row, col, d_row, d_col, player_colour):
        open_ends = 0
        fwd_count = bwd_count = 0
        for step in range(1, 5):
            r, c = row + d_row * step, col + d_col * step
            if 0 <= r < 10 and 0 <= c < 10:
                chip = board[r][c]
                if chip == player_colour:
                    fwd_count += 1
                elif chip == EMPTY:
                    open_ends += 1; break
                else: break
        for step in range(1, 5):
            r, c = row - d_row * step, col - d_col * step
            if 0 <= r < 10 and 0 <= c < 10:
                chip = board[r][c]
                if chip == player_colour:
                    bwd_count += 1
                elif chip == EMPTY:
                    open_ends += 1; break
                else: break
        return 1 + fwd_count + bwd_count, open_ends

    def is_dead_card(self, card, board, colour):
        if card in ['jd', 'jc', 'jh', 'js']:
            return False
        positions = COORDS.get(card, [])
        return all(board[r][c] != EMPTY for r, c in positions)

    def HeuristicBoard(self, board, coords, state, agent_id):
        if coords is None:
            return 0
        r, c = coords
        colour = state.agents[agent_id].colour
        score = max(0, 5 - abs(r - 4.5) - abs(c - 4.5)) * 1.5
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        fork_dirs = 0
        for d_row, d_col in directions:
            aligned, open_ends = self.CountAlignedChips(board, r, c, d_row, d_col, colour)
            if aligned >= 5: score += 200
            elif aligned == 4 and open_ends >= 1: score += 90
            elif aligned == 3 and open_ends == 2: score += 50
            elif aligned == 2 and open_ends == 2: score += 20
            else: score += 2
            if aligned >= 3 and open_ends >= 1:
                fork_dirs += 1
        if fork_dirs >= 2:
            score += (fork_dirs - 1) * 10
        return score

    def GeneratePlacingActions(self, board, hand, draft, state):
        actions = []
        fallback_limit = 12
        card_targets = set()
        for card in hand:
            if card not in ['jd', 'jc']:
                for pos in COORDS.get(card, []):
                    if board[pos[0]][pos[1]] == EMPTY:
                        card_targets.add(pos)

        for card in hand:
            if card in ['jd', 'jc']:
                jack_moves = [(r, c) for r in range(10) for c in range(10)
                              if board[r][c] == EMPTY and (r, c) not in card_targets]
                jack_moves.sort(key=lambda pos: abs(pos[0] - 4.5) + abs(pos[1] - 4.5))
                for (r, c) in jack_moves[:fallback_limit]:
                    for d in draft:
                        actions.append({'type': 'place', 'coords': (r, c), 'play_card': card, 'draft_card': d})
            else:
                added = set()
                for r, c in COORDS.get(card, []):
                    if board[r][c] == EMPTY and (r, c) not in added:
                        added.add((r, c))
                        for d in draft:
                            actions.append({'type': 'place', 'coords': (r, c), 'play_card': card, 'draft_card': d})

        colour = state.agents[self.id].colour
        dead_cards = [card for card in hand if self.is_dead_card(card, board, colour)]
        if dead_cards:
            for d in draft:
                actions.append({'type': 'discard', 'play_card': dead_cards[0], 'draft_card': d})
        else:
            min_score, worst_card = float('inf'), None
            for card in hand:
                if card in ['jd', 'jc', 'jh', 'js']:
                    continue
                max_card_score = max(
                    self.HeuristicBoard(board, (r, c), state, self.id)
                    for r, c in COORDS.get(card, [])
                    if board[r][c] == EMPTY
                ) if any(board[r][c] == EMPTY for r, c in COORDS.get(card, [])) else -1
                if max_card_score < min_score:
                    min_score = max_card_score
                    worst_card = card
            if min_score < 10 and worst_card:
                for d in draft:
                    actions.append({'type': 'discard', 'play_card': worst_card, 'draft_card': d})

        return actions

    def TwoStepLookaheadSearch(self, actions, state):
        start_time = time.perf_counter()
        best_score, best_action = float('-inf'), None
        agent_id = self.id
        original_hand = state.agents[agent_id].hand
        original_draft = state.board.draft
        full_deck = [r + s for r in '23456789tjqka' for s in 'dchs'] * 2

        for a1 in actions:
            if time.perf_counter() - start_time > MAX_THINK_TIME - SAFETY_BUFFER:
                return random.choice(actions)
            if a1['type'] not in ['place', 'discard']:
                continue
            board1 = self.SimulatedBoard(state, a1, agent_id)
            score1 = self.HeuristicBoard(board1, a1.get('coords'), state, agent_id)
            new_hand = original_hand.copy()
            if a1['play_card'] in new_hand:
                new_hand.remove(a1['play_card'])
            new_hand.append(a1['draft_card'])
            new_draft = original_draft.copy()
            if a1['draft_card'] in new_draft:
                new_draft.remove(a1['draft_card'])
            seen_cards = set(original_hand + original_draft)
            available = [c for c in full_deck if c not in seen_cards]
            if available:
                new_draft.append(random.choice(available))
            second_actions = self.GeneratePlacingActions(board1, new_hand, new_draft, state)
            best_future = 0
            for a2 in second_actions:
                if time.perf_counter() - start_time > MAX_THINK_TIME - SAFETY_BUFFER:
                    break
                score2 = self.HeuristicBoard(board1, a2.get('coords'), state, agent_id)
                best_future = max(best_future, score2)
            total_score = score1 + best_future
            if total_score > best_score:
                best_score = total_score
                best_action = a1

        return best_action if best_action else random.choice(actions)
